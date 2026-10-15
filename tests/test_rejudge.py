"""Sensitivity re-judging (eval/rejudge.py): no LLM, no corpus. Run: python -m pytest tests"""
import json
from types import SimpleNamespace as NS

import pytest

import config
import src.llm
import src.retrieve
import src.verify
from eval import rejudge
from src.retrieve import Passage
from src.verify import ClaimVerdict


class FakeEmbedder:
    """Bag-of-letters vectors: enough to rank passages by overlap."""
    def _vec(self, t):
        import numpy as np
        v = np.zeros(26)
        for ch in t.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        return v

    def encode_queries(self, texts):
        return [self._vec(t) for t in texts]

    def encode_passages(self, texts):
        return [self._vec(t) for t in texts]

PASSAGES = [Passage(f"c{i}", t, "", "", 0, 0, "s", [], 1.0) for i, t in enumerate(
    ["walking reduced stress", "zzz qqq xxx", "sleep quality improved", "kkk jjj", "mood was better"])]


def test_needs_rejudge():
    assert rejudge.needs_rejudge("partially_supported", "answer_detail")
    assert not rejudge.needs_rejudge("supported", "answer_detail")
    assert rejudge.needs_rejudge("disputed", "trap_detail")
    assert not rejudge.needs_rejudge("partially_supported", "trap_detail")   # safe either way


def test_already_judged_baseline_is_the_two_closest():
    claim = {"statement": "walking reduced stress", "judged_against": ["c0"]}
    seen = rejudge.already_judged("baseline", claim, PASSAGES, FakeEmbedder())
    assert len(seen) == 2 and "c0" in seen


def test_already_judged_grounded_is_the_cited_passage():
    claim = {"statement": "x", "cited": ["c3"], "judged_against": ["c3"]}
    assert rejudge.already_judged("grounded", claim, PASSAGES, FakeEmbedder()) == {"c3"}


def test_better_keeps_the_higher_label():
    assert rejudge.better("unsupported", "supported") == "supported"
    assert rejudge.better("partially_supported", "unsupported") == "partially_supported"


def test_mcnemar():
    assert rejudge.mcnemar([True] * 5, [False] * 5) == {"a_only": 5, "b_only": 0, "p": 0.0625}
    assert rejudge.mcnemar([True, False], [True, False])["p"] == 1.0


def claim(statement, label, cited=("c0",)):
    return {"statement": statement, "label": label, "cited": list(cited),
            "judged_against": list(cited)}


def results(answer=(), traps=()):
    return {"grounding": [{"answer_detail": list(answer), "trap_detail": list(traps)}]}


class FakeVerifier:
    calls = []

    def __init__(self, llm=None):
        pass

    def describe(self):
        return "judges=[fake]"

    def verify_statement(self, statement, passages):
        FakeVerifier.calls.append((statement, [p.chunk_id for p in passages]))
        ok = "sleep" in statement.lower()
        return ClaimVerdict(statement, [passages[0].chunk_id],
                            "supported" if ok else "unsupported", "fake", "llm")


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeVerifier.calls = []
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(src.verify, "Verifier", FakeVerifier)
    monkeypatch.setattr(src.llm, "LLMClient", lambda: None)
    monkeypatch.setattr(src.retrieve, "get_retriever",
                        lambda s: NS(search=lambda q: PASSAGES, embedder=FakeEmbedder()))
    import eval.run_eval
    monkeypatch.setattr(eval.run_eval, "load_golden", lambda: {
        "retrieval": [{"id": "R1", "question": "q1"}],
        "traps": [{"id": "T1", "question": "t1"}, {"id": "B1", "question": "b1"}]})

    base = results(
        answer=[{"id": "R1", "refused": False, "parse_failed": False, "claims": [
            claim("Walking reduced stress.", "supported"),
            claim("Sleep quality improved a lot.", "unsupported")]}],
        traps=[{"id": "T1", "type": "far_absent", "refused": False, "parse_failed": False,
                "claims": [claim("Mood vs urban was worse.", "unsupported"),
                           claim("Something partial.", "partially_supported")]},
               {"id": "B1", "type": "hypothesis_bait", "refused": False, "parse_failed": False,
                "claims": [claim("Sleep was improved by design.", "disputed")]}])
    grounded = results(answer=[{"id": "R1", "refused": False, "parse_failed": False,
                                "claims": [claim("Walking reduced stress.", "supported")]}])
    g_traps = results(traps=[{"id": "T1", "type": "far_absent", "refused": True,
                              "parse_failed": False, "claims": []}])
    g_bait = results(traps=[{"id": "B1", "type": "hypothesis_bait", "refused": True,
                             "parse_failed": False, "claims": []}])
    for name, data in [("eval_results_baseline_rescored.json", base),
                       ("eval_results_phi4v3_fix.json", grounded),
                       ("eval_results_traps_postfix.json", g_traps),
                       ("eval_results_bait_grounded_v1.json", g_bait)]:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_end_to_end_and_resume(env):
    rejudge.main([])
    # Re-judged: the unsupported answerable sentence and the two trap claims
    # that affect safety; the supported and the partial trap claim are left.
    assert [s for s, _ in FakeVerifier.calls] == [
        "Sleep quality improved a lot.", "Mood vs urban was worse.", "Sleep was improved by design."]
    assert all(2 <= len(ids) <= 3 for _, ids in FakeVerifier.calls)  # only unseen passages
    out = json.loads((env / rejudge.RESULTS).read_text(encoding="utf-8"))
    assert {(f["id"], f["old"], f["new"]) for f in out["flips"]} == {
        ("R1", "unsupported", "supported"), ("B1", "disputed", "supported")}
    m = out["metrics"]["baseline"]
    assert m["hallucination"]["k"] == 0 and m["strict"]["k"] == 2
    assert m["must_refuse_safe"]["k"] == 0                            # T1 still has a rejection
    assert out["comparison"]["bait_safe"]["baseline"]["k"] == 1

    FakeVerifier.calls = []
    rejudge.main([])                                                  # resumes: nothing to judge
    assert FakeVerifier.calls == []


def test_trial_limit_writes_no_results(env):
    rejudge.main(["--limit", "1"])
    assert len(FakeVerifier.calls) == 1 and not (env / rejudge.RESULTS).exists()
