"""Resumable grounding evaluation: no LLM, no corpus. Run: python -m pytest tests

A fake ``ask`` stands in for generation + verification; the tests interrupt a
run and check that the resumed run scores only the remaining questions and
reports the same metrics as an uninterrupted one.
"""
from types import SimpleNamespace as NS

import pytest

import config
import src.pipeline
import src.verify
from eval import run_eval
from src.generate import Claim
from src.verify import ClaimVerdict, VerificationReport

GOLDEN = {
    "retrieval": [{"id": f"R{i}", "question": f"answerable question {i}"} for i in range(1, 5)],
    "traps": [{"id": "T1", "question": "far trap", "type": "far_absent"},
              {"id": "T2", "question": "premise trap", "type": "false_premise"}],
}
LABELS = {"answerable question 1": "supported", "answerable question 2": "partially_supported",
          "answerable question 3": "unsupported", "answerable question 4": "supported",
          "premise trap": "supported"}


class FakeVerifier:
    def __init__(self, llm=None):
        pass

    def describe(self):
        return "judges=[fake]"


class Interrupted(Exception):
    pass


def make_ask(calls, fail_on=None):
    def ask(question, strategy=None, llm=None, verifier=None):
        calls.append(question)
        if fail_on is not None and len(calls) == fail_on:
            raise Interrupted
        refused = question == "far trap"
        claims, verdicts = [], []
        if not refused:
            claims = [Claim(f"claim for {question}", ["c1"], "quoted sentence")]
            verdicts = [ClaimVerdict(claims[0].statement, ["c1"], LABELS[question], "fake",
                                     "llm", {"fake": LABELS[question]}, "verbatim")]
        g = NS(refused=refused, parse_failed=False, misplaced_refusal=False, raw="",
               claims=claims, passages=[NS(chunk_id="c1", text="quoted sentence")])
        return NS(generation=g, verification=VerificationReport(question, verdicts))
    return ask


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(src.verify, "Verifier", FakeVerifier)
    return tmp_path


def run(monkeypatch, calls, fail_on=None, fresh=False):
    monkeypatch.setattr(src.pipeline, "ask", make_ask(calls, fail_on))
    return run_eval.eval_grounding("section", GOLDEN, NS(describe=lambda: "fake:gen"), fresh=fresh)


METRICS = ("n_claims_total", "n_supported", "n_partial", "n_unsupported",
           "citation_precision_strict", "citation_precision_lenient", "hallucination_rate",
           "refusal_correctness", "over_refusal_rate", "answer_hallucination_rate",
           "traps_by_type", "quote_counts", "parse_failures")


def test_metrics(env, monkeypatch):
    res = run(monkeypatch, [])
    assert res["n_claims_total"] == 4 and res["citation_precision_strict"] == 0.5
    assert res["hallucination_rate"] == 0.25 and res["refusal_correctness"] == 1.0
    assert res["traps_by_type"]["false_premise"]["correct_rate"] == 1.0
    assert res["n_resumed"] == 0


def test_interrupted_run_resumes_and_matches_a_full_run(env, monkeypatch):
    full = run(monkeypatch, [])

    calls = []
    with pytest.raises(Interrupted):
        run(monkeypatch, calls, fail_on=4)           # stops while scoring the 4th question
    assert len((env / run_eval.PROGRESS_NAME).read_text(encoding="utf-8").splitlines()) == 3

    calls = []
    resumed = run(monkeypatch, calls)
    assert resumed["n_resumed"] == 3
    assert calls == ["answerable question 4", "far trap", "premise trap"]
    assert {k: resumed[k] for k in METRICS} == {k: full[k] for k in METRICS}


def test_completed_run_retires_its_progress(env, monkeypatch):
    run(monkeypatch, [])
    assert not (env / run_eval.PROGRESS_NAME).exists()
    calls = []
    run(monkeypatch, calls)                          # next run starts fresh
    assert len(calls) == 6


def test_fresh_ignores_saved_questions(env, monkeypatch):
    with pytest.raises(Interrupted):
        run(monkeypatch, [], fail_on=3)
    calls = []
    assert run(monkeypatch, calls, fresh=True)["n_resumed"] == 0
    assert len(calls) == 6


def test_changed_settings_do_not_reuse_saved_questions(env, monkeypatch):
    with pytest.raises(Interrupted):
        run(monkeypatch, [], fail_on=3)
    monkeypatch.setattr(config, "PLAN_RESULT_RULE", "reject")
    calls = []
    assert run(monkeypatch, calls)["n_resumed"] == 0
    assert len(calls) == 6


def test_a_line_cut_off_by_a_hard_stop_is_skipped(env, monkeypatch):
    with pytest.raises(Interrupted):
        run(monkeypatch, [], fail_on=3)
    with (env / run_eval.PROGRESS_NAME).open("a", encoding="utf-8") as f:
        f.write('{"run": "x", "key": "answ')       # partial write
    calls = []
    assert run(monkeypatch, calls)["n_resumed"] == 2
    assert len(calls) == 4
