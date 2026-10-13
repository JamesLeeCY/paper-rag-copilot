"""Baseline arm of the grounded-vs-baseline comparison: no LLM, no corpus.
Run: python -m pytest tests"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

import src.generate
import src.retrieve
from eval import run_eval
from src.retrieve import Passage
from src.verify import ClaimVerdict


def test_split_sentences_mixed():
    text = ("Nature walks reduced stress in the sample. Effects were small (d = 0.2).\n"
            "- Participants walked daily for a week.\n"
            "研究發現步行改善了情緒。效果在追蹤時仍然存在！短句。")
    assert run_eval.split_sentences(text) == [
        "Nature walks reduced stress in the sample.",
        "Effects were small (d = 0.2).",
        "Participants walked daily for a week.",
        "研究發現步行改善了情緒。",
        "效果在追蹤時仍然存在！",
    ]


def test_split_keeps_decimals_and_abbreviations_together():
    assert run_eval.split_sentences("The p value was 0.04 overall. It held.") == [
        "The p value was 0.04 overall."]   # "It held." is under 10 chars


def test_split_skips_abbreviations():
    text = ("The model had fixed effects for Group (Walk vs. Rest) and Time (Day 1 vs. "
            "Day 2). Results matched earlier work (Smith et al. 2020; cf. Fig. 3), e.g. Table 2. "
            "A second sentence follows here.")
    assert run_eval.split_sentences(text) == [
        "The model had fixed effects for Group (Walk vs. Rest) and Time (Day 1 vs. Day 2).",
        "Results matched earlier work (Smith et al. 2020; cf. Fig. 3), e.g. Table 2.",
        "A second sentence follows here.",
    ]


@pytest.mark.parametrize("text,abstains", [
    ("The text does not mention cortisol.", True),
    ("This is not reported in the passages.", True),
    ("The passages do not mention this topic.", True),
    ("The authors didn't report it.", True),
    ("文中未提及皮質醇。", True),
    ("資料中沒有相關數據。", True),
    ("The thesis does not explicitly report sleep outcomes.", True),
    ("This was not directly mentioned in the text.", True),
    ("Walking reduced stress.", False),
    ("研究發現步行改善了情緒。", False),
])
def test_refusal_hint(text, abstains):
    assert bool(run_eval.REFUSAL_HINT_RE.search(text)) is abstains


@pytest.mark.parametrize("text,denies", [
    ("The work does not explicitly report this. It focused on walking instead.", True),
    ("It focused on walking instead. The work does not report this.", False),
    ("Walking reduced stress.", False),
    ("", False),
])
def test_opens_with_denial(text, denies):
    assert run_eval.opens_with_denial(text) is denies


class FakeEmbedder:
    """Bag-of-letters vectors: enough to rank passages by overlap."""
    def _vec(self, t):
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


class FakeVerifier:
    def __init__(self):
        self.seen = []

    def verify_statement(self, statement, passages):
        self.seen.append((statement, [p.chunk_id for p in passages]))
        label = "supported" if "walking" in statement.lower() else "unsupported"
        return ClaimVerdict(statement, [passages[0].chunk_id], label, "fake", "llm")


def run_bundle(monkeypatch, answer):
    monkeypatch.setattr(src.retrieve, "get_retriever",
                        lambda s: NS(search=lambda q: PASSAGES, embedder=FakeEmbedder()))
    monkeypatch.setattr(src.generate, "generate",
                        lambda q, p, llm=None, grounded=True: NS(raw=answer))
    ver = FakeVerifier()
    g, v = run_eval.baseline_bundle("q", "section", None, ver)
    return g, v, ver


def test_baseline_bundle_judges_each_claim_against_its_two_closest(monkeypatch):
    g, v, ver = run_bundle(monkeypatch, "Walking reduced stress a lot. Sleep quality improved too. "
                                        "The text does not mention cortisol levels.")
    assert [c.statement for c in g.claims] == ["Walking reduced stress a lot.", "Sleep quality improved too."]
    assert all(len(ids) == 2 for _, ids in ver.seen)
    assert ver.seen[0][1][0] == "c0" and ver.seen[1][1][0] == "c2"   # closest first
    assert [x.label for x in v.verdicts] == ["supported", "unsupported"]
    assert not g.refused and g.mode == "baseline" and g.passages == PASSAGES


def test_baseline_bundle_pure_abstention_is_a_refusal(monkeypatch):
    g, v, _ = run_bundle(monkeypatch, "The passages do not mention this topic.")
    assert g.refused and not g.claims and not v.verdicts


def test_baseline_record_has_the_grounded_shape(monkeypatch):
    g, v, _ = run_bundle(monkeypatch, "Walking reduced stress a lot.")
    rec = run_eval._question_record(g, v)
    assert rec["n_claims"] == 1 and rec["n_supported"] == 1 and rec["refused"] is False


def test_baseline_and_grounded_runs_have_different_signatures():
    llm = NS(describe=lambda: "fake")
    ver = NS(describe=lambda: "judges=[fake]")
    assert run_eval._run_signature("section", llm, ver, "baseline") != \
        run_eval._run_signature("section", llm, ver, "grounded")


def test_signature_records_the_reused_run():
    llm = NS(describe=lambda: "fake")
    ver = NS(describe=lambda: "judges=[fake]")
    assert run_eval._run_signature("section", llm, ver, "baseline", "abc") != \
        run_eval._run_signature("section", llm, ver, "baseline")


def test_reused_verdicts_round_trip(monkeypatch):
    g, v, _ = run_bundle(monkeypatch, "Walking reduced stress a lot. Sleep quality improved too.")
    rec = run_eval._question_record(g, v)
    cached = run_eval.reused_verdicts(rec)
    assert {s: x.label for s, x in cached.items()} == {
        "Walking reduced stress a lot.": "supported", "Sleep quality improved too.": "unsupported"}
    assert cached["Walking reduced stress a lot."].citation_ids == ["c0"]


def test_reuse_rescores_saved_answers_and_judges_only_new_sentences(tmp_path, monkeypatch):
    import json

    import config
    import src.verify

    judged, generated = [], []

    class CountingVerifier(FakeVerifier):
        def __init__(self, llm=None):
            super().__init__()

        def describe(self):
            return "judges=[fake]"

        def verify_statement(self, statement, passages):
            judged.append(statement)
            return super().verify_statement(statement, passages)

    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(src.verify, "Verifier", CountingVerifier)
    monkeypatch.setattr(src.retrieve, "get_retriever",
                        lambda s: NS(search=lambda q: PASSAGES, embedder=FakeEmbedder()))

    def fake_generate(q, p, llm=None, grounded=True):
        generated.append(q)
        return NS(raw="Walking reduced stress here.")
    monkeypatch.setattr(src.generate, "generate", fake_generate)

    golden = {"retrieval": [{"id": "R1", "question": "q one"}, {"id": "R2", "question": "q two"}],
              "traps": []}
    # The old run split "A vs. B" into fragments; its verdict for the first
    # sentence is reused, the re-joined sentence is judged anew.
    old_raw = "Walking reduced stress a lot. Mood ratings vs. Sleep scores were both better."
    old = {"raw": old_raw, "claims": [
        {"statement": "Walking reduced stress a lot.", "judged_against": ["c0"],
         "label": "supported", "reason": "old", "method": "llm", "votes": {"fake": "supported"}},
        {"statement": "Mood ratings vs.", "judged_against": ["c4"],
         "label": "partially_supported", "reason": "old", "method": "llm", "votes": {}}]}
    key = run_eval._question_key("answerable", golden["retrieval"][0])
    (tmp_path / run_eval.PROGRESS_NAME).write_text(
        json.dumps({"run": "OLDSIG", "key": key, "record": old}) + "\n", encoding="utf-8")

    res = run_eval.eval_grounding("section", golden, NS(describe=lambda: "fake:gen"),
                                  baseline=True, reuse_from="OLDSIG")
    r1, r2 = res["answer_detail"]
    assert generated == ["q two"]                    # R1's answer came from the old run
    assert judged == ["Mood ratings vs. Sleep scores were both better.",
                      "Walking reduced stress here."]
    assert [c["label"] for c in r1["claims"]] == ["supported", "unsupported"]
    assert r1["claims"][0]["reason"] == "old" and r1["n_reused_verdicts"] == 1
    assert "reused_from" not in r2
    assert res["n_reused_answers"] == 1 and res["n_reused_verdicts"] == 1


def test_truncated_answers_are_not_reused(tmp_path):
    import json

    rows = [{"run": "S", "key": "a", "record": {"raw": "x" * 3000, "claims": []}},
            {"run": "S", "key": "b", "record": {"raw": "short", "claims": []}},
            {"run": "S", "key": "c", "record": {"claims": []}}]
    (tmp_path / "eval_progress.done-S.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert list(run_eval.load_reusable(tmp_path, "S")) == ["b"]
