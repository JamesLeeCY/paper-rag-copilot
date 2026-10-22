"""Echo rule: a claim that restates a yes/no question (no model, no corpus)."""
import numpy as np
import pytest

import config
import src.verify as V


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    def enc(xs):
        vocab = sorted({c for x in xs for c in x.lower() if c.isalnum()})
        m = np.array([[x.lower().count(c) for c in vocab] for x in xs], dtype=float) + 1e-6
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    monkeypatch.setattr(V, "_match_encoder", enc)


@pytest.mark.parametrize("q, yn", [
    ("兩個條件的壓力有差異嗎？", True), ("介入是否降低了焦慮？", True), ("有沒有改變？", True),
    ("Did the groups differ?", True), ("Was there an interaction?", True),
    ("How many participants completed it?", False), ("哪一組走得比較遠？", False),
])
def test_yes_no_questions(q, yn):
    assert V.is_yes_no(q) is yn


def test_echo_needs_a_yes_no_question():
    assert V.echo_scores("The control group walked further.", "Which group walked further?") is None


def test_echo_of_a_chinese_question_fires():
    why = V.echo_of_question("兩個條件的壓力有差異。", "兩個條件的壓力有差異嗎？")
    assert why.startswith("rule: claim restates the yes/no question")


def test_an_answer_with_its_own_content_is_not_an_echo():
    assert V.echo_of_question("沉浸與心臟同調兩個條件之間沒有顯著差異（p = 0.80）。",
                              "兩個條件的壓力有差異嗎？") == ""


def test_rule_only_touches_partially_supported_claims(monkeypatch):
    from src.generate import Claim, GenerationResult
    from src.retrieve import Passage
    from src.verify import ClaimVerdict, Verifier

    monkeypatch.setattr(config, "ECHO_RULE", "flag")
    p = Passage("c1", "The two conditions did not differ.", "Results", "", 0, 0, "s", [], 1.0)
    res = GenerationResult(query="兩個條件有差異嗎？", raw="", mode="grounded", passages=[p],
                           claims=[Claim(statement="兩個條件有差異。", citation_ids=["c1"],
                                         quote="The two conditions did not differ.")])
    v = Verifier.__new__(Verifier)
    for label, expected in (("partially_supported", "disputed"), ("supported", "supported"),
                            ("unsupported", "unsupported")):
        v._judge = lambda s, t, label=label: ClaimVerdict(s, [], label, "ok", "llm")
        assert v._verify_claim(res.claims[0], res).label == expected
    monkeypatch.setattr(config, "ECHO_RULE", "off")
    v._judge = lambda s, t: ClaimVerdict(s, [], "partially_supported", "ok", "llm")
    assert v._verify_claim(res.claims[0], res).label == "partially_supported"
