"""Hypothesis-match route of the plan-as-result rule: no model, no corpus.
Run: python -m pytest tests"""
import numpy as np
import pytest

import config
import src.verify as V
from src.retrieve import Passage
from src.verify import ClaimVerdict, Verifier


def bag(text):
    """Unit bag-of-words vector over a tiny vocabulary: enough to rank."""
    vocab = ["walking", "stress", "sleep", "mood", "memory", "regions", "increase", "will"]
    v = np.array([text.lower().count(w) for w in vocab], dtype=float) + 1e-6
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr(V, "_match_encoder", lambda xs: np.array([bag(x) for x in xs]))
    monkeypatch.setattr(config, "PLAN_RESULT_MATCH", "on")
    monkeypatch.setattr(config, "PLAN_RESULT_RULE", "flag")
    monkeypatch.setattr(config, "PLAN_MATCH_MIN_SIM", 0.55)
    monkeypatch.setattr(config, "PLAN_MATCH_MARGIN", -0.02)


def passage(cid, text, path):
    return Passage(cid, text, path[-1], "", 0, 0, "s", [], 1.0, section_path=path)


HYP = passage("h", "H1: Walking will increase memory in older regions of the sample.",
              ["2. Questions", "2.1 Hypotheses"])
RES = passage("r", "Sleep and mood scores were reported for every participant here.",
              ["4. Results", "4.1 Study I"])
CONTEXT = [HYP, RES]
CLAIM = "Walking increased memory in these regions after the program."


def test_fires_when_the_claim_matches_a_hypothesis_more_than_its_citation():
    why = V.hypothesis_match(CLAIM, [RES.text], [(p.text, p.heading()) for p in CONTEXT])
    assert why.startswith("rule: hypothesis restated as finding")


def test_silent_without_a_hypotheses_section_in_the_context():
    assert V.hypothesis_match(CLAIM, [RES.text], [(RES.text, RES.heading())]) == ""


def test_silent_for_a_hedged_claim():
    hedged = "Walking will increase memory in these regions, the study expects."
    assert V.hypothesis_match(hedged, [RES.text], [(p.text, p.heading()) for p in CONTEXT]) == ""


def test_silent_when_the_claim_matches_its_citation():
    claim = "Sleep and mood scores were reported for each participant."
    assert V.hypothesis_match(claim, [RES.text], [(p.text, p.heading()) for p in CONTEXT]) == ""


def verdict(label):
    return ClaimVerdict(CLAIM, ["r"], label, "judge reason", "llm")


def test_only_a_partial_verdict_is_flagged():
    v = Verifier._apply_plan_rule(verdict("partially_supported"), "", [RES.text],
                                  [RES.heading()], CONTEXT)
    assert v.label == "disputed" and "hypothesis restated" in v.reason
    # A supported claim may restate a hypothesis the results confirm.
    v = Verifier._apply_plan_rule(verdict("supported"), "", [RES.text], [RES.heading()], CONTEXT)
    assert v.label == "supported"


def test_route_is_off_by_default_setting(monkeypatch):
    monkeypatch.setattr(config, "PLAN_RESULT_MATCH", "off")
    v = Verifier._apply_plan_rule(verdict("partially_supported"), "", [RES.text],
                                  [RES.heading()], CONTEXT)
    assert v.label == "partially_supported"


def test_reject_mode_marks_it_unsupported(monkeypatch):
    monkeypatch.setattr(config, "PLAN_RESULT_RULE", "reject")
    v = Verifier._apply_plan_rule(verdict("partially_supported"), "", [RES.text],
                                  [RES.heading()], CONTEXT)
    assert v.label == "unsupported"
