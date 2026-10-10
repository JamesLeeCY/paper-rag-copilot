"""Judge prompt v4: classification fields decide stage and attribution mismatches.
No LLM, no corpus. Run: python -m pytest tests"""
import json

import pytest

from src.verify import VERIFY_PROMPTS, VERIFY_SYSTEM_V4, Verifier, stage_verdict


def reply(label="supported", **fields):
    return json.dumps({**fields, "strength": "same", "label": label, "reason": "fake"})


class FakeJudge:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, system, user, **kwargs):
        self.calls.append(kwargs)
        return self.text

    def describe(self):
        return "fake:judge"


@pytest.mark.parametrize("src,claim,attr,expected", [
    ("hypothesis", "finding", "none", "unsupported"),          # hypothesis read as a finding
    ("plan_or_method", "finding", "this_study", "unsupported"),  # plan read as a finding
    ("other_study", "finding", "this_study", "unsupported"),   # misattribution
    ("other_study", "finding", "other_study", ""),             # correctly attributed
    ("hypothesis", "hypothesis_or_plan", "none", ""),          # hypothesis reported as one
    ("plan_or_method", "method", "none", ""),                  # procedure reported as one
    ("result", "finding", "this_study", ""),                   # ordinary result
    ("background", "background", "none", ""),
])
def test_stage_verdict(src, claim, attr, expected):
    data = {"source_stage": src, "claim_stage": claim, "claim_attribution": attr}
    assert stage_verdict(data) == expected


@pytest.mark.parametrize("data", [
    {},                                                        # v1-v3 replies
    {"source_stage": "hypothesis"},                            # incomplete
    {"source_stage": "made_up", "claim_stage": "finding"},     # unknown value
    {"source_stage": "Hypothesis ", "claim_stage": "FINDING", "claim_attribution": "none"},
])
def test_stage_verdict_ignores_missing_or_unknown_fields(data):
    expected = "unsupported" if data.get("claim_stage") == "FINDING" else ""
    assert stage_verdict(data) == expected   # last case: case/space tolerant


def test_v4_overrules_an_accepting_label():
    judge = FakeJudge(reply("supported", source_stage="hypothesis", claim_stage="finding",
                            claim_attribution="none"))
    label, _ = Verifier._llm_vote(judge, "claim", "passage", prompt=VERIFY_SYSTEM_V4)
    assert label == "unsupported"
    assert judge.calls[0]["max_tokens"] == 400


def test_v4_keeps_the_label_when_stages_agree():
    judge = FakeJudge(reply("partially_supported", source_stage="result", claim_stage="finding",
                            claim_attribution="this_study"))
    assert Verifier._llm_vote(judge, "c", "p", prompt=VERIFY_SYSTEM_V4)[0] == "partially_supported"


def test_v3_replies_are_unaffected():
    judge = FakeJudge(reply("supported", plan_as_result=False, misattributed=False))
    assert Verifier._llm_vote(judge, "c", "p", prompt=VERIFY_PROMPTS["v3"])[0] == "supported"
    assert judge.calls[0]["max_tokens"] == 300


def test_v4_prompt_lists_every_field_in_order():
    order = ["source_stage", "claim_stage", "claim_attribution", "strength", "label", "reason"]
    line = next(l for l in VERIFY_SYSTEM_V4.splitlines() if l.startswith('{"source_stage"'))
    positions = [line.index(f'"{f}"') for f in order]
    assert positions == sorted(positions)
