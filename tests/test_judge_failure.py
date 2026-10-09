"""Judge failures and request timeouts: no LLM, no corpus. Run: python -m pytest tests"""
import pytest

import config
from src.llm import request_timeout
from src.verify import Verifier

PASSAGE = "Participants walked in a park for 30 minutes each day."


class FailingJudge:
    def __init__(self, error):
        self.error = error

    def complete(self, *args, **kwargs):
        raise self.error

    def describe(self):
        return "fake:failing"


class Garbled:
    def complete(self, *args, **kwargs):
        return "not json at all"

    def describe(self):
        return "fake:garbled"


def verifier(judges):
    v = Verifier.__new__(Verifier)
    v.judges, v._nli, v.use_nli = judges, None, False
    return v


@pytest.mark.parametrize("judge", [FailingJudge(TimeoutError("timed out")),
                                   FailingJudge(ConnectionError("refused")), Garbled()])
@pytest.mark.parametrize("claim", ["參與者每天在公園步行三十分鐘。",
                                   "Participants walked in a park daily."])
def test_no_usable_vote_goes_to_review_not_rejection(judge, claim):
    out = verifier([judge])._judge(claim, PASSAGE)
    assert out.label == "disputed" and out.method == "none"
    assert "judge unavailable" in out.reason


def test_without_any_judge_the_lexical_fallback_still_runs():
    out = verifier([])._judge("Participants walked in a park daily.", PASSAGE)
    assert out.method == "lexical"


def test_timeout_floor_for_short_requests(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_SECONDS_PER_TOKEN", 0.4)
    assert request_timeout(200, 300) == 300.0


def test_timeout_grows_with_the_prompt(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_SECONDS_PER_TOKEN", 0.8)
    short = request_timeout(1_000, 300)
    long = request_timeout(4_000, 300)       # a judge reading a long passage
    assert long > short and long > 300.0


def test_timeout_scales_with_seconds_per_token(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_SECONDS_PER_TOKEN", 0.4)
    slow_off = request_timeout(4_000, 1600)
    monkeypatch.setattr(config, "OLLAMA_SECONDS_PER_TOKEN", 0.8)
    assert request_timeout(4_000, 1600) > slow_off
