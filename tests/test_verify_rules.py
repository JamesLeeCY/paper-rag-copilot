"""Deterministic verification rules: no LLM, no corpus. Run: python -m pytest tests

Covers the plan-as-result rule, quotes that span chunks, and the section
heading shown to judges. Texts are generic examples, not corpus content.
"""
from types import SimpleNamespace as NS

import pytest

import config
from src.generate import Claim
from src.retrieve import Passage
from src.verify import Verifier, check_quote, plan_as_result, quote_span, with_heading

PLAN = ("Assessments of mood and sleep will be conducted at three time points: "
        "before the intervention, after it, and six months later.")
RESULT = "Mood improved after the intervention and the benefit was maintained at follow-up."
RECRUIT = "Participants will be recruited through online forums."


def passage(cid, text, path=()):
    return Passage(cid, text, path[-1] if path else "", "", 0, 0, "src", [], 1.0,
                   section_path=list(path))


class FakeJudge:
    def __init__(self, label="supported"):
        self.label, self.seen = label, []

    def complete(self, system, user, **kwargs):
        self.seen.append(user)
        return ('{"strength": "same", "plan_as_result": false, "misattributed": false, '
                f'"label": "{self.label}", "reason": "fake"}}')

    def describe(self):
        return "fake:judge"


def verifier(judge):
    v = Verifier.__new__(Verifier)
    v.judges, v._nli = [judge], None
    return v


# -- plan read as result ----------------------------------------------------
@pytest.mark.parametrize("claim", [
    "The six-month follow-up showed that the benefit was maintained.",
    "六個月後的追蹤顯示，介入的效果得以保持。",
])
def test_plan_rule_flags_a_finding_read_from_a_plan(claim):
    assert plan_as_result(claim, "", [PLAN])


@pytest.mark.parametrize("claim", [
    "Assessments will be conducted six months after the intervention.",  # reported as a plan
    "研究預計在介入結束六個月後進行追蹤評估。",                            # hedged in Chinese
    "Participants were recruited through online forums.",                 # procedure, no finding
])
def test_plan_rule_leaves_plans_and_procedures_alone(claim):
    assert not plan_as_result(claim, "", [PLAN + " " + RECRUIT])


def test_plan_rule_ignores_a_finding_from_a_results_passage():
    assert not plan_as_result("Mood improved after the intervention.", "", [RESULT])


def test_plan_rule_uses_the_quote_when_there_is_one():
    claim = "The follow-up showed that the benefit was maintained."
    assert plan_as_result(claim, PLAN, [RESULT])          # quoted sentence is a plan
    assert not plan_as_result(claim, RESULT, [PLAN])      # quoted sentence is a result


HYP_HEADING = "2. Research Questions and Hypotheses > 2.1 Study 1 > 2.1.3 Hypotheses"
HYPOTHESIS = "H1: Walking in a park will lower stress more than walking in a street."


@pytest.mark.parametrize("claim", [
    "公園步行比街道步行更讓人放鬆。",                       # no result cue word at all
    "Park walks left participants calmer than street walks.",
])
def test_structural_rule_flags_any_unhedged_claim_citing_a_hypotheses_section(claim):
    assert plan_as_result(claim, "", [HYPOTHESIS], [HYP_HEADING])
    assert not plan_as_result(claim, "", [HYPOTHESIS], ["2.1.2 Methods"])  # wording route: no cue


@pytest.mark.parametrize("claim", [
    "研究假設公園步行會比街道步行更能降低壓力。",
    "研究預期公園步行比街道步行更讓人放鬆。",
    "The study expected park walks to lower stress more than street walks.",
])
def test_structural_rule_spares_a_hypothesis_reported_as_one(claim):
    assert not plan_as_result(claim, "", [HYPOTHESIS], [HYP_HEADING])


def test_structural_rule_reads_only_the_innermost_heading():
    chapter_only = "2. Research Questions and Hypotheses > 2.1 Study 1 > 2.1.1 Background"
    assert not plan_as_result("公園步行比街道步行更讓人放鬆。", "", [HYPOTHESIS], [chapter_only])


@pytest.mark.parametrize("mode,expected", [
    ("flag", "disputed"), ("reject", "unsupported"), ("off", "supported")])
def test_plan_rule_overrules_an_accepting_judge(monkeypatch, mode, expected):
    monkeypatch.setattr(config, "PLAN_RESULT_RULE", mode)
    result = NS(query="q", passages=[passage("c1", PLAN)], claims=[])
    claim = Claim("The six-month follow-up showed that the benefit was maintained.", ["c1"], "")
    assert verifier(FakeJudge("supported"))._verify_claim(claim, result).label == expected


def test_plan_rule_keeps_a_rejection(monkeypatch):
    monkeypatch.setattr(config, "PLAN_RESULT_RULE", "flag")
    result = NS(query="q", passages=[passage("c1", PLAN)], claims=[])
    claim = Claim("The six-month follow-up showed that the benefit was maintained.", ["c1"], "")
    assert verifier(FakeJudge("unsupported"))._verify_claim(claim, result).label == "unsupported"


# -- quotes spanning chunks ---------------------------------------------------
A = "The task had two phases. In the first phase, participants viewed 24 shapes."
B = "In the second phase, participants were shown old and new shapes. Responses were timed."
SPAN_QUOTE = "In the first phase, participants viewed 24 shapes. In the second phase, participants were shown old and new shapes."


def test_quote_across_a_chunk_boundary_is_not_in_either_chunk():
    assert check_quote(SPAN_QUOTE, [A]) == "not_found"
    assert check_quote(SPAN_QUOTE, [B]) == "not_found"


def test_quote_span_finds_every_sentence_across_chunks():
    pa, pb = passage("a", A), passage("b", B)
    assert quote_span(SPAN_QUOTE, [pa, pb]) == [pa, pb]


def test_quote_span_rejects_an_invented_sentence():
    quote = "In the first phase, participants viewed 24 shapes. Participants were paid for a third session."
    assert quote_span(quote, [passage("a", A), passage("b", B)]) is None


def test_quote_span_leaves_single_sentences_to_check_quote():
    assert quote_span("In the first phase, participants viewed 24 shapes.", [passage("a", A)]) is None


def test_spanning_quote_is_judged_against_both_chunks():
    judge = FakeJudge("supported")
    result = NS(query="q", passages=[passage("a", A), passage("b", B)], claims=[])
    out = verifier(judge)._verify_claim(Claim("The task had two phases.", ["a"], SPAN_QUOTE), result)
    assert out.label == "supported" and out.quote_check == "near"
    assert out.citation_ids == ["a", "b"] and out.citation_repaired_from == ["a"]


def test_spanning_quote_needs_the_next_chunk_retrieved():
    result = NS(query="q", passages=[passage("a", A)], claims=[])
    out = verifier(FakeJudge())._verify_claim(Claim("The task had two phases.", ["a"], SPAN_QUOTE), result)
    assert out.label == "unsupported" and out.method == "quote"


# -- section heading shown to the judge -----------------------------------------
def test_with_heading():
    assert with_heading("text", "2. Methods > 2.1 Task") == "（章節：2. Methods > 2.1 Task）\ntext"
    assert with_heading("text", "") == "text"


def test_judge_sees_the_heading_but_quotes_match_the_bare_text():
    judge = FakeJudge("supported")
    result = NS(query="q", passages=[passage("a", A, ("2. Methods", "2.1 Task"))], claims=[])
    out = verifier(judge)._verify_claim(
        Claim("The task had two phases.", ["a"], "The task had two phases."), result)
    assert out.quote_check == "verbatim"
    assert "（章節：2. Methods > 2.1 Task）" in judge.seen[0]
