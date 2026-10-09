"""Generator prompt v2 context and the translated-quote check: no LLM, no corpus.
Run: python -m pytest tests"""
import pytest

from eval.run_eval import is_translated_quote
from src import generate
from src.retrieve import Passage


def passage(path, text="Some text.", cid="c1"):
    return Passage(cid, text, path[-1] if path else "", "", 0, 0, "src", [], 1.0,
                   section_path=list(path))


@pytest.mark.parametrize("path,expected", [
    (["2. Research Questions and Hypotheses", "2.1 Study 1", "2.1.3 Hypotheses"], "研究假設"),
    (["2. Research Questions and Hypotheses", "2.1 Study 1", "2.1.1 Rationale"], "背景"),
    (["3. Materials and Methods", "3.1 Study 1: A Walking Task"], "研究方法"),
    (["3. Materials and Methods", "3.1 Study 1", "Psychological measurements"], "研究方法"),
    (["4. Results", "4.1 Study 1: Effects on Mood"], "研究結果"),
    (["4. Results", "4.3 Study 3", "4.3.2 Results: Correlation Analysis"], "研究結果"),
    (["5. Discussion", "5.4 Limitations"], "討論"),
    (["1. Introduction", "1.2 Prior Work"], "背景"),
    (["1. Introduction", "1.3 What Determines the Efficacy of Design"], "背景"),   # chapter decides
    (["1. Introduction", "1.2 Theories and Their Implications"], "背景"),
    (["1 Objectives"], "背景"),
    (["Acknowledgements"], "其他"),
])
def test_passage_type(path, expected):
    assert generate.passage_type(passage(path)) == expected


def test_v1_context_is_unchanged():
    p = Passage("c1", "Body.", "1.2 Prior Work", "1.2", 0, 0, "Src", [], 1.0,
                section_path=["1. Introduction", "1.2 Prior Work"])
    assert generate.build_context([p]) == "[c1] (來源: Src, §1.2 1.2 Prior Work)\nBody."


def test_v2_context_labels_heading_and_type():
    p = passage(["4. Results", "4.1 Study 1"], "Body.")
    ctx = generate.build_context([p], "v2")
    assert ctx.startswith("[c1] (來源: src；章節: 4. Results > 4.1 Study 1；段落類型: 研究結果)")


def test_prompt_versions():
    assert set(generate.GROUNDED_PROMPTS) == {"v1", "v2"}
    v2 = generate.GROUNDED_PROMPTS["v2"]
    for must in ("原文語言", "段落類型", "研究假設", "其他研究", "否定的事實", "unsupported_note"):
        assert must in v2


@pytest.mark.parametrize("quote,passages,expected", [
    ("參與者每天步行。", ["Participants walked daily."], True),
    ("Participants walked daily.", ["Participants walked daily."], False),
    ("參與者每天步行。", ["參與者每天步行三十分鐘。"], False),   # Chinese source
    ("", ["Participants walked daily."], False),
    ("參與者每天步行。", [], False),
])
def test_translated_quote(quote, passages, expected):
    assert is_translated_quote(quote, passages) is expected
