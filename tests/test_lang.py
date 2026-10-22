"""Question language switch for the evaluation harness (no model, no corpus)."""
from eval.run_eval import _question_key, in_language

GOLD = {"retrieval": [{"id": "R01", "question": "How many?", "question_zh": "多少？"}],
        "traps": [{"id": "T01", "question": "Why X?", "question_zh": "為何 X？"},
                  {"id": "B01", "question": "結果為何？"}]}


def test_en_is_unchanged():
    assert in_language(GOLD, "en") is GOLD


def test_zh_swaps_questions_and_keeps_the_english():
    zh = in_language(GOLD, "zh")
    assert zh["retrieval"][0]["question"] == "多少？"
    assert zh["retrieval"][0]["question_en"] == "How many?"
    assert zh["traps"][1]["question"] == "結果為何？"     # already Chinese, no question_zh
    assert GOLD["retrieval"][0]["question"] == "How many?"   # input not mutated


def test_languages_never_share_progress_keys():
    zh = in_language(GOLD, "zh")
    assert _question_key("answerable", zh["retrieval"][0]) != _question_key("answerable", GOLD["retrieval"][0])
