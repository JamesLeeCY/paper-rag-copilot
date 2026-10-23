"""Chinese overclaim rule (no model, no corpus)."""
import pytest

import config
from src.verify import ClaimVerdict, Verifier, zh_overclaim

HEDGED = "These findings suggest that exposure may enhance connection to nature."
STRONG = "These findings demonstrate that exposure enhances connection to nature."


@pytest.mark.parametrize("claim, fires", [
    ("這些發現證明暴露增強了與自然的連結。", True),
    ("自然必定總是扮演調節者的角色。", True),
    ("这些发现证明暴露增强了连结。", True),            # simplified
    ("這些發現顯示暴露可能增強了與自然的連結。", False),   # hedged
    ("研究並未證明兩組有差異。", False),                 # negated
    ("目前無法證實這個效果。", False),
    ("These findings prove the effect.", False),          # English claims are left to the judge
])
def test_strong_chinese_words_against_a_hedged_source(claim, fires):
    assert bool(zh_overclaim(claim, [HEDGED])) is fires


def test_a_strong_source_licenses_the_strong_claim():
    assert zh_overclaim("這些發現證明暴露增強了與自然的連結。", [STRONG]) == ""


def test_rule_downgrades_only_supported_and_only_when_on(monkeypatch):
    from src.retrieve import Passage

    v = Verifier.__new__(Verifier)
    p = Passage("c1", HEDGED, "Results", "", 0, 0, "s", [], 1.0)
    claim = "這些發現證明暴露增強了與自然的連結。"
    v._judge = lambda s, t: ClaimVerdict(s, [], "supported", "ok", "llm")
    monkeypatch.setattr(config, "ZH_OVERCLAIM_RULE", "off")
    assert v.verify_statement(claim, [p]).label == "supported"
    monkeypatch.setattr(config, "ZH_OVERCLAIM_RULE", "flag")
    assert v.verify_statement(claim, [p]).label == "partially_supported"
