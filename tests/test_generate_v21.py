"""Generator prompt v2.1 and its <refusal> tag: no LLM. Run: python -m pytest tests"""
import config
from src import generate as G

M = config.REFUSAL_MARKER


def test_v21_registered_and_free_of_the_placeholder():
    p = G.GROUNDED_PROMPTS["v2.1"]
    assert "段落編號,段落編號" not in p and 'citation_ids="c_section_9999"' in p
    assert f"<refusal>{M}</refusal>" in p
    # Shorter than v2 (rules 8-9 dropped), kept: source-language quotes, passage types.
    assert len(p) < len(G.GROUNDED_PROMPTS["v2"]) * 1.15
    assert "段落類型" in p and "不可翻譯" in p


def test_refusal_tag_only_is_a_refusal():
    r = G.parse_grounded(f"<answer>\n  <refusal>{M}</refusal>\n</answer>", "q")
    assert r.refused and not r.parse_failed and not r.claims and r.unsupported_note == M


def test_empty_refusal_tag_is_a_refusal():
    r = G.parse_grounded("<answer><refusal/></answer>", "q")
    assert r.refused and not r.parse_failed


def test_claims_with_a_partial_refusal_are_not_a_refusal():
    raw = (f'<answer><claim citation_ids="c1">A finding.<quote>A finding.</quote></claim>'
           f"<refusal>{M}</refusal></answer>")
    r = G.parse_grounded(raw, "q")
    assert not r.refused and [c.statement for c in r.claims] == ["A finding."]


def test_unclosed_claim_ends_at_the_refusal_tag():
    raw = f'<answer><claim citation_ids="c1">A finding.<quote>A finding.</quote><refusal>{M}</refusal></answer>'
    r = G.parse_grounded(raw, "q")
    assert [c.statement for c in r.claims] == ["A finding."] and r.claims[0].quote == "A finding."


def test_marker_inside_a_claim_is_still_a_misplaced_refusal():
    r = G.parse_grounded(f'<answer><claim citation_ids="c1">{M}</claim></answer>', "q")
    assert r.refused and r.misplaced_refusal and not r.claims


def test_v1_output_parses_as_before():
    raw = f"<answer><unsupported_note>{M}</unsupported_note></answer>"
    r = G.parse_grounded(raw, "q")
    assert r.refused and r.unsupported_note == M
