"""Generator prompt v2.1 and its <refusal> tag: no LLM. Run: python -m pytest tests"""
import config
from src import generate as G

M = config.REFUSAL_MARKER


def test_v21_registered_and_free_of_the_placeholder():
    p = G.GROUNDED_PROMPTS["v2.1"]
    assert "段落編號,段落編號" not in p and 'citation_ids="c_section_9999"' in p
    assert f"<refusal>{M}</refusal>" in p
    # Rules 8-9 were dropped, but the worked examples and the calibration
    # fixes (full sentences, quotes, refusal scope) made it longer than v2
    # (1235 vs 1081 characters); keep it from growing further unnoticed.
    # Kept: source-language quotes, passage types.
    assert len(p) < 1400
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


def test_quote_after_the_claim_is_attached():
    raw = ('<answer><claim citation_ids="c1">A finding.</claim>\n  '
           '<quote>The finding held.</quote></answer>')
    r = G.parse_grounded(raw, "q")
    assert r.claims[0].statement == "A finding." and r.claims[0].quote == "The finding held."


def test_v21_asks_for_a_quote_even_when_the_claim_copies_the_source():
    p = G.GROUNDED_PROMPTS["v2.1"]
    assert "就算論點直接沿用原文句子" in p and 'citation_ids="c_section_9998"' in p


def test_v21_asks_for_full_sentences():
    assert "用完整的句子說明" in G.GROUNDED_PROMPTS["v2.1"]


def test_stray_closing_quote_tag():
    r = G.parse_grounded('<answer><claim citation_ids="c1">The study recruited 80 adults.</quote></claim></answer>', "q")
    assert r.claims[0].statement == "The study recruited 80 adults."
    assert r.claims[0].quote == ""        # left to the verifier's verbatim check


def test_v21_refusal_only_for_unanswered_parts():
    assert "不要加 <refusal>" in G.GROUNDED_PROMPTS["v2.1"]


def _p(cid, path):
    from src.retrieve import Passage
    return Passage(cid, "text", path[-1], "", 0, 0, "s", [], 1.0, section_path=path)


def test_order_passages_results_first_is_stable():
    ps = [_p("h", ["2. Questions", "2.1 Hypotheses"]), _p("b", ["1. Introduction"]),
          _p("r1", ["4. Results"]), _p("m", ["3. Methods"]), _p("d", ["5. Discussion"]),
          _p("r2", ["4. Results", "4.2 Study II"])]
    assert [p.chunk_id for p in G.order_passages(ps, "results_first")] == ["r1", "d", "r2", "b", "m", "h"]
    assert [p.chunk_id for p in G.order_passages(ps, "rank")] == [p.chunk_id for p in ps]


def test_context_order_changes_only_the_grounded_prompt(monkeypatch):
    seen = []

    class FakeLLM:
        available = True

        def complete(self, system, user, **kw):
            seen.append(user)
            return "<answer><refusal/></answer>"

    ps = [_p("h", ["2. Questions", "2.1 Hypotheses"]), _p("r", ["4. Results"])]
    monkeypatch.setattr(config, "CONTEXT_ORDER", "results_first")
    res = G.generate("q", ps, llm=FakeLLM())
    assert seen[0].index("[r]") < seen[0].index("[h]")
    assert [p.chunk_id for p in res.passages] == ["h", "r"]       # retrieval order kept
    G.generate("q", ps, llm=FakeLLM(), grounded=False)
    assert seen[1].index("[h]") < seen[1].index("[r]")            # baseline unchanged
