"""PDF reader filters for page-1 sidebars, bare DOI headers, wrapped DOIs and
equation debris."""
import pytest

from src.ingest_pdf import _BARE_ID_RE, _CONTROL_RE, _DOI_WRAP_RE, _SIDEBAR_RE


@pytest.mark.parametrize("line", [
    "OPEN ACCESS", "EDITED BY", "REVIEWED BY", "*CORRESPONDENCE", "RECEIVED 20 December 2023",
    "ACCEPTED 20 March 2024", "PUBLISHED 04 April 2024", "CITATION", "COPYRIGHT",
    "TYPE Original Research", "DOI 10.3389/frvir.2024.1358981",
])
def test_sidebar_labels_are_dropped(line):
    assert _SIDEBAR_RE.match(line)


@pytest.mark.parametrize("line", [
    "Citation patterns differed between groups.",   # body text, not the capital label
    "Published studies report similar effects.",
    "KEYWORDS ADHD; cognition; EEG",                # kept: a body paragraph in other journals
    "TYPES of stimuli",
])
def test_body_text_is_not_a_sidebar(line):
    assert not _SIDEBAR_RE.match(line)


@pytest.mark.parametrize("line, bare", [
    ("10.3389/frvir.2024.1358981", True),
    ("DOI: 10.1016/j.jenvp.2026.103181", True),
    ("first.author@university.example", True),
    ("see 10.3389/frvir.2024.1358981 for details", False),
    ("Contact: someone@example.org", False),
])
def test_bare_identifier_lines(line, bare):
    assert bool(_BARE_ID_RE.match(line)) is bare


@pytest.mark.parametrize("line, wraps", [
    ("Perform. 88, 520–526. doi:10.3357/AMHP.4747.", True),   # next line "2017" is kept
    ("doi:10.5298/1081-5937-41.", True),
    ("Perform. 88, 520–526. doi:10.3357/AMHP.4747.2017", False),
    ("Cognitive load rose in session 3.", False),             # next "12" is a page number
])
def test_a_number_after_a_wrapped_doi_is_not_a_page_number(line, wraps):
    assert bool(_DOI_WRAP_RE.search(line)) is wraps


def test_math_font_debris_is_detected():
    assert _CONTROL_RE.search("RMSSD \x01")
    assert not _CONTROL_RE.search("RMSSD increased between Baseline and Cardiac coherence\n")


def _pline(text, x0, y1, width=None):
    span = {"text": text, "size": 9.0, "flags": 0, "font": "Times"}
    return {"bbox": [x0, y1 - 8, x0 + (width or 6 * len(text)), y1], "spans": [span]}


def test_baseline_ids_group_words_on_one_line():
    from src.ingest_pdf import _baseline_ids
    block = {"lines": [_pline("of", 40, 100), _pline("20", 60, 100), _pline(" ", 80, 100),
                       _pline("items", 90, 100), _pline("next", 40, 110)]}
    assert _baseline_ids(block) == [0, 0, 0, 1]      # the blank line is skipped


def test_numbers_beside_kept_words_stay_and_page_numbers_go():
    from src.ingest_pdf import _baseline_ids, _block_lines, _filter_lines, _norm_repeat
    footer = "Journal of Invented Studies | www.example.org"
    block = {"lines": [_pline("consisting", 40, 100), _pline("of", 110, 100), _pline("20", 130, 100),
                       _pline("questions", 150, 100),
                       _pline("Week", 40, 110), _pline("1", 80, 110), _pline("31.89", 100, 110),
                       _pline("34", 140, 110),
                       _pline(footer, 40, 700, 200), _pline("7", 250, 700),
                       _pline("12", 300, 720)]}
    lines = _block_lines(block)
    kept = _filter_lines(lines, _baseline_ids(block), {_norm_repeat(footer)})
    assert [ln[0] for ln in kept] == ["consisting", "of", "20", "questions", "Week", "1", "31.89", "34"]


def test_doi_tail_rule_still_applies_without_baselines():
    from src.ingest_pdf import _filter_lines
    lines = [("Perform. 88, 520–526. doi:10.3357/AMHP.4747.", 9.0, False, False),
             ("2017", 9.0, False, False), ("3", 9.0, False, False)]
    assert [ln[0] for ln in _filter_lines(lines, [0, 1, 2], set())] == [lines[0][0], "2017"]
