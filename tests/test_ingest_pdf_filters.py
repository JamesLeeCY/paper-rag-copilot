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
