"""PDF reader filters for page-1 sidebars, bare DOI headers and equation debris."""
import pytest

from src.ingest_pdf import _BARE_ID_RE, _CONTROL_RE, _SIDEBAR_RE


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


def test_math_font_debris_is_detected():
    assert _CONTROL_RE.search("RMSSD \x01")
    assert not _CONTROL_RE.search("RMSSD increased between Baseline and Cardiac coherence\n")
