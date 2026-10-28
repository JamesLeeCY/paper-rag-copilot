"""PDF reference-list splitting: one-word-per-line justified lists, spacing
accents, numbered lists and end matter. Author names are invented."""
from src.ingest_pdf import _block_lines, _fix_accents, _split_references


def _line(text, x0, y1, width=None):
    width = width or 5 * len(text)
    span = {"text": text, "size": 7.0, "flags": 0, "font": "Times"}
    return {"bbox": [x0, y1 - 8, x0 + width, y1], "spans": [span]}


def test_words_on_one_baseline_are_joined_only_when_asked():
    block = {"lines": [_line("Quill,", 45, 624, 22), _line("A.", 75, 624, 8),
                       _line("(2010).", 90, 624, 25), _line("Title", 50, 633)]}
    assert [ln[0] for ln in _block_lines(block)] == ["Quill,", "A.", "(2010).", "Title"]
    assert [ln[0] for ln in _block_lines(block, merge_baseline=True)] == ["Quill, A. (2010).", "Title"]


def test_merged_lines_split_into_entries_and_keep_a_wrapped_doi():
    block = {"lines": [_line("Fennel,", 45, 600, 25), _line("B.", 75, 600, 8),
                       _line("(2003).", 90, 600, 25), _line("Running.", 120, 600, 30),
                       _line("doi:", 160, 600, 12), _line("10.1109/ABC.2013.65", 180, 600, 60),
                       _line("50194", 50, 609, 20),
                       _line("Gorse,", 45, 618, 22), _line("C.", 75, 618, 8),
                       _line("(2012).", 90, 618, 25), _line("Stress.", 120, 618, 25)]}
    lines = [ln[0] for ln in _block_lines(block, merge_baseline=True)]
    assert _split_references(lines) == [
        "Fennel, B. (2003). Running. doi: 10.1109/ABC.2013.65 50194",
        "Gorse, C. (2012). Stress."]


def test_spacing_accents_are_composed_but_apostrophes_kept():
    assert _fix_accents("R´ena, T., & S´ar˝o, L. and T¨ude, Bˆet´e") == "Réna, T., & Sárő, L. and Tüde, Bêté"
    assert _fix_accents("O´Brien, K.") == "O´Brien, K."


def test_numbered_list_with_numbers_on_their_own_lines():
    lines = ["1.", "Hazel JE, Ivy M. Walking in parks. J Nat. 2024;16(3):174-190.",
             "2.", "Larch MG, et al. Trees and mood. J Aff. 2012;", "140(3):300-305.",
             "3. Maple S. A book. 2nd ed. Dublin: Press; 2008."]
    assert _split_references(lines) == [
        "Hazel JE, Ivy M. Walking in parks. J Nat. 2024;16(3):174-190.",
        "Larch MG, et al. Trees and mood. J Aff. 2012; 140(3):300-305.",
        "Maple S. A book. 2nd ed. Dublin: Press; 2008."]


def test_a_number_out_of_sequence_does_not_start_an_entry():
    lines = ["1. Hazel JE. Title. 2020;", "12. and more of the same entry.", "2. Larch MG. Title. 2021."]
    assert _split_references(lines) == ["Hazel JE. Title. 2020; 12. and more of the same entry.",
                                        "Larch MG. Title. 2021."]


def test_end_matter_is_cut_off():
    lines = ["Quill, A. (2010). Title. J. 1, 1–2.", "Conﬂict of Interest: The authors declare none.",
             "Copyright © 2020 Quill."]
    assert _split_references(lines) == ["Quill, A. (2010). Title. J. 1, 1–2."]
    lines = ["1. Hazel JE. Title. 2020.", "Abbreviations", "VR: virtual reality"]
    assert _split_references(lines) == ["Hazel JE. Title. 2020."]
