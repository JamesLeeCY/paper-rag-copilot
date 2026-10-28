"""PDF ingestion — a journal article PDF -> body paragraphs + references.

Produces the same ``BodyPara`` / ``Reference`` records as the .docx path in
``src.ingest``, so both chunkers and everything downstream work unchanged.
Uses PyMuPDF's text blocks, which in typical journal PDFs (including
two-column layouts) already come out one paragraph per block in reading
order.

Heuristics, in order:
  1. Body font size = the most common span size, weighted by characters.
  2. Running headers/footers and page numbers are dropped: lines whose
     digit-stripped text repeats on many pages, and bare page numbers.
  3. A block is a heading when it is short and either numbered
     ("2.3 Measures") or a known section name ("Abstract", "References"),
     set larger than body text or in bold. Letter-spaced headings
     ("A B S T R A C T") are collapsed first.
  4. Body text starts at the Abstract, the first numbered heading or an
     Introduction/Background heading (for an unlabelled abstract), and
     stops at References; the blocks after References are parsed as
     reference entries.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from src.ingest import BodyPara, Reference, parse_references

# Number then title; journals often omit the space ("1.Objectives", "2.1.Ethics statement").
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s*([A-Z].{1,110})$")
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "method", "methods",
    "materials and methods", "results", "discussion", "general discussion",
    "conclusion", "conclusions", "limitations", "references",
    "acknowledgements", "acknowledgments", "funding", "data availability",
    "declaration of competing interest", "credit authorship contribution statement",
    "appendix", "supplementary material", "supplementary data",
}
_STOP_SECTIONS = {"references"}
_BOLD_FLAG = 1 << 4
_ITALIC_FLAG = 1 << 1
# Publication boilerplate that typesetters drop into the reading order (page 1).
_BOILERPLATE_RE = re.compile(
    r"^(\*?corresponding author|e-?mail address|contents lists available|journal homepage|"
    r"https?://doi\.org|received \d|available online|this is an open access article|"
    r"\d{4}-\d{3}[\dx]/|©)", re.I)
# Frontiers (2024 layout) prints article metadata in a page-1 sidebar whose
# blocks fall inside the body's reading order; drop a block opening with one
# of these labels. Case-sensitive: the labels are set in capitals.
_SIDEBAR_RE = re.compile(
    r"^(?:OPEN ACCESS|EDITED BY|REVIEWED BY|\*?CORRESPONDENCE|RECEIVED \d|ACCEPTED \d|"
    r"PUBLISHED \d|CITATION\b|COPYRIGHT\b|TYPE [A-Z]|DOI 10\.)")
# A line that is only a DOI (a running header with too few letters for the
# repeat check) or only an e-mail address.
_DOI_WRAP_RE = re.compile(r"\b10\.\d{4,9}/\S*[.\-/]$")
_BARE_ID_RE = re.compile(r"^(?:(?:DOI:? )?10\.\d{4,9}/\S+|\S+@\S+\.[A-Za-z]{2,})$")
# Glyphs of math fonts without a Unicode mapping come out as control characters.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _collapse_spaced(text: str) -> str:
    """'A B S T R A C T' -> 'ABSTRACT' (Elsevier letter-spaced headings)."""
    if re.fullmatch(r"(?:[A-Za-z] ){3,}[A-Za-z]", text.strip()):
        return text.replace(" ", "")
    return text


def _norm_repeat(text: str) -> str | None:
    """Letters-only key for spotting running headers/footers.

    Letters only, because the same header is often typeset with and without
    spaces ("Journal of X 115 (2026)" on page 1, "JournalofX115(2026)" later).
    Short keys return None: words like "Note." legitimately recur on many
    pages and must not be treated as a header.
    """
    key = re.sub(r"[^a-z]", "", text.lower())
    return key if len(key) >= 12 else None


def _block_lines(block: dict, merge_baseline: bool = False) -> list[tuple[str, float, bool, bool]]:
    """(text, size, bold, italic) per line. With ``merge_baseline``, PDF lines
    on the same baseline are joined first: Frontiers (2020) sets justified
    reference lists one word per PDF line, so "Bowler," and a DOI stood alone."""
    groups: list[list[dict]] = []
    for line in block.get("lines", []):
        prev = groups[-1][-1]["bbox"] if merge_baseline and groups else None
        if (prev and abs(line["bbox"][3] - prev[3]) < 1.0
                and line["bbox"][0] >= prev[2] - 1.0):
            groups[-1].append(line)
        else:
            groups.append([line])
    out = []
    for group in groups:
        texts, spans = [], []
        for line in group:
            all_spans = line.get("spans", [])
            # Join every span, whitespace-only ones included: some typesetters
            # (Frontiers) emit each inter-word space as its own span, and dropping
            # them merged the words ("dynamicallychangingVREs").
            texts.append("".join(s.get("text", "") for s in all_spans).strip())
            spans += [s for s in all_spans if s.get("text", "").strip()]
        if not spans:
            continue
        text = " ".join(t for t in texts if t)
        size = max(s["size"] for s in spans)
        bold = all((s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower() for s in spans)
        italic = all((s["flags"] & _ITALIC_FLAG) or "italic" in s["font"].lower() for s in spans)
        out.append((text, size, bold, italic))
    return out


# Spacing accents some PDFs emit before the letter ("P´oka", "M´er˝o").
_SPACING_ACCENTS = {"´": "́", "˝": "̋", "¨": "̈",
                    "ˆ": "̂", "˜": "̃", "ˇ": "̌"}
# Only between letters, before a lower-case one: "O´Brien" is an apostrophe.
_SPACING_ACCENT_RE = re.compile("(?<=[A-Za-z])([" + "".join(_SPACING_ACCENTS) + r"])([a-z])")


def _fix_accents(text: str) -> str:
    """'P´oka' -> 'Póka'."""
    return unicodedata.normalize("NFC", _SPACING_ACCENT_RE.sub(
        lambda m: m.group(2) + _SPACING_ACCENTS[m.group(1)], text))


def _join_lines(lines: list[str]) -> str:
    """Join PDF lines into one paragraph, undoing end-of-line hyphenation."""
    text = ""
    for ln in lines:
        if text.endswith("\u00ad"):            # soft hyphen: always a word break
            text = text[:-1] + ln
        elif text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln          # "restor-" + "ation" -> "restoration"
        elif text:
            text += " " + ln
        else:
            text = ln
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


def load_pdf(path: Path) -> tuple[list[BodyPara], list[Reference]]:
    doc = fitz.open(str(path))

    # Pass 1: collect blocks with their lines, and body-size / repeat stats.
    blocks: list[dict] = []
    size_chars: Counter = Counter()
    line_pages: dict[str, set[int]] = {}
    for pno, page in enumerate(doc, 1):
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines = _block_lines(b)
            if not lines:
                continue
            merged = _block_lines(b, merge_baseline=True)   # used in the reference list
            blocks.append({"page": pno, "lines": lines, "merged": merged})
            for text, size, _, _ in lines:
                size_chars[round(size, 1)] += len(text)
                line_pages.setdefault(_norm_repeat(text), set()).add(pno)
            for text, *_ in merged:
                line_pages.setdefault(_norm_repeat(text), set()).add(pno)
    if not blocks:
        raise ValueError(f"No extractable text in {path} (scanned PDF? needs OCR)")
    body_size = size_chars.most_common(1)[0][0]
    n_pages = len(doc)
    repeated = {t for t, pages in line_pages.items()
                if t and n_pages >= 3 and len(pages) >= max(3, n_pages // 2)}

    # Pass 2: walk blocks in order, tracking sections.
    body: list[BodyPara] = []
    ref_lines: list[str] = []
    section_path: list[str] = []
    section_number = ""
    started = in_refs = False
    idx = 0
    for b in blocks:
        if _SIDEBAR_RE.match(b["lines"][0][0]):
            continue
        # A bare number is a page number, unless it is the wrapped tail of a
        # DOI ("doi:10.3357/AMHP.4747." + "2017").
        block_lines = b["merged"] if in_refs else b["lines"]
        lines = [ln for i, ln in enumerate(block_lines)
                 if _norm_repeat(ln[0]) not in repeated
                 and not (re.fullmatch(r"\d{1,4}", ln[0])
                          and not (i and _DOI_WRAP_RE.search(block_lines[i - 1][0])))
                 and not _BOILERPLATE_RE.match(ln[0])
                 and not _BARE_ID_RE.match(ln[0])]
        if not lines:
            continue
        text = _collapse_spaced(_join_lines([ln[0] for ln in lines]))
        if _CONTROL_RE.search(text) or len(text) <= 2:
            continue                        # garbled equation fragment
        heading = _heading(text, lines, body_size)

        if heading:
            number, title, level = heading
            # Body starts at the Abstract, a numbered heading, or - for journals
            # that print the abstract unlabelled (Frontiers) - the Introduction.
            if title.lower() in ("abstract", "introduction", "background") or number:
                started = True
            if title.lower() in _STOP_SECTIONS:
                in_refs = True
                continue
            if in_refs:                     # e.g. an Appendix after References
                in_refs = title.lower() in _STOP_SECTIONS
            section_path = section_path[: level - 1] + [f"{number} {title}".strip()]
            section_number = number
            continue

        if in_refs:
            ref_lines.extend(_fix_accents(ln[0]) for ln in lines)
            continue
        if not started:
            continue                        # title page, authors, affiliations
        # Many journals give the introduction no heading: it simply follows
        # the abstract, set at body size while the abstract is smaller.
        in_abstract = bool(section_path) and section_path[-1].lower() == "abstract"
        if in_abstract and max(ln[1] for ln in lines) >= body_size - 0.05:
            section_path, section_number = ["Introduction"], ""
        body.append(BodyPara(idx=idx, text=text, section_path=list(section_path),
                             section_number=section_number, page=b["page"]))
        idx += 1

    if not body:
        raise ValueError(f"Found no body text in {path}: no Abstract or numbered heading detected")
    return _merge_tables(body), parse_references(_split_references(ref_lines))


_TABLE_CAPTION_RE = re.compile(r"^Table \d+[a-z]?\b")


def _numeric_ratio(text: str) -> float:
    toks = text.split()
    return sum(bool(re.fullmatch(r"[\d.,%()\-–−=<>±*]+", t)) for t in toks) / max(1, len(toks))


def _is_prose(text: str) -> bool:
    return len(text) >= 120 and _numeric_ratio(text) <= 0.3 and not text.startswith("Note")


def _merge_tables(body: list[BodyPara]) -> list[BodyPara]:
    """Fold a table's fragments into one paragraph.

    Text extraction shatters a table into one block per cell group — bare
    numbers like "65.60 (61) 61.70 (29)" with their row and column labels in
    other blocks. Everything from a "Table N" caption up to the next prose
    paragraph becomes one paragraph, so the table is retrieved as a unit.
    """
    out: list[BodyPara] = []
    i = 0
    while i < len(body):
        p = body[i]
        if not _TABLE_CAPTION_RE.match(p.text):
            out.append(p)
            i += 1
            continue
        parts = [p.text]
        j = i + 1
        while j < len(body) and not _is_prose(body[j].text) \
                and not _TABLE_CAPTION_RE.match(body[j].text):
            parts.append(body[j].text)
            j += 1
        out.append(BodyPara(idx=p.idx, text=" | ".join(parts), section_path=p.section_path,
                            section_number=p.section_number, page=p.page))
        i = j
    return out


# A reference entry starts with "Surname, I." (or "Surname-Name, I.", "van Surname, I.").
_REF_START_RE = re.compile(r"^(?:[a-z]{1,3} )?[A-Z][A-Za-zÀ-ɏ'’\-]+(?: [A-Z][A-Za-zÀ-ɏ'’\-]+)*, (?:[A-Z]\.|[A-Z][a-z])")
_REF_YEAR_RE = re.compile(r"\((?:19|20)\d{2}[a-z]?[,)]|\(n\.d\.\)|\(in press\)", re.I)


# End matter typeset right after the reference list (Frontiers, JMIR).
_REF_END_RE = re.compile(r"^(?:Abbreviations$|Conflicts? of Interest\b|Copyright ©|Edited by\b|"
                         r"Publisher.s Note\b)", re.I)


def _split_references(lines: list[str]) -> list[str]:
    """Split reference-list lines into entries.

    Typesetters pack many references into one text block, and a long author
    list wraps onto lines that also look like "Surname, I.". So a line starts
    a new entry only if it looks like an author AND the entry being built
    already has its year — i.e. that entry's author list is complete.
    The list ends at the first line of end matter.
    """
    end = next((i for i, ln in enumerate(lines)
                if _REF_END_RE.match(unicodedata.normalize("NFKC", ln))), len(lines))
    lines = lines[:end]
    if lines and _REF_NUMBER_RE.match(lines[0]) and _REF_NUMBER_RE.match(lines[0]).group(1) == "1":
        return _split_numbered(lines)
    entries: list[list[str]] = []
    for ln in lines:
        if entries and not (_REF_START_RE.match(ln) and _REF_YEAR_RE.search(" ".join(entries[-1]))):
            entries[-1].append(ln)
        else:
            entries.append([ln])
    return [_join_lines(e) for e in entries]


# A numbered list (Vancouver, JMIR): "1. Bettmann JE, ..." or "1." on its own line.
_REF_NUMBER_RE = re.compile(r"^(\d{1,3})\.(?:\s+(?=\S)|$)")


def _split_numbered(lines: list[str]) -> list[str]:
    """Split a numbered reference list. A line starts entry n+1 only if it
    begins with exactly that number, so numbers inside an entry ("12. ed.",
    a wrapped "2024." line) do not split it. The numbers are removed."""
    entries: list[list[str]] = []
    for ln in lines:
        m = _REF_NUMBER_RE.match(ln)
        if m and int(m.group(1)) == len(entries) + 1:
            entries.append([ln[m.end():]] if ln[m.end():] else [])
        elif entries:
            entries[-1].append(ln)
    return [_join_lines(e) for e in entries]


def _heading(text: str, lines, body_size: float) -> tuple[str, str, int] | None:
    """(section number, title, level) when the block is a section heading."""
    if len(text) > 120 or len(lines) > 2:
        return None
    larger = max(ln[1] for ln in lines) >= body_size + 0.8
    bold = all(ln[2] for ln in lines)
    italic = all(ln[3] for ln in lines)
    m = _NUMBERED_RE.match(text)
    # Numbered: set apart by size, bold, or italic (italic is common for
    # second-level headings, e.g. "2.1.Participants").
    if m and (larger or bold or italic):
        number = m.group(1)
        return number, m.group(2).strip(), number.count(".") + 1
    title = text.strip().rstrip(".:")
    # Named: set apart by size, bold, or all caps ("ABSTRACT", "REFERENCES").
    if title.lower() in _KNOWN_SECTIONS and (larger or bold or title.isupper()):
        return "", title.title() if title.isupper() else title, 1
    return None


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Inspect how a PDF is parsed")
    ap.add_argument("pdf")
    args = ap.parse_args()
    body, refs = load_pdf(Path(args.pdf))
    last = None
    for p in body:
        sec = " > ".join(p.section_path)
        if sec != last:
            print(f"\n## {sec}  (p. {p.page})")
            last = sec
        print(f"  [{p.idx}] {p.text[:110]}")
    print(f"\n{len(body)} paragraphs, {len(refs)} references")
