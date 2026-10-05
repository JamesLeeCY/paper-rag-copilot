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
  4. Body text starts at the Abstract (or the first numbered heading) and
     stops at References; the blocks after References are parsed as
     reference entries.
"""
from __future__ import annotations

import re
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


def _block_lines(block: dict) -> list[tuple[str, float, bool, bool]]:
    out = []
    for line in block.get("lines", []):
        spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
        if not spans:
            continue
        text = "".join(s["text"] for s in spans).strip()
        size = max(s["size"] for s in spans)
        bold = all((s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower() for s in spans)
        italic = all((s["flags"] & _ITALIC_FLAG) or "italic" in s["font"].lower() for s in spans)
        out.append((text, size, bold, italic))
    return out


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
            blocks.append({"page": pno, "lines": lines})
            for text, size, _, _ in lines:
                size_chars[round(size, 1)] += len(text)
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
        lines = [ln for ln in b["lines"]
                 if _norm_repeat(ln[0]) not in repeated
                 and not re.fullmatch(r"\d{1,4}", ln[0])
                 and not _BOILERPLATE_RE.match(ln[0])]
        if not lines:
            continue
        text = _collapse_spaced(_join_lines([ln[0] for ln in lines]))
        heading = _heading(text, lines, body_size)

        if heading:
            number, title, level = heading
            if title.lower() == "abstract" or number:
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
            ref_lines.extend(ln[0] for ln in lines)
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


def _split_references(lines: list[str]) -> list[str]:
    """Split reference-list lines into entries.

    Typesetters pack many references into one text block, and a long author
    list wraps onto lines that also look like "Surname, I.". So a line starts
    a new entry only if it looks like an author AND the entry being built
    already has its year — i.e. that entry's author list is complete.
    """
    entries: list[list[str]] = []
    for ln in lines:
        if entries and not (_REF_START_RE.match(ln) and _REF_YEAR_RE.search(" ".join(entries[-1]))):
            entries[-1].append(ln)
        else:
            entries.append([ln])
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
