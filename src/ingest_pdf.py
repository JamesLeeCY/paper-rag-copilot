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

_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z].{1,110})$")
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


def _collapse_spaced(text: str) -> str:
    """'A B S T R A C T' -> 'ABSTRACT' (Elsevier letter-spaced headings)."""
    if re.fullmatch(r"(?:[A-Za-z] ){3,}[A-Za-z]", text.strip()):
        return text.replace(" ", "")
    return text


def _norm_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _block_lines(block: dict) -> list[tuple[str, float, bool]]:
    out = []
    for line in block.get("lines", []):
        spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
        if not spans:
            continue
        text = "".join(s["text"] for s in spans).strip()
        size = max(s["size"] for s in spans)
        bold = all((s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower() for s in spans)
        out.append((text, size, bold))
    return out


def _join_lines(lines: list[str]) -> str:
    """Join PDF lines into one paragraph, undoing end-of-line hyphenation."""
    text = ""
    for ln in lines:
        if text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln          # "restor-" + "ation" -> "restoration"
        elif text:
            text += " " + ln
        else:
            text = ln
    return re.sub(r"\s+", " ", text).strip()


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
            for text, size, _ in lines:
                size_chars[round(size, 1)] += len(text)
                line_pages.setdefault(_norm_repeat(text), set()).add(pno)
    if not blocks:
        raise ValueError(f"No extractable text in {path} (scanned PDF? needs OCR)")
    body_size = size_chars.most_common(1)[0][0]
    n_pages = len(doc)
    repeated = {t for t, pages in line_pages.items()
                if n_pages >= 3 and len(pages) >= max(3, n_pages // 2)}

    # Pass 2: walk blocks in order, tracking sections.
    body: list[BodyPara] = []
    ref_texts: list[str] = []
    section_path: list[str] = []
    section_number = ""
    started = in_refs = False
    idx = 0
    for b in blocks:
        lines = [(t, s, bd) for t, s, bd in b["lines"]
                 if _norm_repeat(t) not in repeated and not re.fullmatch(r"\d{1,4}", t)]
        if not lines:
            continue
        text = _collapse_spaced(_join_lines([t for t, _, _ in lines]))
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
            ref_texts.append(text)
            continue
        if not started:
            continue                        # title page, authors, affiliations
        body.append(BodyPara(idx=idx, text=text, section_path=list(section_path),
                             section_number=section_number, page=b["page"]))
        idx += 1

    if not body:
        raise ValueError(f"Found no body text in {path}: no Abstract or numbered heading detected")
    return body, parse_references(ref_texts)


def _heading(text: str, lines, body_size: float) -> tuple[str, str, int] | None:
    """(section number, title, level) when the block is a section heading."""
    if len(text) > 120 or len(lines) > 2:
        return None
    larger = max(s for _, s, _ in lines) >= body_size + 0.8
    bold = all(bd for _, _, bd in lines)
    if not (larger or bold):
        return None
    m = _NUMBERED_RE.match(text)
    if m:
        number = m.group(1)
        return number, m.group(2).strip(), number.count(".") + 1
    title = text.strip().rstrip(".:")
    if title.lower() in _KNOWN_SECTIONS:
        return "", title, 1
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
