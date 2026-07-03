"""Ingestion Layer — dissertation .docx -> structured chunks + references.

Responsibilities
----------------
1. Walk the manuscript, tracking the live section hierarchy (H1>H2>H3>H4).
2. Split off the ``References`` section and parse each entry into structured
   metadata (first-author surname, year, display label).
3. Extract inline citations, e.g. ``(Kaplan, 1995)`` / ``Ulrich et al. (1991)``,
   and resolve them to reference ids so every chunk knows which cited works
   back it up.
4. Emit chunks via two strategies (fixed-size baseline vs. section-aware) so
   the evaluation harness can run the chunking ablation.

Every chunk keeps enough metadata (section number, paragraph range, cited
references) that a generated citation can be traced back to the exact spot in
the manuscript. That traceability is what makes the grounding claims auditable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import docx

import config


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
@dataclass
class Reference:
    ref_id: str
    surname: str          # first-author surname, for inline-citation matching
    year: str
    display: str          # e.g. "Kaplan (1995)"
    raw: str              # full reference string


@dataclass
class BodyPara:
    idx: int              # docx paragraph index (stable locator)
    text: str
    section_path: list[str]
    section_number: str   # e.g. "1.5" ("" if none parsed)


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    text: str
    section: str          # deepest section title
    section_number: str
    section_path: list[str]
    para_start: int
    para_end: int
    n_words: int
    est_tokens: int
    citations: list[dict] = field(default_factory=list)
    source: str = config.SOURCE_LABEL


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SECTION_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\b")
# Inline citation: an author-ish token (possibly "et al." / "&") + a 4-digit year.
_CITE_RE = re.compile(
    r"([A-Z][A-Za-zÀ-ɏ'\-]+)"          # a capitalized surname
    r"(?:\s+(?:et\s+al\.?|and|&|,)[^()\n;]{0,60}?)?"  # optional co-authors
    r",?\s*"
    r"((?:19|20)\d{2})[a-z]?"                       # year
)


def est_tokens(text: str) -> int:
    return int(len(text.split()) * config.TOKENS_PER_WORD)


def _is_section_title(style_name: str, text: str) -> bool:
    """A real section title: a Heading style AND short.

    The manuscript mislabels some long body paragraphs / table captions as
    ``Heading 3``; those should be treated as content, not as new sections.
    """
    return style_name.startswith("Heading") and 0 < len(text) <= 120


# --------------------------------------------------------------------------
# References parsing
# --------------------------------------------------------------------------
def parse_references(ref_texts: list[str]) -> list[Reference]:
    refs: list[Reference] = []
    for i, raw in enumerate(ref_texts):
        raw = raw.strip()
        if not raw:
            continue
        year_m = re.search(r"\((?:.*?)((?:19|20)\d{2})[a-z]?\)", raw) or re.search(
            r"\b((?:19|20)\d{2})[a-z]?\b", raw
        )
        year = year_m.group(1) if year_m else "n.d."
        # First-author surname = leading token before first comma/&/(.
        surname_m = re.match(r"\s*([A-Z][A-Za-zÀ-ɏ'\-]+)", raw)
        surname = surname_m.group(1) if surname_m else raw[:20]
        refs.append(
            Reference(
                ref_id=f"ref_{i+1:04d}",
                surname=surname,
                year=year,
                display=f"{surname} ({year})",
                raw=raw,
            )
        )
    return refs


def build_ref_lookup(refs: list[Reference]) -> dict[tuple[str, str], Reference]:
    return {(r.surname.lower(), r.year): r for r in refs}


def extract_citations(
    text: str, ref_lookup: dict[tuple[str, str], Reference]
) -> list[dict]:
    """Return unique cited references detected in ``text``."""
    seen: dict[str, dict] = {}
    for m in _CITE_RE.finditer(text):
        surname, year = m.group(1), m.group(2)
        ref = ref_lookup.get((surname.lower(), year))
        display = ref.display if ref else f"{surname} ({year})"
        ref_id = ref.ref_id if ref else ""
        key = ref_id or display
        if key not in seen:
            seen[key] = {"ref_id": ref_id, "display": display, "resolved": bool(ref)}
    return list(seen.values())


# --------------------------------------------------------------------------
# Document walk
# --------------------------------------------------------------------------
def load_document(path: Path) -> tuple[list[BodyPara], list[Reference]]:
    """Split the manuscript into (body paragraphs, references)."""
    doc = docx.Document(str(path))
    paras = doc.paragraphs

    # Locate the English body start ("Abstract") and the References heading.
    body_start = 0
    refs_start = len(paras)
    for i, p in enumerate(paras):
        t = p.text.strip()
        if _is_section_title(p.style.name, t):
            if t.lower() == "abstract" and body_start == 0:
                body_start = i
            if t.lower() == "references":
                refs_start = i
                break

    body: list[BodyPara] = []
    section_path: list[str] = []
    section_number = ""
    for i in range(body_start, refs_start):
        p = paras[i]
        t = p.text.strip()
        if not t:
            continue
        if _is_section_title(p.style.name, t):
            level = int(p.style.name.replace("Heading", "").strip() or "1")
            section_path = section_path[: level - 1] + [t]
            num_m = _SECTION_NUM_RE.match(t)
            section_number = num_m.group(1) if num_m else ""
            continue
        body.append(
            BodyPara(
                idx=i,
                text=t,
                section_path=list(section_path),
                section_number=section_number,
            )
        )

    ref_texts = [paras[i].text for i in range(refs_start + 1, len(paras))]
    references = parse_references(ref_texts)
    return body, references


# --------------------------------------------------------------------------
# Chunkers
# --------------------------------------------------------------------------
def _mk_chunk(
    strategy: str,
    idx: int,
    paras: list[BodyPara],
    ref_lookup: dict[tuple[str, str], Reference],
) -> Chunk:
    text = "\n".join(p.text for p in paras)
    deepest = paras[-1].section_path[-1] if paras[-1].section_path else ""
    return Chunk(
        chunk_id=f"c_{strategy}_{idx:04d}",
        strategy=strategy,
        text=text,
        section=deepest,
        section_number=paras[-1].section_number,
        section_path=paras[-1].section_path,
        para_start=paras[0].idx,
        para_end=paras[-1].idx,
        n_words=len(text.split()),
        est_tokens=est_tokens(text),
        citations=extract_citations(text, ref_lookup),
    )


def chunk_section_aware(
    body: list[BodyPara], ref_lookup
) -> list[Chunk]:
    """Group paragraphs by (deepest) section, splitting oversize sections and
    merging tiny trailing paragraphs so no argument is cut mid-thought."""
    chunks: list[Chunk] = []
    idx = 0

    # Group consecutive paragraphs sharing the same section path.
    groups: list[list[BodyPara]] = []
    for p in body:
        key = tuple(p.section_path)
        if groups and tuple(groups[-1][0].section_path) == key:
            groups[-1].append(p)
        else:
            groups.append([p])

    for group in groups:
        buf: list[BodyPara] = []
        buf_tokens = 0
        for p in group:
            pt = est_tokens(p.text)
            if buf and buf_tokens + pt > config.SECTION_MAX_TOKENS:
                chunks.append(_mk_chunk("section", idx, buf, ref_lookup))
                idx += 1
                buf, buf_tokens = [], 0
            buf.append(p)
            buf_tokens += pt
        if buf:
            # Merge a tiny tail into the previous chunk of the same section.
            if (
                buf_tokens < config.SECTION_MIN_TOKENS
                and chunks
                and chunks[-1].section_path == buf[-1].section_path
            ):
                prev = chunks.pop()
                merged_text = prev.text + "\n" + "\n".join(p.text for p in buf)
                chunks.append(
                    Chunk(
                        chunk_id=prev.chunk_id,
                        strategy="section",
                        text=merged_text,
                        section=prev.section,
                        section_number=prev.section_number,
                        section_path=prev.section_path,
                        para_start=prev.para_start,
                        para_end=buf[-1].idx,
                        n_words=len(merged_text.split()),
                        est_tokens=est_tokens(merged_text),
                        citations=extract_citations(merged_text, ref_lookup),
                    )
                )
            else:
                chunks.append(_mk_chunk("section", idx, buf, ref_lookup))
                idx += 1
    return chunks


def chunk_fixed(body: list[BodyPara], ref_lookup) -> list[Chunk]:
    """Baseline: fixed-size sliding window over the concatenated body, respecting
    paragraph boundaries for the window edges."""
    chunks: list[Chunk] = []
    idx = 0
    i = 0
    n = len(body)
    while i < n:
        buf: list[BodyPara] = []
        buf_tokens = 0
        j = i
        while j < n and buf_tokens < config.FIXED_CHUNK_TOKENS:
            buf.append(body[j])
            buf_tokens += est_tokens(body[j].text)
            j += 1
        chunks.append(_mk_chunk("fixed", idx, buf, ref_lookup))
        idx += 1
        if j >= n:
            break
        # Step back by roughly the overlap budget (in paragraphs).
        overlap_tokens = 0
        step_back = 0
        k = j - 1
        while k > i and overlap_tokens < config.FIXED_CHUNK_OVERLAP:
            overlap_tokens += est_tokens(body[k].text)
            step_back += 1
            k -= 1
        i = max(i + 1, j - step_back)
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "section": chunk_section_aware}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def ingest(strategy: str, source: Path | None = None) -> list[Chunk]:
    source = Path(source) if source else config.SOURCE_DOCX
    body, references = load_document(source)
    ref_lookup = build_ref_lookup(references)

    # Persist references once (shared across strategies).
    refs_out = config.DATA_DIR / "references.jsonl"
    with refs_out.open("w", encoding="utf-8") as f:
        for r in references:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    chunks = CHUNKERS[strategy](body, ref_lookup)

    out = config.chunks_path(strategy)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(strategy: str) -> list[Chunk]:
    path = config.chunks_path(strategy)
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def load_references() -> list[Reference]:
    path = config.DATA_DIR / "references.jsonl"
    refs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            refs.append(Reference(**json.loads(line)))
    return refs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ingest dissertation -> chunks")
    ap.add_argument("--strategy", choices=list(CHUNKERS), default=config.CHUNK_STRATEGY)
    ap.add_argument("--all", action="store_true", help="run both strategies")
    args = ap.parse_args()

    strategies = list(CHUNKERS) if args.all else [args.strategy]
    for strat in strategies:
        chunks = ingest(strat)
        refs = load_references()
        toks = [c.est_tokens for c in chunks]
        print(
            f"[{strat}] {len(chunks)} chunks | "
            f"est_tokens min/median/max = "
            f"{min(toks)}/{sorted(toks)[len(toks)//2]}/{max(toks)} | "
            f"{len(refs)} references"
        )
