"""Reference audit, part 1: internal consistency of in-text citations and the
reference list (no network).

Checks, per corpus (CORPUS / SOURCE_PATH as for ingest):
  - unresolved in-text citations: (surname, year) not in the reference list,
    split into a year mismatch (the surname is listed with another year), a
    likely misspelling (a listed surname within edit distance 2 for that
    year) and missing;
  - references never cited in the text;
  - duplicate references and same surname + year entries without a/b suffix.

The report holds reference strings and cited surnames, i.e. the corpus's own
bibliography: it is written to REPORT_DIR (gitignored) and only counts are
printed. Run: CORPUS=<name> python -m eval.ref_audit
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict

import config

_YEAR = r"(?:19|20)\d{2}[a-z]?|n\.d\.|in press"
_PAREN_RE = re.compile(r"\(([^()]{4,400})\)")
# One citation inside parentheses: optional lead-in, an author part, years.
_PART_RE = re.compile(
    r"^(?:e\.g\.,?|i\.e\.,?|see(?: also)?|cf\.|for (?:a )?review,?|reviewed in)?\s*"
    r"(?P<authors>(?:(?:de|del|der|den|van|von|da|di|du|le|la|dos|das)\s+)*[A-Z][^\d;]{0,80}?)[,\s]+"
    r"(?P<years>(?:" + _YEAR + r")(?:\s*,\s*(?:" + _YEAR + r"))*)\s*$")
# Narrative citation: Surname (2020), Surname et al. (2020a, 2021), A and B (2020).
_PARTICLE = r"(?:(?:de|del|der|den|van|von|da|di|du|le|la|dos|das)\s+)*"
_NARR_RE = re.compile(
    r"\b(?P<authors>" + _PARTICLE + r"[A-Z][A-Za-zÀ-ɏ'\-]+"
    r"(?:\s+(?:et al\.|and|&)\s*(?:[A-Z][A-Za-zÀ-ɏ'\-]+)?)?)"
    r"\s+\((?P<years>(?:" + _YEAR + r")(?:\s*,\s*(?:" + _YEAR + r"))*)\)")
_SURNAME_RE = re.compile(_PARTICLE + r"[A-Z][A-Za-zÀ-ɏ'\-]+")
_ACRONYM_RE = re.compile(r"^[A-Z0-9\-]{2,}$")
# A surname of up to four words: "De Franco Tobar", "Olvera Alvarez", "van den Berg".
_MULTI_SURNAME_RE = re.compile(r"(?:(?:[A-Z][A-Za-zÀ-ɏ'\-]+|de|del|der|den|van|von|da|di|du|le|la|dos|das)\s+){0,3}"
                               r"[A-Z][A-Za-zÀ-ɏ'\-]+")
_REF_YEAR_RE = re.compile(r"\(((?:19|20)\d{2}[a-z]?|n\.d\.|in press)[,)]", re.I)
# Capitalised words that are not surnames but precede years in prose.
_NOT_SURNAMES = {"In", "Since", "From", "By", "Until", "During", "After", "Before", "Study",
                 "Figure", "Table", "Spring", "Summer", "Autumn", "Fall", "Winter",
                 "January", "February", "March", "April", "May", "June", "July", "August",
                 "September", "October", "November", "December", "Phase", "Wave", "Version"}


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _first_author(authors: str) -> str:
    """First surname-like token, skipping acronyms ("SSQ, Bouchard et al.");
    with "et al." the surname just before it ("SSQ, French version, Bouchard
    et al.")."""
    m = re.search(r"([^,;]+?)\s+et al\.", authors)
    if m:
        name = re.sub(r"^(?:e\.g\.|i\.e\.|see(?: also)?|cf\.)\s*", "", m.group(1).strip())
        if _MULTI_SURNAME_RE.fullmatch(name):
            return name
    for s in _SURNAME_RE.finditer(authors):
        name = s.group(0)
        if not _ACRONYM_RE.match(name.split()[-1]) and name not in _NOT_SURNAMES:
            return name
    return ""


def citations_in(text: str) -> list[tuple[str, str]]:
    """(first-author surname, year with suffix) for every in-text citation.
    Pass the whole body as one text: a citation group can run across a page
    or paragraph break."""
    text = _POSSESSIVE_RE.sub(r"\1 (", text)          # Smith's (2001) -> Smith (2001)
    out = []
    for m in _PAREN_RE.finditer(text):
        for part in m.group(1).split(";"):
            p = _PART_RE.match(part.strip())
            name = _first_author(p.group("authors")) if p else ""
            if name:
                out += [(name, y.strip()) for y in p.group("years").split(",")]
    for m in _NARR_RE.finditer(text):
        name = _first_author(m.group("authors"))
        if name:
            out += [(name, y.strip()) for y in m.group("years").split(",")]
    return out


_POSSESSIVE_RE = re.compile(r"\b([A-Z][A-Za-zÀ-ɏ\-]+)['’]s\s+\(")


def ref_surname(ref) -> str:
    """First-author surname from the raw reference, particles included."""
    head = ref.raw.strip().split(",")[0].strip()
    if _MULTI_SURNAME_RE.fullmatch(head):
        return head
    m = _SURNAME_RE.match(ref.raw.strip())
    return m.group(0) if m else ref.surname


def ref_year(ref) -> str:
    """The reference's year with its a/b suffix when the raw string has one."""
    m = _REF_YEAR_RE.search(ref.raw)
    return m.group(1).lower() if m else ref.year


def audit(body_texts: list[str], refs: list) -> dict:
    keyed = {(_fold(ref_surname(r)), ref_year(r)): r for r in refs}
    by_surname = defaultdict(set)
    by_year = defaultdict(set)
    for (s, y) in keyed:
        by_surname[s].add(y)
        by_year[y.rstrip("abcdefgh")].add(s)
    ref_text = _fold(" ".join(r.raw for r in refs))
    cites = Counter(citations_in("\n".join(body_texts)))
    cited_keys, unresolved = set(), []
    for (surname, year), n in cites.items():
        k = (_fold(surname), year.lower())
        base = (k[0], year.lower().rstrip("abcdefgh"))
        if k in keyed or base in keyed:
            cited_keys.add(k if k in keyed else base)
            continue
        kind, near = "missing", ""
        if k[0] in by_surname:
            kind, near = "year_mismatch", ", ".join(sorted(by_surname[k[0]]))
        else:
            close = [s for s in by_year.get(base[1], ()) if _edit(s, k[0]) <= 2]
            if close:
                kind, near = "misspelling", ", ".join(close)
        # Present in the reference text but not as its own entry: the list was
        # split wrongly (PDF columns), not a citation error.
        if kind != "misspelling" and re.search(
                re.escape(k[0]) + r"\b.{0,200}?\(" + re.escape(base[1]), ref_text):
            kind = "reference_split_error"
        unresolved.append({"surname": surname, "year": year, "count": n, "kind": kind,
                           "listed": near})
    uncited = [r.raw for k, r in keyed.items() if k not in cited_keys]
    norm = Counter(re.sub(r"\W+", " ", _fold(r.raw))[:80] for r in refs)
    duplicates = [k for k, n in norm.items() if n > 1]
    plain = Counter((_fold(ref_surname(r)), ref_year(r)) for r in refs if not ref_year(r)[-1:].isalpha())
    ambiguous = [f"{s} {y}" for (s, y), n in plain.items() if n > 1]
    return {
        "n_references": len(refs), "n_citation_mentions": sum(cites.values()),
        "n_distinct_citations": len(cites),
        "unresolved": sorted(unresolved, key=lambda u: (u["kind"], u["surname"])),
        "uncited_references": uncited, "duplicate_references": duplicates,
        "ambiguous_same_surname_year": ambiguous,
    }


def main() -> None:
    from src.ingest import load_document

    source = config.SOURCE_PATH
    if source.suffix.lower() == ".pdf":
        from src.ingest_pdf import load_pdf
        body, refs = load_pdf(source)
    else:
        body, refs = load_document(source)
    res = audit([b.text for b in body], refs)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / "ref_audit.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = Counter(u["kind"] for u in res["unresolved"])
    print(json.dumps({"corpus": config.CORPUS, "references": res["n_references"],
                      "citation_mentions": res["n_citation_mentions"],
                      "distinct_citations": res["n_distinct_citations"],
                      "unresolved": dict(kinds), "uncited_references": len(res["uncited_references"]),
                      "duplicates": len(res["duplicate_references"]),
                      "ambiguous": len(res["ambiguous_same_surname_year"]), "report": str(path)}))


if __name__ == "__main__":
    main()
