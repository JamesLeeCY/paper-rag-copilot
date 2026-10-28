"""Reference audit, part 2: look each reference up in Crossref and PubMed.

Per reference: the listed DOI is resolved at Crossref; if there is none, or it
points to another work or is not registered, Crossref's bibliographic search
is tried with the reference string. A record counts as the same work only if
its title words appear in the reference and its first author matches the
listed one (a bare search hit by the same author can be another of their
papers). Then year, volume, first page, DOI and author count are compared,
and PubMed (by DOI) supplies the PMID and the retraction flag.

Statuses: verified; mismatch (with the fields); retracted; not_found (manual
check — books, chapters, manuals and software are often not in Crossref);
error (network). "Not in PubMed" alone is never a finding: many psychology,
environment and HCI journals are not indexed there.

Only bibliographic data leaves the machine (the reference string, its DOI).
No e-mail address is sent unless CROSSREF_MAILTO / NCBI_EMAIL are set.
Responses are cached under .cache/ref_lookup (trimmed to the fields used);
the report goes to REPORT_DIR (gitignored) and only counts are printed.
Run: CORPUS=<name> python -m eval.ref_audit --lookup
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import config
from eval.ref_audit import _edit, _fold, ref_surname, ref_year

CACHE_DIR = config.ROOT / ".cache" / "ref_lookup"
CROSSREF = "https://api.crossref.org/works"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Seconds between uncached calls per host: NCBI allows 3/s without a key;
# Crossref's public pool asks for restraint.
_MIN_GAP = {"api.crossref.org": 1.0, "eutils.ncbi.nlm.nih.gov": 0.4}
_RETRACTION_TYPES = {"retraction", "removal", "withdrawal"}

_DOI_RE = re.compile(r"(?:doi:\s*|doi\.org/)(10\.\s?\d{4,9}/.+)$", re.I)
_VOL_PAGE_RE = re.compile(
    r"[,\s](?P<vol>\d{1,4})(?:\s*\((?P<issue>[^)]{1,12})\))?,\s*(?P<page>[A-Za-z]?\d+)"
    r"(?:\s*[–\-]\s*[A-Za-z]?\d+)?\s*(?:\.|$)")
# "Surname, I. I." — lowercase too ("schraefel, m. c.").
_AUTHOR_RE = re.compile(r"(?<!\w)[A-Za-zÀ-ɏ][\w'’\-À-ɏ ]*?,\s(?:[A-Za-z][a-zA-Z]?\.[\s\-]?)+")
_STOP = {"the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "its"}


# --- the reference as listed ---------------------------------------------------

def listed_fields(ref) -> dict:
    raw = ref.raw.strip()
    year = ref_year(ref)
    doi = ""
    m = _DOI_RE.search(raw)
    if m:   # rejoin a DOI wrapped across PDF lines ("10. 1016/j.x", "41. 1.02")
        doi = re.sub(r"\s+", "", m.group(1)).rstrip(".").lower()
    body = raw[: m.start()] if m else raw
    head, sep, tail = body.partition(f"({year}")
    if not (year and sep):
        head, tail = body, body
    vp = None
    for vp in _VOL_PAGE_RE.finditer(tail):
        pass
    etal = bool(re.search(r"et al\.|\.\.\.|…", head))
    names = [re.sub(r"^(?:&|and)\s+", "", a.split(",")[0].strip()) for a in _AUTHOR_RE.findall(head)]
    return {
        "surname": ref_surname(ref), "year": year.rstrip("abcdefgh"), "doi": doi,
        "volume": vp.group("vol") if vp else "", "first_page": vp.group("page") if vp else "",
        "n_authors": len(names), "surnames": names, "et_al": etal,
        "non_article": not vp,      # no "volume, page": a book, chapter, manual …
    }


# --- HTTP with a response cache --------------------------------------------------

class Client:
    """GET JSON with a file cache. `fetch(url) -> dict | None` (None = 404)
    can be replaced, e.g. by recorded fixtures in tests."""

    def __init__(self, cache_dir: Path | None = CACHE_DIR, fetch=None):
        self.cache_dir = cache_dir
        self.fetch = fetch or self._http
        self._last: dict[str, float] = {}
        self.calls = 0

    def get(self, url: str, trim) -> dict | None:
        path = None
        if self.cache_dir:
            path = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))["body"]
        body = self.fetch(url)
        body = trim(body) if body is not None else None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "body": body}, ensure_ascii=False),
                            encoding="utf-8")
        return body

    def _http(self, url: str) -> dict | None:
        host = urllib.parse.urlsplit(url).hostname or ""
        wait = self._last.get(host, 0) + _MIN_GAP.get(host, 1.0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        ua = "paper-RAG-copilot reference audit"
        if os.environ.get("CROSSREF_MAILTO"):
            ua += f" (mailto:{os.environ['CROSSREF_MAILTO']})"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        for attempt in range(4):
            self._last[host] = time.monotonic()
            self.calls += 1
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
            time.sleep(2 ** (attempt + 1))
        return None


# --- Crossref and PubMed ---------------------------------------------------------

def _trim_work(w: dict) -> dict:
    keep = ("DOI", "title", "author", "issued", "published-print", "published-online",
            "volume", "page", "article-number", "type", "container-title", "updated-by")
    out = {k: w[k] for k in keep if k in w}
    if "author" in out:
        out["author"] = [{k: a[k] for k in ("family", "name") if k in a} for a in out["author"]]
    return out


def crossref_work(client: Client, doi: str) -> dict | None:
    url = f"{CROSSREF}/{urllib.parse.quote(doi, safe='/')}"
    body = client.get(url, lambda b: {"message": _trim_work(b.get("message", {}))})
    return body["message"] if body else None


def crossref_search(client: Client, raw: str, rows: int = 3) -> list[dict]:
    url = f"{CROSSREF}?" + urllib.parse.urlencode({"query.bibliographic": raw[:500], "rows": rows})
    body = client.get(url, lambda b: {"message": {"items": [
        _trim_work(w) for w in b.get("message", {}).get("items", [])]}})
    return body["message"]["items"] if body else []


def _ncbi(params: dict) -> str:
    params = {**params, "retmode": "json", "tool": "paper_rag_copilot"}
    if os.environ.get("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    return urllib.parse.urlencode(params)


def pubmed_search(client: Client, term: str) -> list[str]:
    url = f"{EUTILS}/esearch.fcgi?" + _ncbi({"db": "pubmed", "term": term})
    found = client.get(url, lambda b: {"idlist": b.get("esearchresult", b).get("idlist", [])})
    return (found or {}).get("idlist", [])


def pubmed_record(client: Client, pmid: str) -> dict | None:
    """ESummary mapped to the Crossref fields used here, plus pmid and retracted."""
    url = f"{EUTILS}/esummary.fcgi?" + _ncbi({"db": "pubmed", "id": pmid})
    keep = ("title", "authors", "pubdate", "volume", "pages", "articleids", "pubtype")
    s = client.get(url, lambda b: {k: v for k, v in b.get("result", {}).get(pmid, b).items()
                                   if k in keep})
    if not s or not s.get("title"):
        return None
    year = s.get("pubdate", "")[:4]
    doi = next((a["value"] for a in s.get("articleids", []) if a.get("idtype") == "doi"), "")
    return {
        "title": [s["title"]], "volume": s.get("volume", ""),
        # "Van Orden GC" -> family "Van Orden"; group names have no initials.
        "author": [{"family": a["name"].rsplit(" ", 1)[0]} for a in s.get("authors", [])
                   if a.get("authtype", "Author") == "Author" and " " in a.get("name", "")],
        "issued": {"date-parts": [[int(year)]]} if year.isdigit() else {},
        "page": s.get("pages", ""), "DOI": doi, "type": "pubmed",
        "pmid": pmid, "retracted": "Retracted Publication" in s.get("pubtype", []),
    }


def pubmed_by_doi(client: Client, doi: str) -> dict | None:
    ids = pubmed_search(client, f"{doi}[doi]")
    return pubmed_record(client, ids[0]) if len(ids) == 1 else None


def pubmed_by_citation(client: Client, listed: dict) -> list[dict]:
    """First author + volume + first page, without the year so that a wrong
    year is still found (and then reported)."""
    if not (listed["volume"] and listed["first_page"]):
        return []
    term = (f'{listed["surname"]}[1au] AND {listed["volume"]}[vi] '
            f'AND {listed["first_page"]}[pg]')
    return [r for r in (pubmed_record(client, i) for i in pubmed_search(client, term)[:3]) if r]


# --- matching and comparison --------------------------------------------------

def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", _fold(s)) if len(w) >= 3 and w not in _STOP]


def title_match(rec: dict, ref) -> str:
    """"full" if the record's title words are in the reference, "main" if only
    the part before the subtitle is, else ""."""
    title = " ".join(rec.get("title") or [])
    raw_words = set(_words(ref.raw.replace("ﬁ", "fi").replace("ﬂ", "fl")))
    for kind, part in (("full", title), ("main", re.split(r"[:?]\s", title)[0])):
        words = _words(part)
        if len(words) >= (1 if kind == "full" else 3) and                 sum(w in raw_words for w in words) / len(words) >= (0.8 if kind == "full" else 1):
            return kind
    return ""


def same_work(rec: dict, ref, listed: dict) -> bool:
    """The record's title (at least its main part) is in the reference and its
    first author is the listed one. A book or manual (no volume/pages listed)
    must also match the year: Crossref holds reviews and later editions under
    the same title."""
    if not title_match(rec, ref):
        return False
    if listed["non_article"] and listed["year"] and _years(rec) and listed["year"] not in _years(rec):
        return False
    fams = [_fold(a.get("family", "")) for a in rec.get("author", []) if a.get("family")]
    if not fams:
        return True
    want = _fold(listed["surname"])
    return any(f == want or f in want or want in f or _edit(f, want) <= 2 for f in fams[:1])


def _years(rec: dict) -> list[str]:
    out = []
    for k in ("published-print", "issued", "published-online"):
        parts = (rec.get(k) or {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            out.append(str(parts[0][0]))
    return out


def _page_key(p: str) -> str:
    return re.sub(r"[^a-z0-9]", "", p.lower())


def compare(rec: dict, listed: dict, via_listed_doi: bool = False) -> list[dict]:
    diffs = []
    years = _years(rec)
    if listed["year"] and years and listed["year"] not in years:
        diffs.append({"field": "year", "listed": listed["year"], "record": years[0]})
    vol = re.sub(r"^\D+", "", rec.get("volume", ""))     # "Volume 14" -> "14"
    if listed["volume"] and vol and listed["volume"] != vol:
        diffs.append({"field": "volume", "listed": listed["volume"], "record": vol})
    art = rec.get("article-number", "")   # Frontiers "617814", not OUP "nsz037"
    page = (rec.get("page") or (art if re.fullmatch(r"e?\d+", art) else "")).split("-")[0]
    if listed["first_page"] and page and _page_key(listed["first_page"]) != _page_key(page):
        diffs.append({"field": "first_page", "listed": listed["first_page"], "record": page})
    doi = rec.get("DOI", "").lower()
    if listed["doi"] and doi and listed["doi"] != doi and not via_listed_doi:   # else an alias
        diffs.append({"field": "doi", "listed": listed["doi"], "record": doi})
    return diffs


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", _fold(name))     # "Andrews‐Hanna" (U+2010) = "Andrews-Hanna"


def compare_authors(rec: dict, listed: dict) -> list[dict]:
    """Author count (unless "et al.") and listed surnames absent from the
    record. A record whose authors are a subset of the listed ones is taken
    as truncated (Crossref sometimes has the first author only)."""
    fams = [a["family"] for a in rec.get("author", []) if a.get("family")]
    rec_k, lst_k = [_key(f) for f in fams], [_key(n) for n in listed["surnames"]]
    same = lambda a, b: a in b or b in a
    if not fams or (len(fams) < len(lst_k) and all(any(same(f, n) for n in lst_k) for f in rec_k)):
        return []
    diffs = []
    if not listed["et_al"] and lst_k and len(lst_k) != len(fams):
        diffs.append({"field": "authors", "listed": len(lst_k), "record": len(fams)})
    absent = [n for n, k in zip(listed["surnames"], lst_k) if not any(same(f, k) for f in rec_k)]
    if absent:
        diffs.append({"field": "author_names", "listed": ", ".join(absent),
                      "record": ", ".join(f for f, k in zip(fams, rec_k)
                                          if not any(same(k, n) for n in lst_k))})
    return diffs


def lookup(client: Client, ref) -> dict:
    listed = listed_fields(ref)
    out = {"reference": ref.raw, "status": "not_found", "diffs": [], "doi": "", "pmid": "",
           "source": "", "type": "", "record_title": "", "hint": ""}
    rec = None
    if listed["doi"]:
        rec = crossref_work(client, listed["doi"])
        if rec and same_work(rec, ref, listed):
            out["source"] = "crossref_doi"
        else:
            out["hint"] = "listed DOI points to another work" if rec else "listed DOI not registered"
            rec = None
    if rec is None:
        rec = next((r for r in crossref_search(client, ref.raw) if same_work(r, ref, listed)), None)
        if rec:
            out["source"] = "crossref_search"
    pm = None
    if rec is None:     # not in Crossref: some older biomedical papers have no DOI
        pm = next((r for r in pubmed_by_citation(client, listed) if same_work(r, ref, listed)), None)
        rec, out["source"] = pm, "pubmed_citation" if pm else ""
    if rec is None:
        if listed["non_article"]:
            out["hint"] = "no volume/pages listed (book, chapter, manual, software?): check by hand"
        elif out["hint"] == "listed DOI not registered":
            out["hint"] = "DOI not registered and no matching record: check that this work exists"
        return out
    out["doi"] = rec.get("DOI", "").lower()
    out["type"] = rec.get("type", "")
    out["record_title"] = " ".join(rec.get("title") or [])
    out["diffs"] = compare(rec, listed, via_listed_doi=out["source"] == "crossref_doi")
    if title_match(rec, ref) == "main":
        out["diffs"].append({"field": "title", "listed": "main title only",
                             "record": " ".join(rec.get("title") or [])})
    if pm is None and out["doi"]:
        pm = pubmed_by_doi(client, out["doi"])
    if pm is None:      # older PubMed records often carry no DOI
        pm = next((r for r in pubmed_by_citation(client, listed) if same_work(r, ref, listed)), None)
    if pm:
        out["pmid"] = pm["pmid"]
    # Authors from the fuller of the two records.
    best = max([r for r in (rec, pm) if r], key=lambda r: len(r.get("author", [])))
    out["diffs"] += compare_authors(best, listed)
    retracted = (pm or {}).get("retracted") or any(
        u.get("type") in _RETRACTION_TYPES for u in rec.get("updated-by", []))
    out["status"] = "retracted" if retracted else "mismatch" if out["diffs"] else "verified"
    return out


def lookup_all(refs: list, client: Client | None = None) -> dict:
    client = client or Client()
    results = []
    for r in refs:
        try:
            results.append(lookup(client, r))
        except (urllib.error.URLError, OSError, ValueError) as e:
            results.append({"reference": r.raw, "status": "error", "hint": str(e)[:200],
                            "diffs": [], "doi": "", "pmid": "", "source": "", "type": ""})
    status = Counter(x["status"] for x in results)
    fields = Counter(d["field"] for x in results for d in x["diffs"])
    return {"n_references": len(refs), "status": dict(status), "mismatch_fields": dict(fields),
            "with_pmid": sum(1 for x in results if x["pmid"]), "network_calls": client.calls,
            "results": results}
