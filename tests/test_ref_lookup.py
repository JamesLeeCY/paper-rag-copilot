"""Reference audit, part 2 (Crossref + PubMed lookups), offline: responses
recorded from Crossref and PubMed for references of the open-access Pratviel
et al. (2024) paper, plus two altered copies of one of them. The retraction
case uses an invented record."""
import json
from pathlib import Path

import pytest

from src.ingest import Reference
from eval.ref_lookup import Client, compare, compare_authors, listed_fields, lookup, same_work

FIX = json.loads((Path(__file__).parent / "fixtures" / "ref_lookup_pratviel.json")
                 .read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_contact_params(monkeypatch):     # they would change the recorded URLs
    monkeypatch.delenv("NCBI_EMAIL", raising=False)


def ref(raw):
    return Reference(ref_id="x", surname="", year="", display="", raw=raw)


def run(key, responses=None):
    responses = responses or FIX["responses"]
    client = Client(cache_dir=None, fetch=lambda url: responses.get(url))
    return lookup(client, ref(FIX["references"][key]))


def test_listed_fields_rejoin_wrapped_dois_and_count_authors():
    f = listed_fields(ref(FIX["references"]["24"]))       # "doi:10.5298/1081-5937-41. 1.02"
    assert f["doi"] == "10.5298/1081-5937-41.1.02" and (f["volume"], f["first_page"]) == ("41", "26")
    assert listed_fields(ref(FIX["references"]["57"]))["doi"] == "10.1016/j.chb.2022.107607"
    assert listed_fields(ref(FIX["references"]["50"]))["n_authors"] == 4   # "schraefel, m. c."
    assert listed_fields(ref(FIX["references"]["45"]))["non_article"]       # a 1956 book


def test_verified_by_doi_with_pmid_and_without_pubmed():
    r = run("14")                                       # Dickerson & Kemeny 2004
    assert (r["status"], r["source"], r["pmid"]) == ("verified", "crossref_doi", "15122924")
    r = run("21")                                       # Kaplan 1995: not indexed in PubMed
    assert (r["status"], r["pmid"]) == ("verified", "")


def test_pubmed_fallback_when_crossref_has_nothing():
    r = run("42")                                       # Sackeim 2001a, no DOI
    assert (r["status"], r["source"], r["pmid"]) == ("verified", "pubmed_citation", "11234909")


def test_record_formatting_is_not_a_mismatch():
    assert run("58")["status"] == "verified"            # Crossref volume "Volume 14"
    assert run("50")["status"] == "verified"            # four authors, one lowercase


def test_not_found_and_book_hint():
    assert run("4")["status"] == "not_found"
    r = run("45")
    assert r["status"] == "not_found" and "by hand" in r["hint"]


def test_wrong_year_and_wrong_doi_are_reported():
    r = run("dickerson_year_2005")
    assert r["status"] == "mismatch"
    assert r["diffs"] == [{"field": "year", "listed": "2005", "record": "2004"}]
    r = run("dickerson_kaplan_doi")                     # DOI of another listed work
    assert r["status"] == "mismatch" and r["hint"] == "listed DOI points to another work"
    assert [d["field"] for d in r["diffs"]] == ["doi"]
    assert r["diffs"][0]["record"] == "10.1037/0033-2909.130.3.355"


def test_retracted_via_pubmed():
    raw = ("Quillwort, A., and Rushby, B. (2015). Invented findings on fern growth. "
           "J. Imag. Bot. 12, 34–56. doi:10.9999/jib.2015.034")
    work = {"message": {"DOI": "10.9999/jib.2015.034", "title": ["Invented findings on fern growth"],
                        "author": [{"family": "Quillwort"}, {"family": "Rushby"}],
                        "issued": {"date-parts": [[2015]]}, "volume": "12", "page": "34-56"}}
    responses = {
        "https://api.crossref.org/works/10.9999/jib.2015.034": work,
        **{u: {"idlist": ["99999999"]} for u in [
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
            "10.9999%2Fjib.2015.034%5Bdoi%5D&retmode=json&tool=paper_rag_copilot"]},
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=99999999"
        "&retmode=json&tool=paper_rag_copilot": {
            "title": "Invented findings on fern growth.", "pubdate": "2015",
            "pubtype": ["Journal Article", "Retracted Publication"]},
    }
    client = Client(cache_dir=None, fetch=lambda url: responses.get(url))
    r = lookup(client, ref(raw))
    assert (r["status"], r["pmid"]) == ("retracted", "99999999")


def _rec(title, families, year, **kw):
    return {"title": [title], "author": [{"family": f} for f in families],
            "issued": {"date-parts": [[year]]}, **kw}


def test_author_names_and_truncated_records():
    listed = listed_fields(ref("Alder, A., Birch-Holm, B., & Cedar, C. (2010). Leaf litter. "
                               "J. Inv. 3, 10–20."))
    assert compare_authors(_rec("Leaf litter", ["Alder"], 2010), listed) == []        # first author only
    assert compare_authors(_rec("Leaf litter", ["Alder", "Birch‐Holm", "Cedar"], 2010), listed) == []
    diffs = compare_authors(_rec("Leaf litter", ["Alder", "Dogwood", "Cedar", "Elm"], 2010), listed)
    assert diffs == [{"field": "authors", "listed": 3, "record": 4},
                     {"field": "author_names", "listed": "Birch-Holm", "record": "Dogwood, Elm"}]


def test_book_needs_the_year_and_a_subtitle_may_be_missing():
    book = ref("Alder, A. (1998). A compendium of leaf tests (2nd ed.). Invented Press.")
    review = _rec("A compendium of leaf tests", ["Alder"], 2000, type="journal-article")
    assert not same_work(review, book, listed_fields(book))
    art = ref("Alder, A. (2006). Leaf plasticity across the life span. J. Inv. 30, 775–790.")
    rec = _rec("Leaf plasticity across the life span: Evidence and modeling", ["Alder"], 2006)
    assert same_work(rec, art, listed_fields(art))


def test_doi_alias_and_publisher_article_ids():
    listed = listed_fields(ref("Alder, A. (2003). Leaf statistics. J. Inv. 14, 391–412. "
                               "https://doi.org/10.1088/0954-898X/14/3/302"))
    rec = _rec("Leaf statistics", ["Alder"], 2003, volume="14", page="391-412",
               DOI="10.1088/0954-898x_14_3_302")
    assert compare(rec, listed, via_listed_doi=True) == []
    assert [d["field"] for d in compare(rec, listed)] == ["doi"]
    rec = _rec("Leaf statistics", ["Alder"], 2003, volume="14", **{"article-number": "nsz037"})
    assert compare(rec, listed, via_listed_doi=True) == []
