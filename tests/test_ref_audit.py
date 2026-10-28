"""Reference audit, internal consistency (no network, no corpus). Author names
are invented."""
from src.ingest import Reference
from eval.ref_audit import audit, citations_in


def ref(raw, surname="", year=""):
    return Reference(ref_id=raw[:8], surname=surname or raw.split(",")[0], year=year,
                     display="", raw=raw)


def test_parenthetical_narrative_and_multiple_years():
    text = ("Nature helps (Alder, 1995; Birch et al., 2008, 2012) and "
            "Cedar et al. (2015) agree; Dogwood's (1984) view differs.")
    assert set(citations_in(text)) == {("Alder", "1995"), ("Birch", "2008"), ("Birch", "2012"),
                                       ("Cedar", "2015"), ("Dogwood", "1984")}


def test_acronyms_particles_and_multiword_surnames():
    text = ("(QSX, English version, Elmwood et al., 2009) (de Fern et al., 2006; "
            "De Grove Hazel et al., 2022; Ivy Juniper et al., 2018)")
    assert set(citations_in(text)) == {("Elmwood", "2009"), ("de Fern", "2006"),
                                       ("De Grove Hazel", "2022"), ("Ivy Juniper", "2018")}


def test_a_citation_group_across_a_paragraph_break():
    body = ["Effects were shown (Larch et al., 2018;", "Maple et al., 2019)."]
    res = audit(body, [ref("Larch, A. (2018). T."), ref("Maple, B. (2019). T.")])
    assert res["unresolved"] == [] and res["uncited_references"] == []


def test_unresolved_kinds_uncited_and_ambiguous():
    refs = [ref("Oakley, C. (2020). One."), ref("Pinewood, E. O. (1984). Two."),
            ref("Quince, C., & Rowan, G. (2021). A."), ref("Quince, C., & Rowan, G. (2021). B."),
            ref("Sorrel, J. (2013). Three.")]
    body = ["(Oakley et al., 2018; Pinewod, 1984; Tamarack et al., 2025; Quince and Rowan, 2021)"]
    res = audit(body, refs)
    kinds = {u["surname"]: u["kind"] for u in res["unresolved"]}
    assert kinds == {"Oakley": "year_mismatch", "Pinewod": "misspelling", "Tamarack": "missing"}
    # Cited only with a wrong year or spelling, so those two count as uncited too.
    assert sorted(res["uncited_references"]) == ["Oakley, C. (2020). One.",
                                                 "Pinewood, E. O. (1984). Two.",
                                                 "Sorrel, J. (2013). Three."]
    assert res["ambiguous_same_surname_year"] == ["quince 2021"]


def test_an_entry_merged_into_another_is_a_split_error():
    merged = ref("Upas, M. (2003). Running. Vine, D. E. (2010). A systematic review.")
    res = audit(["(Vine et al., 2010; Upas, 2003)"], [merged])
    assert [u["kind"] for u in res["unresolved"]] == ["reference_split_error"]


def test_numbered_citations_expand_ranges_and_lists():
    from eval.ref_audit import numbered_citations_in
    assert numbered_citations_in("shown [1-3], again [1,5] and [6, 8–9]; not [DOCX File]") == \
        [1, 2, 3, 1, 5, 6, 8, 9]


def test_a_numbered_list_is_audited_by_number():
    refs = [ref(f"Hazel{i} JE. Title {i}. J Nat. 2020;1:{i}.") for i in range(1, 6)]
    res = audit(["Parks help [1,2] and [4]; see [3] and [7].", "Alder (1995) is cited once [1]."], refs)
    assert res["style"] == "numbered"
    assert [u["number"] for u in res["unresolved"]] == [7]
    assert res["uncited_references"] == ["[5] Hazel5 JE. Title 5. J Nat. 2020;1:5."]
    assert res["first_cited_out_of_order"] == [3]


def test_author_year_text_with_a_stray_bracket_number_stays_author_year():
    res = audit(["As shown (Alder, 1995; Birch, 2001) and in step [2]."],
                [ref("Alder, A. (1995). T."), ref("Birch, B. (2001). T.")])
    assert res["style"] == "author-year" and res["unresolved"] == []
