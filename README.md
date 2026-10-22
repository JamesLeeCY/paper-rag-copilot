# 論文寫作 RAG Copilot — Dissertation RAG Copilot

A **strictly citation-grounded** retrieval-augmented assistant over a PhD
dissertation (or a small set of journal-article PDFs), with a **quantified
evaluation harness** that measures whether its hallucination rate is low enough
to trust — and a second harness that measures whether the *judges* doing that
measuring can themselves be trusted.

Main corpus: *Lee (2025), "The Effects of Nature-Based Interventions on
Psychological, Cognitive, and Neural Functioning Across the Lifespan"* —
Introduction through Discussion, plus its 84-entry reference list. Pilot corpus:
Watkins-Martin et al. (2026), *Journal of Environmental Psychology* 115:103188
(CC-BY). Every answer traces back to the exact section, paragraph span and (for
PDFs) page it came from.

> Built to demonstrate three things end-to-end: **prompt & context engineering**
> (citation-forcing prompts, chunking ablation), **RAG system design** (hybrid
> search + rerank + a layered verification cascade), and **evaluation
> methodology** (golden sets, trap tiers, and a known-answer benchmark for the
> LLM judges — not vibes).

**Contents:**
[Results](#results) ·
[Architecture](#architecture) ·
[Prompt & context engineering](#prompt--context-engineering) ·
[Setup](#setup) ·
[Usage](#usage) ·
[Evaluation methodology](#evaluation-methodology) ·
[Running long local evaluations](#running-long-local-evaluations) ·
[Known limitations](#known-limitations) ·
[Roadmap](#roadmap) ·
[Layout](#layout)

---

## Results

All numbers below are aggregate; the corpora, golden sets and full reports stay
local (see [Data & privacy](#data--privacy)).

**Current default** (as of 2026-10-23): embedder **`bge-m3`** (multilingual;
tables before the embedder comparison below used `bge-small-en-v1.5`),
generator `qwen2.5:7b-instruct` with generator prompt `v2.1` (results tables
before the v2.1 comparison below were produced with `v1`), judge **`phi4`** alone, judge prompt **`v3`** with each
passage's section heading, section-aware chunking, quotes may span chunks, and
the deterministic plan-as-result rule (`flag`) with its structural, wording,
inline and hypothesis-match routes, and the echo rule (`flag`, since
2026-10-23: a partially supported claim restating a yes/no question goes to
review); local models run on Ollama, CPU only. The judge was
chosen on the judge validation below (2026-10-07). "Previous" below means
judges `qwen2.5:7b-instruct` + `gemma3:12b` (unanimous) with prompt `v2`.

### At a glance

| | Dissertation, previous | **Dissertation, current** | Pilot, previous | **Pilot, current** | Target |
|---|---|---|---|---|---|
| Retrieval Hit@5 (section-aware) | 100% (MRR 0.952) | 100% (MRR 0.952) | 95% | 95% | ≥ 90% |
| Strict citation precision | 96% (26/27) | 84% (26/31) → **97% (30/31)**¹ | 90% (18/20) | 90% (18/20) | ≥ 95% |
| Lenient citation precision | 96% | 97% → **100%**¹ | 90% | **95%** | — |
| Claim / answer hallucination rate | 0% / 0% | 3% / 4% → **0% / 0%**¹ | 10% / 11% | **5% / 5%** | low |
| Must-refuse traps refused | 95% | 95% | 100% | 100% | ≥ 90% |
| False-premise traps safe | 83% | **100%** → 67%² | 67% | **83%** → 67%² | — |
| Hypothesis-bait traps safe (12 local items) | — | **67%** (8/12)³ | — | — | — |
| Over-refusal on answerable questions | 0% | 0% | 5% | 5% | low |
| Parse failures | 1 | **0** | 0 | 0 | — |

² Rerun with all fixes, 2026-10-09 (see "After all fixes" below): two of the
misses were judge timeouts since fixed, one a translated quote, one a claim
the judge rightly rejected. ³ Questions built on hypotheses or planned
analyses whose results the thesis does not report; the generator restated a
hypothesis on 4 of them and verification stopped all 4 (2026-10-16).

**Against an ungrounded baseline** (same generator, context and judge;
[details](#grounded-vs-ungrounded-baseline-dissertation)): the grounded
design is far better at refusing what the source does not say (25 traps:
88% vs 24% correct, p < 0.001) and never worse on a hypothesis-bait item
(8/12 vs 5/12 safe, not significant on 12 items). On answerable questions
the two are close once the judge sees every retrieved passage; there the
grounded design's gain is that every claim carries a verbatim quote that
can be checked.

**No false claim was shown to the user as supported in any run**: every
"hallucination" counted here is a claim the verifier flagged. The current
default's lower strict precision on the dissertation was audited claim by claim
(the same pattern appeared in two independent runs):

| Non-supported claim | Who flagged it | Verdict on the verdict |
|---|---|---|
| 2 claims | phi4 (partial) | **Correct** — the generator mischaracterised the passage; the previous judges would likely have accepted them |
| 1 claim | phi4 (partial) | Reasonable on what it saw, but true in the document: the chunk lacks the section context that names the analysis |
| 1 claim | phi4 (partial) | Too strict — a narrower but consistent term |
| 1 claim | quote check (unsupported) | False rejection — the quote ran across a chunk boundary (a paragraph split between two chunks) while the claim cited only the first, so the match against that chunk failed |

Two fixes follow from this, both now in the code (a quote may span retrieved
chunks if every sentence is found; the judge sees each passage's section
heading). ¹ A confirmation rerun of the 26 answerable questions with both fixes
(the generator produced the same 31 claims) reached **97% strict / 100%
lenient**: the three false rejections became supported, and the more serious
generator error is still flagged. The milder one (same imprecise wording, core
facts right) now passes. A check on the judge validation set (below) found that
the heading does not make phi4 more lenient, so this is most likely judge
variance on a borderline claim.

**After all fixes** (traps for both corpora and the full pilot paper, rerun
2026-10-09): far- and near-absent traps unchanged (dissertation 100% / 89%,
pilot 100% / 100%); pilot answerable 89% strict / 95% lenient (was 90% / 95%);
false-premise traps 67% in both corpora (was 100% / 83%). Of those four
misses, two were judge timeouts on a long prompt that the lexical fallback then
rejected (both since fixed: the timeout covers prompt reading, and a claim no
judge could vote on goes to review), one was a translated quote, and one a
generator claim the judge correctly rejected. Still no false claim shown as
supported. On
the pilot paper all four generator errors under the previous default were
caught as well (three rejected, one disputed).

### How certain are these numbers?

Every rate here rests on a small sample, so the reports now print a 95%
Wilson confidence interval with the counts (`eval/stats.py`). For the headline
numbers:

| Rate | Estimate [95% CI] (k/n) |
|---|---|
| phi4 false-accept, dissertation held-out | 5% [2–15] (3/55) |
| phi4 false-reject, dissertation held-out | 7% [2–23] (2/28) |
| phi4 false-accept, pilot held-out | 6% [2–19] (2/35) |
| phi4 + plan rule false-accept, dissertation | 2% [0–10] (1/55) |
| Strict citation precision, dissertation, before → after the audit fixes | 84% [67–93] → 97% [84–99] (26/31 → 30/31) |
| Strict citation precision, dissertation, previous default | 96% [82–99] (26/27) |
| Strict citation precision, pilot, after fixes | 89% [69–97] (17/19) |
| Must-refuse traps refused, dissertation / pilot | 95% [75–99] (18/19) / 100% [74–100] (11/11) |
| Over-refusal, dissertation / pilot | 0% [0–13] (0/26) / 5% [1–24] (1/20) |

Read differences accordingly. The intervals of the judges compared here
overlap, so "best judge" means best measured, not proven better. The 84% → 97%
gain was measured on the same 31 claims, so it is a paired comparison — 4
claims changed, all in the same direction — yet with only 4 discordant claims
even that is not significant on its own (exact two-sided binomial p = 0.125);
it rests on the claim-by-claim audit as much as on the numbers. Claim-level
intervals also treat claims as independent, which understates the uncertainty
when one answer has several claims.

### Grounded vs ungrounded baseline (dissertation)

Does the citation-forcing design matter, or would the same generator answer as
well from the same context without it? `eval --with-llm --baseline` asks the
same questions over the same top-5 passages with an ungrounded prompt (plain
prose), splits the answer into sentences and has the same judge (phi4, prompt
v3) check each sentence against its two closest passages. Sentences that only
abstain ("the text does not report X") or only advise where to look are not
judged. Same 26 answerable questions and 25 traps as the grounded run
(2026-10-12/13; Wilson 95% intervals; paired exact McNemar tests). The
baseline rows were re-scored after the 2026-10-22 change to sentence
judging (plan rule per passage, closest non-hypothesis passage added; see
"Independent check of the hypothesis routes"): the 43 sentences near a
Hypotheses section were judged again from the saved answers. One answerable
sentence moved from partially supported to supported (strict 85 → 86 of 94;
answers with a non-supported claim 7 → 6 of 26, p 0.031 → 0.063); every
trap and bait outcome stayed the same item by item:

| | Grounded (default) | Ungrounded baseline | Paired test |
|---|---|---|---|
| Strict citation precision (claims / sentences) | **97%** [84–99] (30/31) | 91% [84–96] (86/94) | — |
| Claim hallucination rate | 0% [0–11] | 4% [2–10] (4/94) | — |
| Answers with any non-supported claim | **4%** (1/26) | 23% (6/26) | p = 0.063 |
| Answers with an unsupported claim | 0% (0/26) | 15% (4/26) | p = 0.125 |
| Must-refuse traps refused | **95%** [75–99] (18/19) | 16% [6–38] (3/19) | p < 0.001 |
| Must-refuse traps safe (no rejected/disputed claim) | 95% [75–99] | 74% [51–88] (14/19) | p = 0.125 |
| All 25 traps correct | **88%** [70–96] | 24% [12–43] (6/25) | p < 0.001 |
| False-premise traps correct | 67% (4/6) | 50% (3/6) | — |
| Hypothesis-bait traps safe (12 local items, Ollama 0.40.5) | **67%** [39–86] (8/12) | 42% [19–68] (5/12) | p = 0.25 |
| Hypothesis-bait traps strict safe (same) | **67%** [39–86] (8/12) | 33% [14–61] (4/12) | p = 0.125 |

What the numbers mean, from a manual audit of every baseline trap answer:

- The baseline rarely just plays along. On must-refuse traps it usually first
  says the source does not cover the topic (68% open that way), then keeps
  going with related, mostly true content. Under the refusal definition both
  arms share (abstain and make no claim) that is not a refusal, so the
  refusal gap is large; the safety gap is smaller and, on 19 traps, not
  significant.
- The errors that do get through are the ones the grounded design targets:
  about a quarter of the shared traps were answered by accepting the premise
  or inventing content (one answer invented a participant group; one reported
  a planned analysis as a result); on hypothesis bait it stated a
  hypothesis, a plan or a guess as a finding on at least 7 of 12 items.
- Hypothesis bait (12 local items: questions built on hypotheses or planned
  analyses whose results the thesis does not report) is where the grounded
  generator is weakest, and where verification earns its keep. It refused 8
  and restated a hypothesis as a finding 4 times; all 4 were stopped before
  being shown as supported — once by the judge, twice by the structural rule
  route (the quote came from a Hypotheses section) and once by the
  hypothesis-match route (it cited a *results* passage with a verbatim quote,
  which neither older route sees; Known limitations, item 1). The baseline
  often denied the premise and then added content; it was safe on 5 of 12
  (4 of 12 strict). The grounded arm was never worse on an item (3–4
  discordant items, all one way), but 12 items do not make that significant.
  The grounded results were identical item by item on Ollama 0.35.1 and
  0.40.5. An earlier 6-item version had 2 items whose hypothesis the thesis
  does report as a result; they were replaced.
- On answerable questions the grounded arm is more precise, but the
  baseline's hallucination rate is overstated: at least 2 of its 4
  unsupported sentences have their evidence in the context but not in the
  two passages the judge saw.

Caveats: not paired at the claim level (different outputs from the same
generator), the baseline is judged against 2 of the 5 passages, and abstention
detection is a pattern list (audited on these answers, see HANDOFF §10).

**Sensitivity: judge on all 5 passages** (`python -m eval.rejudge`,
2026-10-15). Every non-supported answerable claim and every unsupported or
disputed trap claim of *both* arms (36 claims) was re-judged against the
context passages it had not been judged on, keeping the better label:

| | Grounded | Baseline | Paired test |
|---|---|---|---|
| Strict citation precision | 97% → 100% (31/31) | 90% → **96%** [90–98] (90/94) | — |
| Claim hallucination rate | 0% → 0% | 4% → **2%** [1–7] (2/94) | — |
| Answers with any non-supported claim | 1/26 → 0/26 | 7/26 → 4/26 | p = 0.031 → 0.125 |
| Answers with an unsupported claim | 0/26 → 0/26 | 4/26 → 2/26 | p = 0.125 → 0.5 |
| Must-refuse traps safe | 18/19 → 18/19 | 14/19 → 14/19 | p = 0.125 |
| All 25 traps correct | 88% → 96% | 24% → 24% | p < 0.001 |
| Hypothesis-bait traps safe (first bait set) | 5/6 → 6/6 | 0/6 → 1/6 | p = 0.063 |
| Premise traps (false-premise + first bait set) **strict safe** | 8/12 → 9/12 | 1/12 → 1/12 | p = 0.016 → 0.008 |

The "before" column is the 2026-10-15 scoring; the 2026-10-22 re-score of
the main table moved one baseline sentence (R19), which this all-passage
re-judge had already upgraded, so the lenient bound is unchanged.
14 labels changed. Six baseline flips were justified (the evidence was in an
unseen passage), so the 2-passage view does penalise the baseline. But the
wider view also made the judge more lenient on traps: a claim stating a
hypothesis as confirmed (both arms) and one accepting a false premise
(grounded) moved from unsupported to partially supported, which counts as
safe. Read the main table as the strict bound and this one as the lenient
bound. Under both, the grounded arm is far better at refusing what the source
does not say; on answerable questions the two arms are close once the judge
sees the evidence, and the grounded arm's advantage there is that every claim
carries a verbatim, checkable quote.

A stricter trap rule closes the leniency gap: *strict safe* (also no
partially supported claim; reported per trap type by `eval` and by
`eval.rejudge`) fails the claims the wider view let through as partially
supported (it cannot catch the one it upgraded to supported), and on the 12
premise traps the grounded arm leads under both labelings (last row). It also fails
baseline answers that correct the premise but add one overreaching sentence,
so it favours short answers; report it next to the plain safe rate, not
instead of it.

### Out-of-corpus check: two open-access papers (2026-10-17)

The same pipeline and both arms on two CC BY papers the system was never
tuned on: a **study protocol** (Zhang et al. 2026, *JMIR Research
Protocols* 15:e82970 — every hypothesis and outcome is planned, no data
exist) and a **completed study whose hypotheses all have reported results**
(Rockstroh et al. 2020, *Frontiers in Virtual Reality* 1:598282). Local
question sets: protocol 4 answerable design questions + 6 bait questions
asking for the planned results; completed study 6 answerable questions, 5
of them on the results of its hypotheses (true claims that restate a
hypothesis).

| | Grounded | Baseline |
|---|---|---|
| Protocol bait: safe | **5/6** | 3/6 |
| Protocol bait: invented a result | **0/6** | 3/6 (all caught: judge 3, plan rule 1 sentence) |
| Protocol answerable: claims supported (strict) | **7/7** | 10/14 (lenient 14/14) |
| Completed study: claims supported (strict) | **6/6** | 13/19 |
| Completed study: true results disputed by a rule | 0 | 0 |

The grounded generator never reported a result the protocol does not
contain; its one "unsafe" item is a denial written inside a claim with a
translated quote, which the quote check rejected. The baseline invented
outcomes on half the bait (e.g. an EEG effect, a symptom improvement) and
verification stopped every one. On the completed study the plan-as-result
rule did not touch any true result, including claims that restate a
hypothesis; two of the baseline's three rejected sentences are true (their
evidence was not in the two passages the baseline is judged against).
Neither paper has a Hypotheses heading, so the structural and
hypothesis-match routes could not fire here: this checks the generator,
the judge and the wording route out of corpus, not those two routes.
Pooled with the thesis's 12 bait items, the grounded arm is safe on 13/18
vs 8/18, all 5 discordant items in its favour (exact McNemar p = 0.063).
These grounded numbers are with generator prompt v1.

**Rerun with generator prompt v2.1 (2026-10-19, grounded arm only):**

| | v1 | **v2.1** |
|---|---|---|
| Protocol bait: safe / refused / invented a result | 5/6 / 3 / 0 | **6/6 / 5 / 0** |
| Misplaced refusal markers / translated quotes | 2 / 1 | **0 / 0** |
| Protocol answerable: claims supported | 7/7 | 4/4 (shorter answers) |
| Completed study: strict / lenient | **6/6** / 6/6 | 4/6 / 6/6 |
| True results disputed by a rule | 0 | 0 |

v2.1 fixed v1's out-of-corpus format error (a denial written inside a claim
with a translated quote) and refused more of the bait. Its two partially
supported claims on the completed study are an added detail the source does
not give and an answer that copies a (confirmed) hypothesis sentence and
cites the hypothesis passage rather than the result; neither is
unsupported. Two format habits seen here — a dropped opening `<quote>` tag
(now handled by the parser) and a `<refusal>` appended to a complete
answer — were fixed: after rule 7 was tightened, a 20-question calibration
on all three corpora had no `<refusal>` after a complete answer and still
refused every trap and bait item. One habit remains: when a hypothesis
passage is retrieved for a question about a result, v2.1 sometimes answers
with the hypothesis — correctly labelled as a hypothesis, but citing a
passage that does not contain it (one unsupported claim in that
calibration). A sentence telling the generator to answer result questions
from result passages was tried and withdrawn: it did not change that answer,
fixed a second one and made a third worse (caught by the hypothesis-match
route) — with a 7B generator, single-sentence prompt edits move individual
answers in both directions. Showing results passages first
(`CONTEXT_ORDER=results_first`) was tried next and not adopted: 13 of 21
calibration answers were identical, and on the completed study the generator
still copied the hypothesis sentences — it now cited a results passage id,
the quote check re-attributed the quote to the introduction, and the judge
accepted the (confirmed) hypothesis. Part of the habit is a question-design
artefact: the control questions paraphrase the hypotheses, so retrieval and
generator both gravitate to the hypothesis sentence. And because this
paper's hypotheses sit in the introduction rather than under a Hypotheses
heading, neither the structural nor the hypothesis-match route could see a
hypothesis restated as a result there. That gap now has an **inline route**:
a sentence the cited passage labels as a hypothesis ("Hypothesis 1 (H1):",
"H3b:"), quoted or copied by an unhedged claim, is flagged like a
Hypotheses-section citation, and inline hypotheses also join the
hypothesis-match route's pool. Re-applied offline to every saved output of
the four corpora, it fired on exactly the two claims above (completed-study
R01 and R04) and on nothing else. Pooled bait with
v2.1 (thesis full run + protocol): 17/18 safe.

### Independent check of the hypothesis routes (2026-10-22)

Two follow-ups to the out-of-corpus check, current defaults (bge-m3,
generator v2.1, phi4 judge), both arms, `--fresh`.

**Control questions rewritten (Rockstroh et al. 2020).** The old controls
paraphrased H1–H3c; with bge-m3, two of them (R01, R04) retrieved the
hypothesis passages and no results passage at all, so a hypothesis was the
only possible answer. Rewritten to ask for the measured outcome (scale,
statistic, group means) plus one exploratory result: no hypothesis passage
in any top-5, and the grounded arm answered all 7 from results or methods
passages (strict 7/8, the eighth a true detail from a figure caption the
judge did not see; baseline 17/18). The residual "answers a result question
from the hypothesis" was mostly that question-design artefact.

**A paper with a Hypotheses heading (Pratviel, Bouny & Deschodt-Arsac 2024,
*Frontiers in Virtual Reality* 5:1358981, CC BY).** Its "1.5 Hypotheses"
section holds four hypotheses with mixed outcomes: H1A supported, H1B not
supported, H2 supported, H3 not supported for immersion and not tested
directly for guided breathing. Local set: 7 answerable (three worded like
the hypotheses, whose true results the rules must not flag), 3 false
premises built on H1B/H2/H3, 3 bait items (an unreported psych–physio
correlation and two analyses proposed only as future work), 2 near-absent,
1 far-absent. The hypothesis passage was ranked first for two answerable
questions and in the top 5 for two false premises.

| | Grounded | Baseline |
|---|---|---|
| Answers fully supported (7) | **7** | 4 |
| Claims supported, strict | **9/9** | 19/26 |
| Traps correct (9) | **9** | 4 |
| Traps strict safe (9) | **9** | 2 |
| Plan-as-result rule fired on a true claim | 0 | 0 (1 before the fix below) |
| Mean seconds per question | 184 | 706 |

Grounded vs baseline on traps (exact McNemar): strict safe 7–0 discordant,
p = 0.016; correct 5–0, p = 0.063 — nine items, same direction as on the
other corpora.
The grounded generator never stated a hypothesis as a result, so no rule had
to fire: it refused 8 traps and answered the RMSSD false premise with true
results, and on the hypothesis-worded controls it cited the results passage
(or quoted H1A explicitly as "We hypothesized that…", which the hedge check
correctly leaves alone). In the baseline the hypothesis-match route flagged
the false claim "RMSSD rose during immersion alone" (H3 restated as a
finding), and the structural route disputed one **true** sentence (H2 is
supported) — a false flag: the baseline judges each sentence against its two
closest passages, and a true finding worded like its hypothesis is closest
to the hypothesis. Fixed (2026-10-22): `verify_statement` applies the
plan-as-result rule per passage before picking the best one, and a sentence
whose two closest passages include a Hypotheses section is also judged
against the closest passage outside it. Re-scored from the saved answers,
that sentence is supported and the flagged false claim stays disputed.
Three baseline abstentions not caught by the refusal patterns ("not possible
to determine", "unclear from the provided context", "does not explicitly
break down") were judged as claims; the patterns now cover them. The table
shows the baseline after both fixes (before: 3 answers, 18/26, 3 traps).
Two things the metrics do not show: one grounded answer conflated two
statistics (a p-value from the high-anxiety subgroup reported as the overall
VAS result) and the judge accepted it; and two grounded answers were true
but did not answer the question (R01 gave the method and the hypothesis —
the results passage was not retrieved; R04 gave the guided-breathing effect,
not the immersion-alone null). Answer completeness is not measured. The
thesis baseline was re-scored under the fix (43 sentences judged again):
one answerable sentence improved, no trap or bait outcome changed (see the
baseline section above).

### Chinese questions over English sources (2026-10-21)

The sources are English but a user may ask in Chinese. Every retrieval
question of the four corpora (56) was also written in Traditional Chinese
(local golden sets) and run with `python -m eval.retrieval_lang`
(section-aware chunks):

| Hit@5 (Hit@1), pooled over 56 questions | English questions | Chinese questions |
|---|---|---|
| `bge-small-en-v1.5`, hybrid (default) | 95% (77%) | 52% (25%) |
| `bge-m3`, hybrid | 93% (82%) | 73% (34%) |
| **`bge-m3`, dense-only for Chinese queries** | 93% (82%) | **88% (71%)** |

A multilingual embedder closes most of the gap, and BM25 — which cannot
match Chinese words against English text — has to be left out for a Chinese
query: fused in, its noise pushed the right passage down. With
`EMBED_MODEL=BAAI/bge-m3` (index built with that model) a query containing
Chinese characters now skips BM25 automatically (`CJK_QUERY_DENSE_ONLY=auto`);
English questions keep hybrid retrieval and stay as good.

**bge-m3 became the default after a full run** (thesis, 26 answerable
questions + 37 traps, same generator, judge and code for both embedders,
2026-10-21):

| | `bge-small-en-v1.5` | **`bge-m3`** |
|---|---|---|
| Retrieval Hit@1 / Hit@5 (English questions) | 92% / 100% | 88% / 100% |
| Strict citation precision | 26/28 | **27/28** |
| Answers fully supported (26) | 24 | **25** |
| Traps strict safe (37) | 34 | **35** |
| Hypothesis-bait safe (12) | 10 | **11** |
| Mean seconds per question | 127 | **111** |

Every difference is one item (not significant): on English questions the
multilingual embedder is as good, and on Chinese questions it is much
better. The run also exposed a parser bug introduced two days earlier — a
stray `</quote>` after a paraphrased claim turned the claim into its own
"quote", which then failed the quote check — that rejected five true claims
across the two runs; it was fixed and those five claims re-judged from the
saved outputs before the comparison above.

**End to end with Chinese questions** (2026-10-22, `eval --with-llm --lang
zh`, grounded arm, current defaults; every question asked in Traditional
Chinese — the thesis's 12 bait items were already Chinese; compared item by
item with the English runs of 2026-10-21 under the same settings):

| | Thesis en → zh | Pratviel 2024 en → zh | Rockstroh 2020 en → zh |
|---|---|---|---|
| Retrieval Hit@5 (answerable) | 100% → 92% | 100% → 86% | 100% → 100% |
| Answers fully supported | 25 → 23 (of 26) | **7 → 3** (of 7) | 6 → 5 (of 7) |
| Claims supported, strict | 27/28 → 27/29 | **9/9 → 4/8** | 7/8 → 6/8 |
| Traps correct / strict safe | 36, 35 → **36, 35** (of 37) | 9, 9 → **9, 9** | — |
| Answerable questions refused | 0 → 1 | 0 → 0 | 0 → 0 |
| Quotes translated into Chinese | 0 → **0** | 0 → 0 | 0 → 0 |

Refusal and safety do not depend on the question language: every trap and
bait outcome is the same item by item, and every quote stayed verbatim
English. Answer precision does: pooled over the three corpora, 9 answers
were clean in English only and 2 in Chinese only (exact McNemar p ≈ 0.065).
On the thesis the difference is small (two partially supported claims and
one refusal of a question whose passage was ranked first; one answer
improved). The losses concentrate on yes/no questions ("did X reduce Y?",
"did A and B differ?"): the generator tends to turn the Chinese question
into an affirmative claim and attach a quote. Verification caught most of
it — a true claim with an irrelevant quote was rejected, a quote the
generator paraphrased failed the quote check — but not all: a **false**
echo ("the two conditions differed"; they did not) got only partially
supported, and phi4 rejected one true claim with a correct quote. With
Chinese claims phi4 also reasons in Chinese and was less reliable here
than on the English runs. Usable for Chinese questions over English
sources, with lower answer precision on yes/no questions.

**A prompt rule against the echo was tried and not adopted** (generator
prompt `v2.2` = v2.1 + rule 8: on a yes/no question, do not restate the
question as an affirmative claim; state the reported result, which may be
"no difference", and quote that sentence). 34-item calibration (thesis 9
answerable + 9 traps/bait in Chinese, Pratviel 10 in Chinese, Rockstroh 2
in Chinese, Pratviel 4 in English), compared item by item with v2.1:
30 outcomes identical, most answers word for word — including the two
target echoes, so the false "the conditions differed" stayed. Two improved
(a verbatim quote instead of a missing or paraphrased one) and two English
answers got worse: one now restated the confirmed H1A as a finding and was
disputed by the hypothesis-match route, one answered an English question in
Chinese with a quote from the wrong comparison. Net zero, below the
adoption bar (≥ +3 and no regression), so the full run was not done and
v2.1 stays the default; `v2.2` remains selectable for reproduction. As with
the earlier rule-5 attempt, one-sentence prompt edits barely move a 7B
generator where intended and move it elsewhere.

**An echo check on the verification side instead (default since
2026-10-23).** Restating the question is not itself an error — of the 8
saved claims that echo a yes/no question (44 claims answer one, all
corpora, both languages), 5 are true — so the rule only touches what the
judge rated *partially supported*: such a claim, when it restates the yes/no
question (bge-m3 similarity ≥ 0.90 and character overlap ≥ 0.70 once
question words are removed), goes to review as `disputed`. Applied offline
to every saved grounded output it fires once, on the false "the conditions
differed", and flags no true claim; the other two partially supported
claims on yes/no questions score far below the thresholds (0.66 and 0.62).
That is one positive, with thresholds set on the same outputs, so it is not
independently validated; it was turned on because Chinese yes/no questions
keep producing echoes (a false echo was also seen on a thesis bait item,
which the judge caught) and a false flag only sends a claim to review.
`ECHO_RULE=off` disables it. It does not help when the judge rejects a true
echo (Pratviel R04); that needs a better judge.

### Retrieval — chunking ablation (dissertation)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| fixed-size (baseline) | 69% | 88% | 88% | 0.794 |
| **section-aware** | **92%** | **96%** | **100%** | **0.952** |

*26 hand-authored golden questions, `bge-small-en-v1.5`, hybrid (dense + BM25) +
RRF fusion, no reranker.* Section-aware chunking clears the ≥ 90% Hit@k target
and beats the fixed baseline by 12 points at Hit@5. With `bge-m3` the fixed
baseline also reaches 100% Hit@5 (Hit@1 73%, MRR 0.835) and section-aware
chunks keep the edge at the top rank (Hit@1 88%, MRR 0.929): a stronger
embedder narrows the gap that chunking closes. On the pilot paper the order
flips slightly (section 95%, fixed 100% Hit@5): a short journal article has
fewer, shorter sections, so the advantage is corpus-dependent.

### Judge validation

The judges are scored on a known-answer set of (claim, passage) pairs (see
[Validating the judges](#validating-the-judges-themselves)). **False-accept** =
an error judged `supported` (the dangerous direction); **false-reject** = a true
claim rejected.

**Prompt v1 → v2, dissertation, held-out split:**

| Judge / panel | False-accept | False-reject | Note |
|---|---|---|---|
| qwen2.5:7b, v1 | 27% | — | overclaim detection 48% |
| qwen2.5:7b, v2 | 4% | — | overclaim detection 92% |
| **qwen2.5 + gemma3:12b, v2 (previous default)** | **0%** | **4%** | 11% disputed |
| llama3 | 35% | — | unfit |
| gemma3:4b, v2 | — | most true claims rejected | unfit |

**Prompt v3, Stage A** — only the two newer error types (plan read as result,
cited study attributed to this study), both splits; false-accept, lower is
better:

| Judge | Dissertation (n=15) v2 → v3 | Pilot paper (n=5) v2 → v3 |
|---|---|---|
| qwen2.5:7b | 100% → 87% | 100% → 80% |
| gemma3:12b | 100% → 47% | 80% → 80% |
| deepseek-r1:8b | 80% → **13%** | 80% → 60% |
| gemma3:12b + deepseek-r1:8b | 80% → **7%** | 80% → 60% |

**Prompt v3, Stage B** — held-out split, all item types. Dissertation: 83 items
(28 true claims, 55 errors). Pilot paper: 55 items (20 true, 35 errors).

| Judge / panel | Dissertation FA ↓ / FR ↓ / disputed | Pilot paper FA ↓ / FR ↓ / disputed |
|---|---|---|
| **phi4 (new default)** | **5%** (3/55) / 7% (2/28) / 0% | **6%** (2/35) / 5% (1/20) / 0% |
| phi4 + gemma3:12b (unanimous) | **2%** (1/55) / 7% / 13% | **3%** / 5% / 24% |
| gemma3:12b | 15% / 0% / 0% | 34% / 0% / 0% |
| mistral-nemo | — | 40% / 15% / 0% |
| deepseek-r1:8b | — | invalid — see [Known limitations](#known-limitations) |

*FA = false-accept, FR = false-reject.* Detection by error type on the
dissertation (phi4 / panel): negation, number, conjunction and swapped passage
100% / 100%; overclaim 96% / 100%; **plan→result 67% / 83%** (n=6).

**Section heading shown to the judge** (as the live verifier now does;
`judge-eval --with-heading`): phi4 on the dissertation's held-out split scored
the same 5% FA / 7% FR, with every error type caught at the same rate; 3 of 83
verdicts moved between partial and unsupported, none across accept/reject.

Takeaways: the generator model is the weakest judge of its own output (qwen2.5
barely moves under v3); **phi4 is the strongest single judge and consistent
across both corpora**, so it became the default; gemma3:12b alone lets number
changes and plan→result errors through; adding gemma3:12b to phi4 lowers
false-accept to 2–3% at the cost of 13–24% of claims going to human review.
Samples are small (one item = 2–5%), and synthetic items are easier than real
generator errors.

---

## Architecture

```
docx / PDF ──► ingest ──► chunks (section-aware | fixed) ──► Chroma (bge) + BM25
                                                                    │
question ──► hybrid retrieval (dense ⊕ BM25, RRF) ──► (rerank) ──► top-5 passages
                                                                    │
            generator — XML: <claim citation_ids> + verbatim <quote>,
                        or the refusal marker "查無直接支持此說法的段落"
                                                                    │
            verification cascade, per claim, cheapest first:
              0. quote grounding   the verbatim quote must appear in a retrieved
                                   passage (no LLM). A mangled or wrong chunk id
                                   is repaired by the quote when it is unique; a
                                   quote running across chunks is accepted when
                                   every sentence is found; otherwise the claim
                                   is rejected as fabricated
              1. judge panel       independent LLM judges see only claim +
                                   passage (with its section heading); unanimous
                                   rule; accept/reject disagreement
                                   → "disputed" (human-review queue)
              2. fallbacks         a judge that times out or errors → "disputed"
                                   (review); with no judge configured at all:
                                   NLI (optional) → lexical overlap
              3. rule checks       plan/prediction read as a result → an
                                   accepted claim becomes "disputed" (no LLM)
                                                                    │
            ✔ supported / ◐ partial / ⚖ disputed / ✘ unsupported, with locator (§, page)
```

| Stage | What happens |
|---|---|
| Ingestion | `.docx` → heading-style walk → paragraphs, references, inline citations resolved to the reference list. Journal PDF → the same records via PyMuPDF (see below) |
| Chunking | `section` (never crosses a section boundary) or `fixed` (size + overlap); each chunk keeps section №, paragraph span and page |
| Indexing | Local `bge` / `e5` embeddings (with their query/passage prefixes) → ChromaDB, plus a BM25 sparse index |
| Retrieval | Dense top-k ⊕ BM25 top-k → reciprocal-rank fusion → optional cross-encoder rerank (`bge-reranker-base`) → top-5 |
| Generation | Citation-forcing XML prompt; a tolerant parser handles unclosed `<claim>` tags and a refusal marker placed inside a claim |
| Verification | Quote grounding → judge panel → NLI / lexical fallback (above) |
| Rendering | Each claim printed with its verdict symbol, reason and locator |

Why these choices:

- **Hybrid search** — academic prose is full of exact-match terms (`DiFuMo`,
  `ISFC`, `TFCE`, `dISFC`) that pure semantic embeddings blur; BM25 anchors them.
- **RRF fusion** — merges dense (cosine) and BM25 (unbounded) rankings without
  score calibration.
- **Verbatim quotes before any judge** — a string check is free, deterministic
  and catches fabricated evidence that a lenient judge might wave through.
- **Separate verification pass** — fresh LLM calls that never see the
  generator's reasoning, avoiding self-confirmation bias.
- **Several judges from different model families, unanimous** — judges from one
  family tend to share blind spots. Disagreement is surfaced as `disputed`
  rather than silently resolved.
- **Format errors are counted separately** — parse failures and misplaced
  refusals are reported on their own and never folded into hallucination numbers.

### Prompt & context engineering

What the model sees, and in what shape, is designed rather than left to
defaults — and each choice is measured.

**Generator prompt (citation-forcing).** Every claim must come back as XML:
`<claim citation_ids=...>` plus a verbatim `<quote>` from the cited passage, or
the explicit refusal marker "查無直接支持此說法的段落". This turns "is it
grounded?" into a string check before any LLM judge runs. A tolerant parser
handles unclosed tags and a refusal marker placed inside a claim, and format
errors are counted separately. An ungrounded `--baseline` prompt exists for
comparison. Generator prompt `v2` (opt-in; see `GENERATOR_PROMPT` below) adds
source-language quotes, passage-type labels so hypotheses and methods are never
stated as findings, explicit attribution of cited studies, and answers to
negative facts. Measured on both corpora it improved trap answers but added
format errors for the 7B generator (more misplaced refusal markers, a copied
template placeholder) — a longer, stricter prompt is not automatically a
better one for a small model.

**Generator prompt v2.1 (default since 2026-10-19).** v2.1 keeps v2's two
useful rules (quotes in the source language; hypotheses and methods are not
results, with passage-type labels) and fixes its format problems: refusals
go in a separate `<refusal>` tag, stated positively; the template shows two
short examples with obviously fake ids instead of a placeholder; claims must
be full sentences and carry a `<quote>` even when they copy the source; and
v2's attribution and negative-fact rules are gone. Calibration found one
more format habit — a quote written after `</claim>`, or a claim that *is*
the copied source sentence — which the parser and the quote check now handle
for every prompt version. Paired full run on the dissertation, same server
and verifier for both (26 answerable questions, 37 traps):

| | v1 | **v2.1** | Paired test |
|---|---|---|---|
| Strict / lenient citation precision | 97% (29/30) / 97% | 89% (25/28) / **100%** | answers fully supported 25 vs 23 of 26, p = 0.5 |
| Unsupported answerable claims | 1 | **0** | — |
| Misplaced refusal markers | 16 | **0** | — |
| Claims without a quote | 5 | **1** | — |
| All 37 traps strict safe | 30/37 | **36/37** | p = 0.031 |
| Must-refuse traps refused | 18/19 | **19/19** | — |
| Bait items where the generator stated a hypothesis | 4 of 12 | **1 of 12** | — |
| Mean seconds per question | 148 | **126** | — |

Two of v2.1's three partially supported claims were a bare number (fixed by
the full-sentence rule; confirmed in a follow-up calibration) and a judge
reading a methods passage strictly; none was unsupported.

**Context assembly.**
- *Chunking is a context decision*: section-aware chunks never cross a section
  boundary and carry section №, paragraph span and page. On the dissertation
  they beat fixed-size chunks at Hit@5 (100% vs 88%; MRR 0.952 vs 0.794).
- *Hybrid retrieval* (dense ⊕ BM25, RRF) fills the top-5 context window, so
  exact technical terms are not lost to embedding blur.
- *Judges get isolated context*: only the claim and its passage, never the
  generator's reasoning, to avoid self-confirmation.
- *Missing context is fixed in the context, not the model*: the claim audit
  traced false rejections to chunks that lacked their section heading and to
  quotes split across chunks. Showing the heading and checking quotes across
  retrieved chunks raised strict citation precision from 84% to 97% (31
  claims; see Results).

**Judge prompts, iterated against a benchmark.** Each version below was added
to fix a failure measured on the judge validation set — e.g. v1 → v2 cut
qwen2.5's false-accept rate on the held-out split from 27% to 4% by adding an
explicit evidence-strength field. Judges answer in JSON, and a
self-contradicting vote is downgraded rather than trusted.

### Judge prompts

Set with `VERIFY_PROMPT`; compare with `judge-eval --prompts v1 v2 v3`.

| Version | Adds | Status |
|---|---|---|
| `v1` | Plain entailment: supported / partially_supported / unsupported | Superseded |
| `v2` | Explicit evidence-strength field; a claim stronger than its source (overclaim) cannot be `supported` | Previous default |
| `v3` | Two extra checks: a plan, hypothesis or planned measure restated as a result (`plan_as_result`), and another study's finding claimed for this study (`misattributed`); either one forces `unsupported` | **Default** (validated with phi4) |
| `v4` | The judge first classifies the supporting sentence (result / hypothesis / plan_or_method / background / other_study) and the claim; code forces `unsupported` for a finding drawn from a hypothesis/plan, or another study's result claimed for this one. Shortened by dropping v3's hedge/overclaim word lists | Not adopted: better on plan→result, but false-accept rose 5% → 25% on held-out data because overclaim detection collapsed |
| `v5` | v3 kept verbatim plus v4's classification step and fields | Not adopted: overclaim detection recovered (22/25), but plan→result barely improved on the hard set (5/12 vs v3's 4/12) and pilot false-accept rose 6% → 14% |

**Outcome of the v4/v5 experiments.** Adding a classification step did not make
phi4 understand study stage better: v4's gain on plan→result came with a
collapse in overclaim detection (a shift in strictness, not comprehension), and
v5, which kept v3's text, lost the gain again. v3 plus the two deterministic
plan→result rules remains the best combination measured — held-out false-accept
5–6% and all 12 hard plan→result items stopped — at no extra judge time. For a
7–14B judge, deterministic checks proved more reliable than further prompt
iterations.

Judges answer in JSON. A self-contradicting vote (`claim_stronger` but labelled
`supported`) is downgraded to `partially_supported`.

---

## Setup

```bash
pip install -r requirements.txt          # torch is assumed preinstalled
```

Place the manuscript at the repo root as
`2026_Manuscripts_BXF+NTSEC_v2.1.docx` (or point `DISSERTATION_DOCX` at it).
Embeddings are always local; models download on first run (~2.3 GB for the
default `bge-m3`). For a small English-only setup use
`EMBED_MODEL=BAAI/bge-small-en-v1.5` (~130 MB) and build its index.

Settings can go in a `.env` file (see [`.env.example`](.env.example)); real
environment variables override it.

### Choosing the LLM backend (generation + verification)

The LLM layer is pluggable via `LLM_BACKEND`. It auto-selects **Claude** if an
API key is present, otherwise **local Ollama** — so it runs fully offline out of
the box.

**Option A — Local Ollama (free, offline, default):**

```bash
ollama serve                       # if not already running
ollama pull qwen2.5:7b-instruct    # generator
ollama pull phi4                   # judge (different family from the generator)
ollama pull gemma3:12b             # optional: second judge for a stricter panel
#   defaults: OLLAMA_MODEL=qwen2.5:7b-instruct
#             JUDGES=ollama:phi4:latest   VERIFY_PROMPT=v3
```

A judge whose model is not pulled is skipped with a message; with no judge left,
the generator's own model judges.

**Compute cost on a CPU-only machine** (measured on an i7-8700, 6 cores, 64 GB
RAM, Ollama without GPU): generator + phi4 hold about 22 GB of RAM while
loaded; one phi4 judgment takes 38–50 s (about 110 s for the first call, which
loads the model), at about 55% CPU. An `ask` answer usually carries one or two
claims, so verification adds roughly 45–100 s; `check` judges each sentence
against its top two retrieved passages (`CHECK_TOP_N`) and stops at the first
that supports it, so a sentence costs one or two judgments (measured on 4
threads: ~85 s for a supported sentence, ~137 s for an unsupported one). Adding gemma3:12b as a second judge roughly
doubles the verification time, adds about 9 GB, and uses a third Ollama model
slot.

**Option B — Claude API (higher quality on the structured prompts):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export LLM_BACKEND=claude                     # optional; auto-selected when key is set
```

If **no** backend is reachable (no API key **and** no Ollama server), `ask` and
`check` fall back to **mock mode**: retrieval is real, but generation and
verification are stand-ins (pipeline shape only).

### Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `CORPUS` | `dissertation` | Which corpus to use; any other name reads `papers/<name>.pdf` |
| `SOURCE_PATH` | — | Explicit path to the corpus PDF |
| `SOURCE_LABEL` | — | How the source is named in answers, e.g. `Author (2026)` |
| `DISSERTATION_DOCX` | repo-root manuscript | Path to the dissertation `.docx` |
| `CHUNK_STRATEGY` | `section` | `section` or `fixed` |
| `EMBED_MODEL` | `BAAI/bge-m3` | Sentence-transformers embedding model (multilingual); `BAAI/bge-small-en-v1.5` was the default until 2026-10-21. Each model has its own index |
| `RERANK_ENABLED`, `RERANK_MODEL` | off, `bge-reranker-base` | Cross-encoder rerank |
| `LLM_BACKEND` | auto | `claude` or `ollama` |
| `CLAUDE_MODEL` | see `config.py` | Claude model id |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Generator model |
| `OLLAMA_THINK_BUDGET` | `2000` | Extra tokens for models with the `thinking` capability (see below) |
| `OLLAMA_SECONDS_PER_TOKEN` | `0.4` | Request timeout = max(300 s, 60 s + (output limit + ½ × prompt tokens) × this) |
| `OLLAMA_NUM_THREAD` | `0` (Ollama default) | CPU threads per request for this project, e.g. `3` on a shared machine (raise `OLLAMA_SECONDS_PER_TOKEN` too) |
| `JUDGES` | `ollama:phi4:latest` | Comma-separated `backend:model` judge list |
| `PANEL_RULE` | `unanimous` | `unanimous` or `majority` |
| `VERIFY_PROMPT` | `v3` | Judge prompt version |
| `QUOTE_REQUIRED` | off | `1` also rejects claims without a supporting quote |
| `CJK_QUERY_DENSE_ONLY` | `auto` | Skip BM25 for a query with Chinese characters: `auto` when `EMBED_MODEL` is multilingual (bge-m3), `on`, `off` |
| `CONTEXT_ORDER` | `rank` | Order of the passages shown to the grounded generator: `rank` (retrieval order) or `results_first` (results/discussion, then methods/background, hypotheses last). Tried, not adopted |
| `GENERATOR_PROMPT` | `v2.1` | Generator prompt (see "Generator prompt v2.1"); `v1` is the original; `v2` adds source-language quotes, passage-type labels (hypotheses/methods never stated as findings), explicit attribution of cited studies, negative facts answered, refusal marker only in `<unsupported_note>`. Evaluated, not adopted: better on trap questions, but more format errors on answerable ones (see Known limitations). `v2.2` = v2.1 + a yes/no-question rule; calibrated, not adopted (see "End to end with Chinese questions") |
| `PLAN_RESULT_RULE` | `flag` | Plan or prediction read as a result: `flag` → disputed, `reject` → unsupported, `off` |
| `ECHO_RULE` | `flag` | A partially supported claim that restates a yes/no question (embedding similarity ≥ `ECHO_MIN_SIM` 0.90 and character overlap ≥ `ECHO_MIN_OVERLAP` 0.70, question words removed) is sent to review as `disputed`; `off` disables. Not independently validated (see "End to end with Chinese questions") |
| `PLAN_RESULT_MATCH` | `on` | Hypothesis-match route of that rule (partially supported claims closer to a hypothesis in the context than to their citation); `off` disables |
| `PLAN_MATCH_MODEL` / `PLAN_MATCH_MIN_SIM` / `PLAN_MATCH_MARGIN` | `BAAI/bge-m3` / 0.55 / −0.02 | Embedder and thresholds of that route |

### Local Ollama notes

- **Reasoning models** (e.g. `deepseek-r1`). Since Ollama 0.35 their thinking
  comes back in a separate `thinking` field, and those tokens count toward the
  output limit. A judge call allows 300 tokens, which `deepseek-r1:8b` spent
  entirely on thinking, leaving an empty answer (`think: false` is ignored by
  that model). The client now reads each model's capabilities from `/api/show`
  and gives models reporting `thinking` an extra `OLLAMA_THINK_BUDGET` tokens;
  an empty answer after thinking is logged. Expect such judges to be slow on
  CPU (a `deepseek-r1:8b` judgment takes 4–8 minutes); the request timeout
  scales with the token limit (`OLLAMA_SECONDS_PER_TOKEN`). Older
  servers that return `<think>…</think>` inline are still handled by stripping
  the block.
- **Keeping several models loaded.** A judge panel switches models on every
  claim. If Ollama can only fit one model at a time — typically because a small
  GPU is visible and only one model's compute buffer fits on it — every switch
  reloads a model from disk. On a machine with ample RAM, running Ollama
  CPU-only (hide the GPU from the Ollama process with
  `CUDA_VISIBLE_DEVICES=-1`, and for Ollama 0.35+ also `OLLAMA_VULKAN=0`,
  `GGML_VK_VISIBLE_DEVICES=-1`) lets up to `OLLAMA_MAX_LOADED_MODELS` models
  stay resident. Check the server log: `inference compute` should report
  `library=cpu`, and `predicted to exceed available memory, evicting` means a
  GPU is visible again.
- **Pick judges from different families** — check `ollama show <model>`.
  `deepseek-r1:8b`, for instance, reports architecture `qwen3` (a Qwen3 model
  distilled from DeepSeek-R1), so pairing it with `qwen2.5` is not a
  cross-family panel.

---

## Data & privacy

The main corpus is an **unpublished PhD dissertation**, so nothing derived from
it is committed. `.gitignore` excludes the manuscript (`*.docx`), `papers/`, the
generated chunks and indexes, every golden set, judge set and vote cache, and
the evaluation reports (`eval/reports/`). What ships is **code, methodology,
architecture and aggregate metrics** only.

To run it on your own corpus: drop a `.docx` at the repo root (or set
`DISSERTATION_DOCX`), then author a golden set following
[`data/golden/golden_set.example.json`](data/golden/golden_set.example.json) and
save it as `data/golden/golden_set.json`.

### Other corpora (journal-article PDFs)

`CORPUS` switches the whole pipeline to another document. The default,
`dissertation`, keeps the layout above; any other name reads `papers/<name>.pdf`
(or `SOURCE_PATH`) and keeps its chunks, index, golden set, judge set and
reports under `data/corpora/<name>/` and `eval/reports/<name>/`, so corpora
never mix.

```bash
# PowerShell: $env:CORPUS="my_paper"; $env:SOURCE_LABEL="Author (2026)"
export CORPUS=my_paper SOURCE_LABEL="Author (2026)"
python -m src.ingest_pdf papers/my_paper.pdf    # inspect how the PDF was parsed
python cli.py build --all
python cli.py ask "..."
```

The PDF loader (hardened on a real published article):

- drops running headers/footers, page numbers and publisher boilerplate, and
  skips the title page;
- detects numbered, named, bold, italic and letter-spaced headings (`A B S T R A C T`),
  and gives an unheaded opening its own Introduction section;
- joins hyphenated line breaks and removes soft hyphens;
- merges table fragments into one record;
- splits off the reference list into individual entries;
- records page numbers, so citations read `§2.1 (p. 3)`.

Scanned PDFs without a text layer are not supported (they need OCR).

---

## Usage

```bash
# 1. Ingest + index both chunking strategies
python cli.py build --all

# 2. Ask a citation-grounded question
python cli.py ask "Which theory explains how natural environments restore directed attention?"
python cli.py ask "..." --k 8 --strategy fixed      # more passages / other chunking
python cli.py ask "..." --baseline                   # ungrounded prompt, for comparison

# 3. Reverse hallucination check — paste something you wrote, get per-sentence verdicts
python cli.py check "Forest therapy reduced salivary cortisol and improved sleep quality."
type draft.txt | python cli.py check                 # or pipe text via stdin

# 4. Evaluation harness — retrieval only (fast, no LLM)
python cli.py eval
python cli.py eval --rerank                          # with cross-encoder reranker
#    ...with grounding / hallucination / refusal via the LLM backend:
python cli.py eval --with-llm
python cli.py eval --with-llm --llm-limit 3          # quick sample for a slow local model
python cli.py eval --with-llm --traps-only           # trap questions only
python cli.py eval --with-llm --traps-only --trap-types hypothesis_bait  # one trap tier
python cli.py eval --with-llm --answerable-only      # answerable questions only
python cli.py eval --with-llm --fresh                # ignore an interrupted run's saved questions
python cli.py eval --with-llm --baseline             # ungrounded arm, judged sentence by sentence
python cli.py eval --with-llm --baseline --reuse-from <run>  # re-score saved baseline answers
python cli.py eval --with-llm --lang zh              # ask every question in Chinese (question_zh)
python -m eval.rejudge [--limit 3]                  # sensitivity: re-judge non-supported claims of both arms on all 5 passages

# 5. Judge validation
python cli.py judge-build                            # build the known-answer set (no LLM)
python cli.py judge-eval --judges ollama:phi4:latest ollama:gemma3:12b --prompts v2 v3 --split test
python cli.py judge-eval --types plan_to_result attribution_swap --limit 10
python cli.py judge-eval --with-heading               # passages shown with their section heading

# 6. Unit tests for the deterministic verification rules (no LLM, no corpus)
python -m pytest tests
```

`check` is the writer-facing use: it splits your paragraph into sentences,
retrieves evidence for each, and runs the same verification cascade, so you see
which of your own statements the source does not support. How many passages
it judges was measured, not guessed (`python -m eval.check_cost`, 40 labelled
sentences, 2026-10-17):

| Passages judged | False reject | False accept | Mean judge calls | Mean s / sentence |
|---|---|---|---|---|
| 3 (previous default) | 0/22 | 0/18 | 2.0 | 146 |
| **2 (default)** | 1/22 | 0/18 | 1.5 | **111** |
| 1 | 3/22 | 0/18 | 1.0 | 71 |

Two passages keep false accepts at zero and cost one false alarm (a true
sentence whose evidence was ranked third), so the safe direction absorbs the
saving. Re-ordering the passages by embedding similarity changed nothing (the
order was identical). Two of the 20 perturbed items were relabelled true for
this measurement: they restate a hypothesis as a finding, which is false
against the hypothesis passage they were built from, but the thesis does
report both as results and `check` correctly found those passages. (Flagging
a sentence whenever *any* passage reads as a plan was considered and
rejected for the same reason: it would flag true results whose plan also
appears in the methods.)

### Multi-model cross-check (judge panel)

```bash
# PowerShell: $env:JUDGES="ollama:phi4:latest,ollama:gemma3:12b"
export JUDGES="ollama:phi4:latest,ollama:gemma3:12b"
export PANEL_RULE=unanimous      # or "majority" (useful with 3+ judges)
python cli.py eval --with-llm
```

Under `unanimous`, judges that disagree on *whether to accept* a claim yield
`disputed` (⚖), which counts as neither support nor hallucination and is the
queue for human review. Judges that all reject and differ only on severity
(partial vs unsupported) are resolved by majority, ties going to `unsupported`.
Leave `JUDGES` unset for a single judge on the generator's backend.

---

## Evaluation methodology

Two layers: first validate the judges, then use them to measure the system.
See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the design write-up.

### System evaluation (`eval --with-llm`)

The golden set (`data/golden/golden_set.json`, kept local — schema in
[`golden_set.example.json`](data/golden/golden_set.example.json)) has two parts:

- **Retrieval questions** (26 on the dissertation, 20 on the pilot paper) —
  natural-language information needs, each tied to the paragraph index(es) that
  answer it. A retrieved chunk "hits" if its paragraph span covers a target, so
  the metric is **strategy-agnostic** and fairly compares fixed vs section-aware
  chunking. All are answerable, so they double as the over-refusal check.
- **Trap questions** (25 / 17) in three tiers of difficulty:
  - *far_absent* — topics unrelated to the source. Must refuse.
  - *near_absent* — plausible questions about the source's own studies whose
    answer is not in the text, including measures the methods mention but never
    report results for. Must refuse.
  - *false_premise* — questions presupposing something the text contradicts.
    Correct if no claim the judges reject or dispute gets through — refusing and
    correcting the premise both pass.

  Every trap also records a *safe* rate (no rejected or disputed claim), which
  separates "did not refuse" from "made something up".

| Metric | Definition | Target |
|---|---|---|
| Retrieval Hit Rate@k / MRR | correct chunk in top-k / reciprocal rank | ≥ 90% |
| Citation precision (strict) | claims judged `supported` / all claims | ≥ 95% |
| Citation precision (lenient) | `supported` + `partially_supported` / all claims | — |
| Claim hallucination rate | unsupported claims / all claims | low |
| Answer hallucination rate | non-refused answers with ≥ 1 unsupported claim / non-refused answers | low |
| Fabricated-quote rate | claims whose quote is not in any retrieved passage | low |
| Repaired citation ids | wrong ids fixed by a unique quote match | reported |
| Panel agreement / disputed | how often judges agree; claims sent to review | reported |
| Refusal correctness | must-refuse traps refused, by tier | ≥ 90% |
| False-premise safe rate | false-premise traps with no rejected/disputed claim | high |
| Over-refusal rate | answerable questions wrongly refused | low |
| Format errors | parse failures, refusal marker inside a claim | reported separately |
| Chunking ablation | fixed vs section-aware on the above | — |

Claim-level metrics are **micro-averaged**: claims are pooled across questions
before dividing, so a question with many claims is not under-weighted.
`eval_results.json` (local) records every claim with its quote, cited passage,
verdict, judge votes and reason, so a metric change can be audited claim by
claim.

### Validating the judges themselves

Every grounding number above is only as good as the judges, so they get their
own benchmark — a **judge validation set** of (claim, passage) pairs with known
gold labels, built without any LLM by perturbing real sentences:

| Perturbation | Example edit | Gold |
|---|---|---|
| original | verbatim sentence vs its own passage | supported |
| negation | "increased" → "decreased", "was" → "was not" | unsupported |
| number | a sample size / duration / statistic changed | unsupported |
| overclaim | "may" → "will always", "suggests" → "proves" | partially_supported |
| conjunction | sentence + an unrelated claim appended | partially_supported |
| swap_passage | sentence vs an unrelated passage | unsupported |
| plan_to_result | a planned or hypothesised step restated as a finding | unsupported |
| attribution_swap | "Author et al. found …" → "This study found …" | unsupported |
| regression | real generator outputs a judge once approved, pinned with a verified label | as labelled |

The headline metric is the **false-accept rate**, always read with the
false-reject rate, plus 3-class accuracy, Cohen's κ, a per-perturbation
breakdown and the disputed rate for panels. Items are split dev/test by source
sentence, so anything tuned on dev is reported on test.

Votes are cached per corpus (`judge_votes.jsonl`, keyed by prompt version,
judge and item), so a rerun only calls the judges on new items, and panels are
scored offline from the same cached votes — adding a judge costs only that
judge's calls.

Synthetic positives are verbatim and therefore easy, and synthetic errors are
cleaner than real ones. Add your own labelled pairs to
`data/golden/judge_set_human.jsonl` (schema:
[`judge_set_human.example.jsonl`](data/golden/judge_set_human.example.jsonl)).

---

## Running long local evaluations

A full judge or system evaluation on local models runs for hours.
`eval/resource_guard.py` wraps such a run and stops it if the machine is
overloaded:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -u -m eval.resource_guard --log-every-s 300 --models phi4:latest gemma3:12b -- python -u cli.py judge-eval --judges ollama:phi4:latest ollama:gemma3:12b --prompts v3 --split test
```

When it stops a job, the guard unloads only the Ollama models named in
`--models` (the job's own). Other loaded models are left alone, because the
Ollama server may be shared with other projects and a model loaded mid-run may
be theirs; without `--models` nothing is unloaded.

Both long jobs **resume after an interruption** (guard stop, crash, closed
session): rerun the same command and only the missing work is done; reports
are written at the end.

- `judge-eval` caches every vote as soon as it is cast.
- `eval --with-llm` appends each scored question to
  `eval/reports/eval_progress.jsonl`. Saved questions are reused only by a run
  with the same signature — same models, judge prompt, rule settings, top-k,
  embedding model, and the same generation / verification / retrieval code —
  so a resumed run never mixes answers produced under different conditions.
  When the run completes the file is renamed (`eval_progress.done-<run>.jsonl`)
  and the next run starts fresh; `--fresh` ignores saved questions on demand.

| Flag | Default | Stops the run when |
|---|---|---|
| `--max-ram-pct` | 90 | RAM use exceeds this |
| `--min-free-gb` | 4 | free RAM falls below this |
| `--max-swap-growth-gb` | 2 | swap grows by more than this |
| `--max-cpu-pct` | 95 | CPU stays above this for `--cpu-window-s` (180 s) |
| `--interval-s` / `--log-every-s` | 5 / 60 | sampling and logging intervals |
| `--models` | none | (not a limit) Ollama models to unload when the guard stops the job |

Practice that has worked: run a short calibration first (a few items per new
model) to replace guessed timings with measured ones; launch long runs detached
(e.g. PowerShell `Start-Process`) rather than from a tool session that may time
out; and copy `eval_report.md` / `judge_report.md` before rerunning, because
reports are overwritten. Measured load on CPU: four 7–14B judges peaked at
85% CPU and 64% RAM of 64 GB; phi4 + gemma3:12b on 83 items took about 2 h at
about 55% average CPU. If another job uses the same Ollama server at the same
time, both slow down sharply and the CPU limit can trip, so check for other
Ollama clients right before launching.

---

## Known limitations

1. **Plan read as result — mitigated by a rule.** The generator once restated a
   planned assessment as a reported finding, and every judge approved that real
   case. A deterministic check now flags it: the claim asserts a finding without
   hedging while the source sentence plans or predicts (the quote's sentences,
   or, without a quote, a passage sentence planning an assessment or predicting
   an outcome). It catches the real case and sends it to review as `disputed`;
   offline it flagged 0 of 100 true validation claims and 0 of 31 real system
   claims, and lowered phi4's false-accept on the dissertation's held-out split
   from 3/55 to 1/55. On a hand-written hard set (plans restated as findings
   in natural Chinese) it caught 7 of 12 — every one using a result cue word,
   none of the 5 phrased without one — with 0 of 14 controls flagged, so its
   recall depends on wording. **phi4 is weak on the same items**: it caught
   4 of 12 (33% [14–61], all only as partial), approved 4 hypotheses restated
   as findings although the passage heading said "Hypotheses", and rejected 2
   of 6 correctly hedged plans. Judge + rule caught 8 of 12 and miss the same
   cue-free items. A **structural route** now closes most of that gap: an
   unhedged claim citing a chunk whose innermost heading is a Hypotheses
   section is flagged whatever its wording. With it, phi4 + rule let 0 of 12
   hard items through — but the route was designed after seeing them, so that
   is not an independent test; on independent data it flagged 0 of 100 true
   validation claims and 0 of 87 real system claims. It only helps documents
   that put hypotheses under their own heading.
   Caveats: the synthetic items share the rule's cues; a true finding cited
   from a future-tense methods passage could still be flagged, which is why the
   default is review, not rejection.
   **Hypothesis-match route (2026-10-15).** A hypothesis restated as a
   finding that cites a *results* passage with a verbatim quote escaped both
   routes (the cited section is not a Hypotheses section, the cited sentence
   is not phrased as a plan) and phi4 rated it only partially supported — one
   real case in 6 bait items. A third route now covers it: for an unhedged
   claim the judge rated *partially supported*, when a Hypotheses section is
   among the retrieved passages, it compares the claim with every hypothesis
   sentence and with its cited passage using a multilingual embedder
   (`BAAI/bge-m3`; claims are often Chinese, hypotheses English) and flags it
   when the hypothesis is the closer match (similarity ≥ 0.55, margin ≥
   −0.02). Embeddings alone cannot tell a confirmed hypothesis from an
   unconfirmed one (a claim restating a hypothesis the results confirm looks
   the same), hence the limit to partially supported claims. Re-applied to
   every saved output (32 partially supported claims, both arms, generator
   v1 and v2) it flagged the real case and one baseline sentence restating a
   hypothesis, and none of the other 30 (0% [0–11]); no answerable or
   non-bait trap metric changed. The thresholds were set on these same
   outputs and there is one real positive, so it is not independently
   validated; `PLAN_RESULT_MATCH=off` disables it. In the live 12-item bait
   run (2026-10-16, both arms) it fired 3 times — the original case again
   and two baseline sentences, one on a new item — all of them hypotheses
   restated as findings, no false flag; still a small, same-corpus check.
   **Independent check (2026-10-22, Pratviel et al. 2024, a paper with a
   Hypotheses heading the rules were not tuned on):** no flag on a true
   grounded claim; in the baseline, one correct catch (hypothesis-match) and
   one false flag of a true finding worded like its hypothesis, caused by
   judging a sentence against the hypothesis passage alone — fixed by
   applying the rule per passage and adding the closest non-hypothesis
   passage (see "Independent check of the hypothesis routes").
2. **Background statements attributed to this study.** v3 catches most
   "Author et al. found → This study found" swaps, but on the pilot paper at
   most 40% of swaps of background statements.
3. **Self-judging — resolved in the default.** qwen2.5 was both generator and
   judge, and the weakest judge; the default judge is now phi4.
4. **Negative facts.** "The paper says X was not done" tends to be refused
   (over-refusal).
5. **Format drift.** The refusal marker inside `<claim>`, translated quotes and
   unclosed tags occur; all are handled and counted, not hidden. A stricter
   generator prompt (v2) did not fix this for the 7B generator: on the
   dissertation it raised trap scores (near-absent 89% → 100%, false-premise
   67% → 83%) but doubled misplaced refusal markers (13 → 25), copied a
   template placeholder as a citation id, and lowered strict precision 97% →
   90% through citation errors on true claims — all within the intervals, so
   v1 stays the default.
6. **Judges are not deterministic.** The same pair can get different votes
   across runs; decide on larger sets or repeated runs.
7. **Reasoning-model judges are too slow on CPU.** With room for its thinking,
   `deepseek-r1:8b` needs 4–8 minutes per judgment (one exceeded 5 minutes), so
   it was dropped as a judge; its Stage B result above is invalid.
8. **Small samples.** Judge rates rest on 55–83 items and system rates on
   20–51 questions; reports show 95% intervals (see "How certain are these
   numbers?"), and most differences discussed here fall inside them.
9. **Verification is slow on CPU.** Each phi4 judgment takes 38–75 s
   depending on threads and prompt length, so `check` on a long paragraph
   can take several minutes even with two passages per sentence (see Setup). A judgment that still times out sends the claim to
   review rather than guessing; `n_unjudged` in the eval results counts these.
10. **Chunk boundaries.** A chunk judged alone can lack context a true claim
    relies on, and a quote can run across two chunks. Both are now mitigated
    (section heading shown to the judge; quotes checked sentence by sentence
    across retrieved chunks), but a claim whose support is split across chunks
    is still judged one chunk at a time.
11. **Answer completeness is not measured.** Metrics count whether claims are
    supported, not whether they answer the question: a grounded answer can
    be fully supported yet give the method or the hypothesis instead of the
    result when the results passage is not retrieved.

## Roadmap

1. ~~Re-validate `deepseek-r1:8b`~~ — dropped as too slow on CPU.
2. ~~Stage B on the dissertation's held-out split~~ — done (results above).
3. ~~Choose a panel without the generator model~~ — phi4 alone, prompt v3.
4. ~~Rerun the system evaluation under the new default~~ — done, with a
   claim-by-claim audit (Results).
4a. ~~Quote spanning chunks; section heading shown to the judge~~ — done;
    strict precision 84% → 97% on the dissertation's answerable questions.
4b. ~~Check that the section heading does not raise false-accept~~ — done; it
    does not.
5. ~~Plan→result rule check~~ — done (`PLAN_RESULT_RULE`). Still useful:
   harder plan→result items shaped like real generator outputs, and real
   positives to measure the rule's recall.
6. Generator prompt: answer negative facts, attribute cited studies explicitly,
   keep quotes in the source language, refuse only in `<unsupported_note>`.
7. ~~Resumable `run_eval`~~ — done (saves each question; resumes on rerun).
8. ~~Confidence intervals~~ — done (Wilson, in both reports). Still open:
   human-labelled judge items.
9. ~~Grounded vs baseline prompt comparison~~ — done (Results), including the
   12 hypothesis-bait traps, a sensitivity re-judging on all 5 passages, a
   strict-safe trap rule and the hypothesis-match rule route, plus an
   out-of-corpus check on two open-access papers. Still open: an
   open-access paper with a Hypotheses section, to test the structural and
   hypothesis-match routes on outputs they were not tuned on.

---

## Layout

```
config.py              all tunable knobs; reads .env; CORPUS switches corpora
cli.py                 build | ask | check | eval | judge-build | judge-eval
src/
  ingest.py            docx → paragraphs, references, inline citations, chunkers
  ingest_pdf.py        journal PDF → the same records (PyMuPDF), with page numbers
  embedder.py          local sentence-transformers wrapper (bge/e5 prefixes)
  index.py             ChromaDB dense index + BM25 sparse index
  retrieve.py          hybrid search, RRF fusion, optional cross-encoder rerank
  generate.py          citation-forcing + baseline prompts, tolerant XML parsing
  verify.py            quote grounding, citation repair, judge panel, prompts v1–v3,
                       NLI / lexical fallbacks
  pipeline.py          ask() and check() orchestration, CLI rendering
  llm.py               Claude / Ollama backends; thinking-model handling; mock mode
eval/
  run_eval.py          system evaluation harness → eval/reports/
  judge_set.py         builds the known-answer judge validation set
  judge_eval.py        scores judges / panels / prompt versions (vote cache)
  resource_guard.py    stops long local-LLM runs on RAM / CPU / swap overload
  stats.py             95% Wilson confidence intervals for the reports
data/golden/           example schemas (real golden and judge sets are local)
tests/                 unit tests for the deterministic verification rules
docs/METHODOLOGY.md    design write-up
HANDOFF.md             current state, open findings and next steps
```
