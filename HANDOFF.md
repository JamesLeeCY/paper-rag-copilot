# Handoff — Dissertation RAG Copilot

State as of **2026-10-07**. Read this first when picking the project up. The
README covers setup and usage; this file covers what has been built, what the
measurements say, what is in flight, and what to do next.

> **Privacy rule for everything committed here:** the main corpus is an
> unpublished dissertation. Committed files carry code, methodology and
> aggregate numbers only, never its content or study-design details. All
> corpora, golden sets, judge sets, votes and reports are gitignored (see
> `.gitignore`); keep it that way.

---

## 1. What the system does

A citation-grounded RAG assistant over a single document (or a small set of
journal PDFs), built so that its hallucination rate can be **measured**, not
just hoped for. Every claim it outputs carries a chunk id and a verbatim quote,
and passes an independent verification layer before it is shown as supported.

## 2. Architecture

```
docx / PDF ──► ingest ──► chunks (section-aware | fixed) ──► Chroma (bge) + BM25
                                                                    │
question ──► hybrid retrieval (dense ⊕ BM25, RRF) ──► top-5 passages
                                                                    │
            generator (qwen2.5:7b) — XML: <claim citation_ids> + <quote>, or refusal marker
                                                                    │
            verification cascade, per claim:
              0. quote grounding   verbatim quote must be in a retrieved passage
                                   (no LLM; mangled/wrong chunk ids are repaired
                                    by the quote when it is unique; else fabricated)
              1. judge panel       independent LLM judges, unanimous rule;
                                   accept/reject disagreement -> "disputed"
              2. fallbacks         NLI (optional) -> lexical overlap
              3. rule checks       plan/prediction read as a result -> an accepted
                                   claim becomes "disputed" (PLAN_RESULT_RULE)
                                                                    │
            ✔ supported / ◐ partial / ⚖ disputed / ✘ unsupported, with locator (§, page)
```

| Module | Role |
|---|---|
| `config.py` | All knobs; reads `.env` (real env vars win); `CORPUS` switches corpora |
| `src/ingest.py` | `.docx` → paragraphs (heading styles), references, chunkers |
| `src/ingest_pdf.py` | Journal PDF → same records (PyMuPDF): headers/footers, boilerplate, bold/italic/unspaced headings, unheaded intro, soft hyphens, tables merged, reference splitting, page numbers |
| `src/index.py`, `src/embedder.py` | Chroma dense index + BM25, bge/e5 prefixes |
| `src/retrieve.py` | Hybrid search, RRF, optional cross-encoder rerank |
| `src/generate.py` | Citation-forcing prompt; tolerant XML parser (unclosed `<claim>`, refusal marker inside `<claim>`) |
| `src/verify.py` | Quote check, citation repair, judge panel, prompts v1/v2/v3 |
| `src/llm.py` | Claude / Ollama backends; thinking-model budget, timeout scaled to token limit; strips `<think>`; checks the model is pulled |
| `src/pipeline.py` | `ask`, `check` (verifies a user's own sentences), CLI rendering |
| `eval/run_eval.py` | Main harness: retrieval, grounding, refusal, traps by tier |
| `eval/judge_set.py` | Builds the known-answer judge validation set (rule-based perturbations) |
| `eval/judge_eval.py` | Scores judges / panels / prompt versions; vote cache makes it resumable |
| `eval/resource_guard.py` | Wraps long local-LLM jobs; stops them on RAM/CPU/swap overload; unloads only the job's own models (`--models`) |

### Default configuration (changed 2026-10-07)

| Setting | Value | Why |
|---|---|---|
| Generator | `qwen2.5:7b-instruct` | Follows the XML/quote format far better than llama3 |
| Judge | **`phi4`** alone | Best single judge on both corpora under v3 (held-out: dissertation 5% FA / 7% FR, pilot 6% / 5%, 0% disputed); takes the generator model off the judge seat. Previous: `qwen2.5` + `gemma3:12b`, unanimous |
| Judge prompt | **`v3`** | phi4 was validated under v3. Previous: `v2` |
| Chunking | section-aware | Best on the dissertation; see §4 for the pilot paper |

**The system results in §4 were measured under the previous default**; the
rerun under the new default is §7 step 1. Stricter option:
`JUDGES=ollama:phi4:latest,ollama:gemma3:12b` (FA 2–3%, but 13–24% disputed).

Corpora: `CORPUS=dissertation` (default) and `CORPUS=nature_walking_2026`
(pilot: Watkins-Martin et al., 2026, *J. Environ. Psychol.* 115:103188, CC-BY;
PDF at `papers/nature_walking_2026.pdf`, local only). For the pilot also set
`SOURCE_LABEL="Watkins-Martin et al. (2026)"`.

## 3. How hallucination is evaluated

Two layers: first validate the judges, then use them to measure the system.

**Layer 1 — judges (`judge-build`, `judge-eval`).** Real sentences are
perturbed by rule into items with known gold labels: original (supported),
negation, number, overclaim, conjunction, swapped passage, and (new)
plan→result and attribution swap; plus pinned **regression items** (real
generator outputs a judge once approved). Split dev/test by source sentence.
Headline metric: **false-accept rate** (an error judged supported), always read
with false-reject rate.

**Layer 2 — system (`eval --with-llm`).** Strict/lenient citation precision,
claim- and answer-level hallucination rate, fabricated-quote rate, repaired
citation ids, panel agreement, trap refusal by tier (far-absent, near-absent,
false-premise; correct = refused for the first two, no rejected/disputed claim
for false-premise), **over-refusal** on answerable questions, and format errors
(parse failures, misplaced refusals) reported separately, never folded into
hallucination numbers.

## 4. Results so far

**Dissertation** (26 answerable + 10 far-absent traps, default config):
strict citation precision **96%** (26/27), claim and answer hallucination
**0%**, trap refusal **100%**, over-refusal **0%**. Harder trap set (25):
far-absent 100%, near-absent 89%, false-premise 83%; must-refuse refusal 95%.
Retrieval: section-aware Hit@5 100% (MRR 0.952) vs fixed 88% (MRR 0.794).

**Judge validation, held-out split:** qwen2.5 false-accept 27% (v1) → 4% (v2);
overclaim detection 48% → 92%. Panel qwen2.5 + gemma3:12b: false-accept 0%,
false-reject 4%, disputed 11%. llama3 (35% false-accept) and gemma3:4b
(rejects most true claims under v2) are unfit as judges.

**Pilot paper** (20 answerable + 17 traps): retrieval Hit@5 95% (section) /
100% (fixed); strict citation precision **90%** (18/20, below the 95% target);
must-refuse traps **100%** (11/11); false-premise 67% (4/6); over-refusal 5%;
**no false claim was output as supported** — all four generator errors were
caught (three rejected, one disputed; one of them by the quote check alone).

**Prompt v3, Stage A** (new error types only, both splits; false-accept rate,
lower is better — no true claims in this subset, so false-reject is not
measured here):

| Judge | Dissertation (n=15) v2 → v3 | Pilot paper (n=5) v2 → v3 |
|---|---|---|
| qwen2.5:7b | 100% → 87% | 100% → 80% |
| gemma3:12b | 100% → 47% | 80% → 80% |
| deepseek-r1:8b | 80% → **13%** | 80% → 60% |
| panel gemma3:12b + deepseek-r1:8b | 80% → **7%** | 80% → 60% |

By type under v3: deepseek-r1 catches 91% of synthetic plan→result items and
75% of the dissertation's attribution swaps; gemma3:12b catches 100% of those
attribution swaps but 36% of plan→result. The pilot paper's attribution
swaps (background statements prefixed "In this study, …") stay hard: at most
40% caught.

**Real regression cases under v3:** the plan-read-as-result claim is still
approved by **all three** judges; the cited-study claim is rejected by
deepseek-r1 only (gemma3:12b moved from partial under v2 to supported under v3).

**Prompt v3, Stage B step 2** (pilot paper, held-out split, all item types:
55 items, 20 true claims, 35 errors; 2026-10-06):

| Judge / panel | False-accept ↓ | False-reject ↓ | Disputed | Note |
|---|---|---|---|---|
| **phi4** | **6%** (2/35) | **5%** (1/20) | 0% | Best single judge; caught every attribution swap, negation, number change and conjunction |
| **gemma3:12b + phi4** (unanimous) | **3%** | **5%** | 24% | Lowest false-accept, but a quarter of claims go to human review |
| gemma3:12b | 34% | 0% | 0% | Missed half the negations and number changes, all conjunctions |
| mistral-nemo | 40% | 15% | 0% | Worst on both errors; drop |
| deepseek-r1:8b | — | — | — | **Invalid: 95% of outputs unparsable** (see §5, item 7) |

Small sample: 5% false-reject is one claim. gemma3:12b's weak showing is
unexplained — it has not been run under v2 on this corpus, so v3 and corpus
difficulty cannot be separated yet. Panels containing deepseek-r1 equal the
same panel without it (its unparsable votes are dropped).

**Prompt v3, Stage B step 1** (dissertation, held-out split, all item types:
83 items, 28 true claims, 55 errors; 2026-10-07 17:08–19:05):

| Judge / panel | False-accept ↓ | False-reject ↓ | Acc. (3-class) | κ | Disputed |
|---|---|---|---|---|---|
| **phi4** | **5%** (3/55) | 7% (2/28) | 81% | 0.71 | 0% |
| gemma3:12b | 15% (8/55) | **0%** | 82% | 0.73 | 0% (1 unparsed) |
| **phi4 + gemma3:12b** (unanimous) | **2%** (1/55) | 7% (2/28) | 75% | 0.64 | 13% |

Share of perturbed items not waved through (phi4 / gemma3:12b / panel):
negation 100/100/100, number 100/**50**/100, conjunction 100/83/100,
overclaim 96/92/100, swap_passage 100/100/100, **plan→result 67/50/83** (n=6),
attribution swap 100 across (n=1); originals judged supported 93/100/93.
This split holds no regression items, so the real plan→result case was not
tested here. gemma3:12b is weaker on both corpora (15% and 34% FA), so its
pilot showing was not just corpus difficulty. One gemma3 call got HTTP 500
from Ollama; it was recorded as no vote and the run continued (it will be
re-judged on the next run). Report copies: `eval/reports/judge_report_stageB1.md`,
`judge_results_stageB1.json` (local).

**System evaluation under the new default** (generator qwen2.5, judge phi4,
prompt v3; 2026-10-08):

| | Dissertation, old default (10-03) | Dissertation, new (10-08 night) | Dissertation, new, rerun with per-claim detail (10-08 15:31, 3 threads) | Pilot, old (10-05) | Pilot, new (10-08) |
|---|---|---|---|---|---|
| Strict / lenient citation precision | 96% / 96% (26/27) | 83% / 97% (25/30) | **84% / 97% (26/31)** | 90% / 90% | 90% / **95%** |
| Claim / answer hallucination | 0% / 0% | 3% / 4% | 3% / 4% | 10% / 11% | **5% / 5%** |
| Must-refuse traps refused | 95% | 95% | (traps not rerun) | 100% | 100% |
| False-premise traps safe | 83% | **100%** | — | 67% | **83%** |
| Over-refusal | 0% | 0% | 0% | 5% | 5% |
| Parse failures / misplaced refusals | 1 / 3 | 0 / 12 | 0 / — | 0 / 12 | 0 / 13 |

No false claim was shown as supported in any run. The dissertation's strict
precision drop has the same shape in both new runs (4 partial + 1 unsupported),
so it is systematic. **Audit of the 5 non-supported claims in the
per-claim rerun** (done locally against the cited passages; content not
recorded here):

- 2 claims: **phi4 right** — the generator mischaracterised what the passage
  describes (a real error the previous judges would likely have accepted).
- 1 claim: phi4 reasonable on what it saw, but the claim is true in the
  document — the cited chunk does not name the analysis it belongs to; that
  context is in the section heading / neighbouring chunk. Fix: show the
  section heading with the passage to the judge.
- 1 claim: **phi4 too strict** — the claim uses a narrower but consistent term
  than the passage. Left as is (loosening the prompt risks letting overclaims
  through).
- 1 claim: rejected by the **quote check**, not phi4 — the quote ran across
  a chunk boundary: its first sentences are in the cited chunk, the rest in the
  next chunk of the same section (a paragraph split between chunks), and the
  claim cited only the first. Fix: accept a quote whose every sentence is in
  some retrieved chunk, and judge against all chunks it spans.

With the two fixes, strict precision on this run would be 28/31 (90%).

**Confirmation rerun with both fixes** (2026-10-08 22:37 – 10-09 00:18,
3 threads, 26 answerable questions; the generator produced the **same 31
claims** as the per-claim run, so the difference is the fixes plus judge
variance): strict / lenient precision **97% / 100% (30/31)**, 0 unsupported,
0 fabricated quotes, 0 over-refusal, 0 parse failures. The spanning quote was
accepted (recorded as "quote spans chunks …"); the claim missing section
context and the one phi4 had judged too strict both became supported. Of the
two real generator errors, the more serious one is still flagged partial; the
milder one (same imprecise wording, core facts right) is now **supported** —
judge variance or the heading making phi4 more lenient; unresolved (§7 step 1).
Reports: `eval_report_phi4v3_fix.md`, `eval_results_phi4v3_fix.json` (local).

**Does the section heading make phi4 more lenient? No** (2026-10-09
09:15–10:55, `judge-eval --with-heading`, dissertation held-out split, 83 items,
3 threads). False-accept 5% (3/55) and false-reject 7% (2/28), identical to the
bare run; accuracy 82% vs 81%, κ 0.73 vs 0.71. Every error type caught at the
same rate. Only 3/83 verdicts changed, all between partially_supported and
unsupported (both count as caught), in both directions; no item crossed
accept/reject. So both audit fixes stay, and the milder generator error that
passed in the confirmation rerun is most likely judge variance on a borderline
claim. Thread count (3 vs all cores) did not change verdicts either. Reports:
`judge_report_stageB1_heading.md`, `judge_results_stageB1_heading.json` (local).
Per-claim detail (claim, quote, cited passage, verdict, votes, reason) is now
saved in `eval_results.json` (local; contains corpus text). Reports:
`eval/reports/eval_report_phi4v3.md`, `eval_report_phi4v3_detail.md`,
`eval_results_phi4v3_detail.json`, and the pilot's `eval_report_phi4v3.md`.

**Plan-as-result rule** (`verify.plan_as_result`, `PLAN_RESULT_RULE`, default
`flag`; 2026-10-09, evaluated offline, no model run). Fires when the claim
asserts a finding (zh/en result cues), carries no hedge (將/預計/假設/will …),
and the source plans or predicts — the quote's sentences, or without a quote a
passage sentence that plans an assessment or predicts an outcome. Then an
accepted verdict becomes `disputed` (`reject` → unsupported).

| Check | Result |
|---|---|
| Real regression case REG-T19 (all judges approved it) | **flagged** |
| Synthetic plan→result, dissertation | 11/11 (dev 5/5, test 6/6) — circular: built with the same cues |
| True validation claims (originals, both corpora) | **0/100 flagged** — weak test, originals with "will" are exempt via the hedge |
| Real system claims (two per-claim runs) | **0/31, 0/31 flagged** |
| Pilot REG-T14 (misattribution, not plan→result) | not flagged, as expected |
| phi4 false-accept, dissertation held-out, + rule | **3/55 → 1/55**, true claims flagged 0/28 (same with and without heading); the remaining miss is an overclaim |

**Hard set** (2026-10-10, `data/golden/judge_set_hard.jsonl`, local,
`judge-eval --split hard`): 26 hand-written items over dissertation passages —
12 plan→result restated as findings in natural Chinese (5 deliberately without
the rule's result cues), 6 plans kept as plans, 8 real results paraphrased in
Chinese; gold relative to the cited passage. Rule offline: **7/12 caught — all
7 with cue words, none of the 5 without** — and 0/14 controls flagged. The
rule's recall depends on the claim's wording; it was deliberately not tuned on
this set (the only real test of it).

**phi4 on the hard set** (2026-10-10 04:20–05:10, v3 + heading, 3 threads,
`judge_report_hard.md`, local): on realistic plan→result items **phi4 is
weak**.

| | Caught (not accepted as supported), 12 plan items | False rejections, 14 controls |
|---|---|---|
| phi4 alone | 33% [14–61] (4/12) — all four only `partially_supported`, none `unsupported` | 14% [4–40] (2/14) — both hedged-plan controls |
| rule alone | 58% [32–81] (7/12) | 0/14 |
| phi4 + rule | 67% [39–86] (8/12) | 14% (2/14) |

Of the 5 cue-free items the combination caught 1. The 4 still accepted all
restate a sentence from a **Hypotheses section** as a finding, and phi4
approved them although the heading it was shown says "Hypotheses". It judged
all 8 real-result paraphrases correctly but rejected two correctly hedged
plans — it appears to match content, not study stage. So the v3
`plan_as_result` check barely works for Chinese claims over English
hypothesis passages, and the rule and the judge miss the same items.

**Structural rule added (2026-10-10, no model run):** an unhedged claim whose
cited chunk's *innermost* heading is a Hypotheses section ("x.y.3 Hypotheses";
4 chunks in the dissertation, none in the pilot) is flagged whatever its
wording; "會" joined the hedge words. Matching the full heading path first
false-flagged 2 background / research-question sentences, because the chapter
title also says "Hypotheses" — hence innermost only. Results: hard set
phi4 + rule false-accept 8/12 → **0/12** (not independent: the rule was
designed after seeing these items); held-out unchanged at 1/55 FA, 2/28 FR;
false flags on independent data **0/100** true validation claims and **0/87**
real system claims. No unseen positives remain (all 4 hypothesis chunks are
in the hard set), so recall must be checked on real generator outputs — trap
questions that tempt citing a hypotheses section, in the generator-prompt
rerun.

Planned response, in order: (1) a structural rule — an unhedged claim citing
a chunk from a Hypotheses section is flagged regardless of wording (designed
after seeing this set, so it needs separate validation: false flags on real
outputs, and a fresh hard set); (2) fold "do not state hypotheses or methods as
findings" plus passage-type labels into the generator prompt work (§7);
(3) a v4 judge prompt that makes the judge name the supporting sentence's
type (result / hypothesis / plan / background) before its verdict, if (1)
and (2) are not enough.

One real positive only, so recall is unknown; the residual risk is a true
finding cited from a future-tense methods passage, hence review (`disputed`)
rather than rejection by default. Unit tests: `tests/test_verify_rules.py`
(19 tests, generic texts, also cover quote spans and headings).

**Traps (dissertation) and full pilot run after the fixes** (2026-10-09
14:15–17:16, 3 threads, `eval_results_traps_postfix.json`,
`nature_walking_2026/eval_results_postfix.json`, local). No false claim shown
as supported in either corpus.

| | Dissertation, before → after | Pilot, before → after |
|---|---|---|
| far-absent refused | 100% → 100% | 100% → 100% |
| near-absent refused | 89% → 89% | 100% → 100% |
| false-premise safe | 100% → 67% (4/6) | 83% → 67% (4/6) |
| Pilot answerable strict / lenient | — | 90% / 95% → 89% / 95% (17/19, 18/19) |

The false-premise drop, item by item: two (one per corpus) were **phi4
timeouts** — the judge hit the 300 s floor reading a long passage on 3
threads, and the lexical fallback, which cannot read Chinese, rejected the
claim ("no content tokens"); one generator claim accepted the false premise and
phi4 rejected it; one quote was **translated into Chinese**, so the quote check
called it fabricated (known format drift). The real plan→result trap was
rejected by phi4 itself this time (reason: plan read as result), so the rule
had nothing to do; the rule flagged 0 of all trap and pilot claims. Pilot
non-supported answerable claims: one result not in the cited passage (phi4
right), one added inference (phi4 reasonable). No quote spanned chunks.

**Fixes after this run** (no model run; tests in `tests/test_judge_failure.py`):
- `llm.request_timeout`: the timeout now covers reading the prompt as well as
  writing the output — max(300 s, 60 s + (output limit + ½ × prompt chars /
  2.5) × `OLLAMA_SECONDS_PER_TOKEN`); a judge on a ~3,500-character prompt at
  0.8 now gets ~860 s instead of 300 s.
- A claim no configured judge could vote on (timeout, error, unparsable) is now
  `disputed` ("judge unavailable; needs review") instead of going to the
  lexical fallback, which is kept only for runs with no judge at all. It still
  counts as not safe for false-premise traps (conservative); `n_unjudged` in
  the eval results shows how many claims this was.

**Timeout fix confirmed** (2026-10-09 22:50–22:57): phi4 re-judged the two
saved claims whose judge had timed out, with identical inputs (same claim,
quote, full passage + heading; no regeneration), 3 threads, running alongside
another project's Ollama job (peak CPU 76%, RAM 55%). Both finished well under
the new ceiling (~860 s): 279 s and 153 s. The 279 s case shows why the old
300 s floor failed under contention. Verdicts: the dissertation claim
`supported` (its fact matches the passage; the wording is muddled — a
generator writing issue, judged leniently but not a hallucination); the pilot
claim `partially_supported` (phi4 caught a wrong study duration). Re-scored
with these verdicts, false-premise safe would be 5/6 in both corpora (an
estimate; partial counts as safe under the trap rule).

## 5. Known weaknesses and open findings

1. **Plan read as result — mitigated by a rule (2026-10-09, §4).** The
   generator restated a planned assessment (future tense) as a reported
   finding; all judges approved it under v2 and v3. The rule check now flags
   that real case and sends it to review. Still open: recall on real outputs
   (one known positive), and harder plan→result items shaped like real outputs
   (paraphrased, Chinese claim / English passage).
2. **Cited study attributed to this study — partly addressed.** v3 helps on
   "Author et al. found → This study found" swaps, but the pilot paper's
   background-statement swaps stay at ≤ 40% detection.
3. **Self-judging — resolved in the default (2026-10-07).** qwen2.5 was both
   generator and judge, had approved its own errors twice, and barely changed
   under v3 (87% false-accept on the new error types). The default judge is now
   **phi4** (dissertation 5% / 7%, pilot 6% / 5%); mistral-nemo is ruled out.
4. **Negative facts refused.** "The paper says X was not done" was answered
   with the refusal marker (over-refusal).
5. **Format drift.** Refusal marker placed inside `<claim>` (3× dissertation,
   12× pilot), translated quotes, occasional unclosed tags. All handled and
   counted; worth reducing in the generator prompt.
6. **Judges are not deterministic.** The same claim/passage can get different
   votes across runs; decide on larger sets or repeated runs, not one replay.
7. **deepseek-r1:8b broke after an Ollama update.** Under Ollama 0.35.1
   (auto-updated 2026-10-06) 95% of its v3 replies did not parse, and each
   judgment took ~80 s instead of ~25 s; under 0.9.3 the night before it
   parsed normally (Stage A). **Diagnosed 2026-10-06** with three raw
   `/api/chat` requests on one pilot test item (v3 prompt): Ollama 0.35 returns
   the thinking in `message.thinking`, and those tokens count toward
   `num_predict`. With the judge's 300 tokens the model thought until the
   limit (`done_reason=length`) and `content` was empty — with and without
   `think: false`, which this model ignores. With 1,500 tokens it stopped on
   its own after 1,174 tokens (234 s) and returned valid JSON. **Fixed in
   `src/llm.py`:** models whose `/api/show` capabilities include `thinking`
   get `OLLAMA_THINK_BUDGET` (default 2000) extra tokens; an empty answer
   after thinking is logged. **Re-validation attempt 2026-10-07 09:28** (5
   pilot test items, no competing load, CPU ~58%): the first judgment was
   still generating at the fixed 300 s client timeout and the uncaught
   `TimeoutError` aborted the run. Real cost is therefore 4–8 min per
   judgment on CPU (pilot 55 items ≈ 4–7 h, dissertation 83 ≈ 6–11 h).
   **Decision (owner, 2026-10-07): drop deepseek-r1 as a judge**; phi4 is
   faster (~38 s) and already at 6% / 5% on the pilot split. Follow-up fixes
   made anyway: the Ollama timeout now scales with the token limit
   (`OLLAMA_SECONDS_PER_TOKEN`), a judge that times out or errors casts no vote
   instead of aborting the run, and `judge_eval.py` no longer treats cached
   `unparsed` votes as cache hits (they are re-judged on the next run).
   An earlier 4b attempt (2026-10-06 19:29) was stopped by the guard because a
   `line_chat` labelling job started on `qwen3:8b` 25 s later; the guard's
   unload of *all* models also interrupted that job (see §8).

## 6. v3 validation status

**Stage A — done (2026-10-05, 21:51).** v2 vs v3 on the new error types,
judges qwen2.5 / gemma3:12b / deepseek-r1:8b, both corpora
(`--types plan_to_result attribution_swap`, both splits; v3 was written once
from the observed failures and not tuned on these items). Results in §4;
reports saved as `eval/reports/judge_report_stageA.md` and
`eval/reports/nature_walking_2026/judge_report_stageA.md` (local). Votes are
cached in each corpus's `judge_votes.jsonl`, so later stages reuse them.

To rerun (skips cached votes):

```powershell
$env:PYTHONIOENCODING="utf-8"
python -u -m eval.resource_guard --log-every-s 300 -- cmd /c eval\reports\stage_a.cmd
```

`stage_a.cmd` (local) runs, for `CORPUS=dissertation` and then
`nature_walking_2026`:

```
python -u cli.py judge-eval --judges ollama:qwen2.5:7b-instruct ollama:gemma3:12b ollama:deepseek-r1:8b --prompts v2 v3 --split all --types plan_to_result attribution_swap
```

**Stage B — v3 on the held-out split, all item types (measures false-reject).**
Run step by step, each approved by the owner after a plan and resource
assessment (see §8):

| Step | Scope | Status |
|---|---|---|
| 0. Calibration | phi4, mistral-nemo, 3 items each | Done. Per judgment on a loaded model: phi4 ~38 s, mistral-nemo ~21 s (first call incl. load: 111 s / 62 s). Peak CPU 61%, RAM 40% |
| 1. Dissertation | 83 items (28 true), phi4 / gemma3:12b | Done 2026-10-07, 17:08–19:05 (phi4 ~50 s per item, gemma3 ~30 s). Avg CPU ~55%, peak 100%; RAM peak 56%; guard never fired. Results in §4 |
| 2. Pilot paper | 55 items (20 true), gemma3:12b / deepseek-r1:8b / phi4 / mistral-nemo | Done 2026-10-06, 12:00–14:51. Peak CPU 85%, RAM 64%; guard never fired. Results in §4 |

Step 2 ran from `eval/reports/stage_b2.cmd` (local) under the guard; report
copies `eval/reports/nature_walking_2026/judge_report_stageB2.md` and
`judge_results_stageB2.json`. Command:

```
CORPUS=nature_walking_2026 python -u cli.py judge-eval --judges ollama:gemma3:12b ollama:deepseek-r1:8b ollama:phi4:latest ollama:mistral-nemo:latest --prompts v3 --split test
```

**Ollama contention (fixed 2026-10-06, see §8).** Stage A was paused mid-run
because another local project (`D:\side_project\line_chat`, a labelling job on
`qwen3:8b`) shared the Ollama server, and Ollama kept only one model loaded:
the two jobs evicted each other's model on every request (~2.5 min per
judgment instead of ~8 s). A detached waiter resumed Stage A once that job
exited. Ollama now keeps up to three models loaded, so this should not recur.

## 7. Next steps

Every step that runs local models needs the owner's approval of a plan and
resource assessment first (§8). Completed steps, with their full numbers and
audits, are in §10 (newest work first under items 0000 → 0, then the older
numbered items).

**State (2026-10-16).** Defaults: generator qwen2.5:7b-instruct with prompt v1;
judge phi4 alone, prompt v3, section heading shown; plan-as-result rule
`flag` with the structural, wording and hypothesis-match routes; quotes may
span chunks. The grounded vs baseline comparison is finished and reported in
README (strict bound = main table, lenient bound = sensitivity table): the
grounded arm refuses what the source does not say far better (25 traps: 88%
vs 24% correct, p < 0.001); on answerable questions the two arms are close
once the judge sees every passage; hypothesis bait (revised set) 3/6 vs 0/6
safe — the grounded generator still restates a hypothesis on half the items,
and verification catches all three.

Open, in suggested order:

1. **Machine stability (owner).** The blue screens came from WSL/Docker
   GPU paravirtualisation (§8), now disabled in `.wslconfig`; watch for
   further resets, update the NVIDIA driver and WSL, then install the
   pending Ollama update and check a few items still behave the same.
2. ~~Larger hypothesis-bait set~~ — done 2026-10-16. 6 new items (local
   golden set, backup `golden_set.before-bait-expand.json`): B09–B12 planned
   analyses or measures with no reported result, B13–B14 reworded questions
   on the two unreported hypotheses (same hypotheses as B07/B08, so only a
   partly independent check of the hypothesis-match route). A first run on
   Ollama 0.35.1 was cut by the 06:58 reset; both arms were rerun on 0.40.5
   (10:27–12:53, 4 threads, peak CPU 79%, RAM 53%, no reset; local
   `eval_results_bait4_*.json`). Grounded: safe 8/12 (identical item by item
   to the 0.35.1 run); refused 8 including all of B09–B13; restated the
   hypothesis on B02 (judge rejected), B07 (hypothesis-match route), B08 and
   B14 (structural route). Baseline as first scored: 0/12, but Chinese
   abstentions with words between negation and verb ("並未在文獻中明確提及",
   "尚未…得到詳細報告") and Chinese advice sentences ("需要通過實驗數據來確認")
   had been judged as claims → `REFUSAL_HINT_RE` / `META_RE` gained Chinese
   patterns, re-scored without model calls (`--reuse-from f678809e7c141f94`,
   local `eval_results_bait4_baseline_rescored.json`): safe 5/12, strict 4/12.
   Paired: safe 8 vs 5 (p = 0.25), strict 8 vs 4 (p = 0.125), all discordant
   items in the grounded arm's favour. Manual reading: baseline B11's
   remaining sentence is a hedged guess about an unreported measure that phi4
   rated supported, so a manual count is 4/12. A sentence holding both a
   denial and a claim is treated as a denial (known leniency). Hypothesis-
   match route fired 3 times (grounded B07, baseline B02 and B14), all
   correct. The thesis has no further unreported hypotheses, so a larger
   bait set needs another corpus.
3. **R06 label (owner).** In the sensitivity run phi4's reason reads as
   support but its label is unsupported; check which is right.
4. ~~`check` cost~~ — done 2026-10-17. It judged the top 3 passages (not 5
   as noted earlier). `eval.check_cost` (40 dev-split synthetic statements,
   20 true / 20 perturbed; 120 judged pairs, 09:16–11:40, phi4 4 threads,
   ~72 s per judgment) simulated 1–3 passages in retrieval or similarity
   order: top 2 keeps false accepts at 2/20, false rejects 0 → 1/20 (S023,
   evidence ranked third), 146 → 111 s per sentence (unsupported sentences
   203 → 137 s); top 1 has 3/20 false rejects. Similarity order identical
   to retrieval order on all 40. `CHECK_TOP_N` now defaults to 2. The two
   apparent false accepts (P204, P469, plan_to_result) are gold-label
   artefacts for this use: their gold label is relative to the hypothesis
   passage they were built from, but the thesis reports both hypotheses as
   results (results / discussion passages, which `check` found and phi4
   accepted). Relabelled: false accepts 0/18 for 1–3 passages, false
   rejects 0/22 (3), 1/22 (2), 3/22 (1). The rule was left as it is: flagging
   whenever any passage reads as a plan would flag true results whose plan
   also appears in the methods. Local results: `check_cost_results.json`.
5. ~~Generator prompt v2.1~~ — **default since 2026-10-19** (owner's
   decision). Built 2026-10-18 (`36aab0a`): v2 minus rules 8–9, `<refusal>`
   tag, two-example template with fake ids c_section_9999/9998. Calibration
   1 (5 answerable + 5 traps + 3 bait): no format errors but 4/7 claims
   without a quote; a raw-output diagnostic (`--ids`, grounded records now
   keep raw XML) showed a quote after `</claim>` (R04) and English claims
   that copied the source sentence with no quote (R01, R05) → parser
   attaches a trailing quote, the verifier checks an unquoted claim's own
   text as its quote, the prompt asks for a quote even then (`812fb96`).
   Calibration 2: 0/7 missing. Full paired run 2026-10-18 21:33–02:21
   (`--fresh`, Ollama 0.40.5, 4 threads; local `eval_results_gen_v21.json`,
   `eval_results_gen_v1.json`): v2.1 vs v1 strict 25/28 vs 29/30, lenient
   28/28 vs 29/30, unsupported 0 vs 1, misplaced refusals 0 vs 16, missing
   quotes 1 vs 5, traps strict safe 36/37 vs 30/37 (McNemar p = 0.031; v2.1
   better on T19, T25, B02, B07, B14, never worse), must-refuse 19/19 vs
   18/19, generator restated a hypothesis on bait 1 vs 4, 126 vs 148 s per
   question. Criterion "strict precision not lower by more than one claim"
   was missed by one claim: R24 was a bare number and R07 a strict judge
   reading. Full-sentence rule added (`bc78857`, generic example — an
   example built from the thesis was caught before commit), calibration 3
   (R01–R05, R07, R24, T01–T05, B01–B03): R24 now a full supported sentence,
   R07 still partial, 0 format errors, 2/9 claims without a quote (both
   paraphrases), all traps refused.
6a. ~~Out-of-corpus check~~ — done 2026-10-17 (README "Out-of-corpus
   check"). Corpora `jmir_protocol_2026` (CC BY 4.0 protocol, no outcome
   data) and `frvir_restorative_walk_2020` (CC BY, H1–H3c all reported);
   PDFs in `papers/`, question sets in `data/corpora/<name>/golden/`
   (local). PDF ingestion fixed on the way (`82f91fb`: whitespace spans,
   body starting at Introduction); the pilot corpus's index was not
   rebuilt. Retrieval Hit@5: protocol 100%, completed study 67% (Chinese
   control questions retrieved poorly with the English embedder, so the
   controls were asked in English; two still retrieve hypothesis +
   discussion passages, which also report the results). Run 17:34–20:32,
   4 threads, peak CPU 83%, RAM 52%. Protocol bait: safe grounded 5/6 vs
   baseline 3/6; grounded invented no result (its unsafe J01 is a denial
   inside a claim with a translated quote), baseline invented results on
   J01, J02, J05, all rejected or disputed. Protocol answerable: grounded
   7/7 supported, baseline 10/14 strict. Completed study: grounded 6/6,
   baseline 13/19 strict with 3 unsupported (2 true sentences judged
   against the wrong 2 passages, 1 invented number); no rule fired on any
   true result. Neither paper has a Hypotheses heading, so only the wording
   route was exercised. Pooled bait (thesis 12 + protocol 6): 13/18 vs 8/18,
   McNemar p = 0.063.
   Rerun with generator v2.1, grounded only (2026-10-19 06:16–07:12, `--fresh`,
   local `eval_results_grounded_v21.json`): protocol bait 6/6 safe, 5 refused
   (v1: 5/6, 3), J01 now a clean refusal; misplaced refusals 0 (v1: 2),
   translated quotes 0 (v1: 1); protocol answerable 4/4 supported but shorter
   (R02 one claim vs four) and 2 without a quote (R01: dropped opening
   `<quote>` tag → parser fix `bc0cba0`; R02: no quote); completed study
   strict 4/6 vs 6/6 (R06 adds a detail the source lacks; R02 copies the
   confirmed H2 sentence citing the hypothesis passage), lenient 6/6, no rule
   fired on a true result. R01 and R02 of the protocol appended a `<refusal>`
   to a complete answer → rule 7 tightened (`6f8ec09`). Calibration 4
   (2026-10-19 09:56–11:05; thesis R01–R05, T01–T03, B01–B02; protocol R01,
   R02, J01, J03; completed study R01–R06): no `<refusal>` after a complete
   answer, every trap and bait item refused (J03 answered with a supported
   plan statement), no format errors, 3/19 claims without a quote. One
   unsupported claim: completed-study R01 answered a result question with
   "Hypothesis H1: …" (correctly labelled) citing a passage that does not
   hold it — v2.1 tends to answer from a retrieved hypothesis passage
   (also R02 in the previous run). Tried and withdrawn 2026-10-19
   (`a9beec7`, reverted): rule 5 told the generator to answer result
   questions from results/discussion passages and refuse when only
   hypothesis passages exist. Calibration 5 (11:51–12:51; thesis R01–R05,
   T01–T03, B01, B02, B07; protocol R01, J01, J03; completed study R01–R06):
   completed-study R01 still answered with H1 (unsupported), R02 now cited
   the results passage, thesis B07 regressed to the hypothesised regions
   (disputed by the hypothesis-match route, so not shown as supported), and
   a `<refusal>` after a complete answer reappeared once (R03). Mixed effect,
   so the sentence was removed; the prompt is back to `6f8ec09`.
   Context order tried next (2026-10-20, `8da43f8`, `CONTEXT_ORDER`, default
   stays `rank`): `results_first` calibration 15:55–16:57 on the calibration-4
   set + B07 (local `eval_results_order_cal.json` per corpus), compared with
   calibration 4 (same prompt and code, rank order). 13 of 21 answers
   identical, all traps and bait refused as before, no format errors, thesis
   R02/R04 and completed-study R06 quotes slightly closer. Completed-study
   R01 went unsupported → supported and R04 supported → partial, but both
   still copy the hypothesis sentences (H1, H3b): the generator cited a
   results passage id, the quote check re-attributed the quote to the
   introduction, and phi4 accepted the confirmed hypothesis. Thesis B07 still
   restated the hypothesised regions (disputed by the hypothesis-match
   route). Not adopted; step 3 (full paired run) not run. Two findings: (1)
   the completed-study control questions paraphrase H1–H3c, so retrieval and
   generator gravitate to the hypothesis sentence — a question-design
   artefact; (2) hypotheses written in an introduction without a Hypotheses
   heading are invisible to the structural and hypothesis-match routes.
   Follow-ups: control questions not worded like the hypotheses (not
   started); inline hypotheses — done 2026-10-21: `verify.inline_hypotheses`
   extracts the sentence after "Hypothesis N (HN):" / "HN:" / "假設N："
   (completed study 5, thesis 12 — all already in Hypotheses sections —
   protocol 0, pilot 0); a third plan-as-result route flags an unhedged claim
   whose quote (or unquoted text) restates one in a cited passage; inline
   hypotheses also join the hypothesis-match pool and are dropped from the
   cited-sentence side. Offline over every saved output of the four corpora:
   4 supported/partial claims cite a passage with inline hypotheses (2
   unique: completed-study R01 and R04 of the results-first calibration) and
   the route fires on all of them, nothing else. Pooled bait with v2.1:
   17/18.
6b. **Chinese questions over English sources** — measured 2026-10-21, default
   not changed yet. Chinese versions of all 56 retrieval questions
   (`question_zh` in the local golden sets; backup
   `golden_set.before-zh.json`), `eval.retrieval_lang`, bge-m3 indexes built
   for the section strategy of all four corpora (the pilot corpus was also
   rebuilt with bge-small after the PDF spacing fix). Pooled Hit@5 (Hit@1):
   bge-small hybrid en 53/56 (43), zh 29/56 (14); bge-m3 hybrid en 52 (46),
   zh 41 (19); bge-m3 dense-only zh 49 (40). BM25 hurts cross-lingual
   queries, so `CJK_QUERY_DENSE_ONLY=auto` skips it for a CJK query when the
   embedder is multilingual; embedder: bge-m3 takes no query instruction.
   **Default since 2026-10-21 (owner's decision).** bge-m3 fixed-strategy
   indexes built for all four corpora; chunking ablation with bge-m3:
   thesis fixed Hit@5 88% → 100% (section Hit@1 88%, MRR 0.929). Full thesis
   run, both embedders, current code, `--fresh` (07:22–11:33; local
   `eval_results_emb_{small,m3}.json`): five claims had been rejected as
   fabricated citations by a parser bug from `bc0cba0` (a stray `</quote>`
   after a paraphrased claim was taken as the quote) → fixed in `0405f9d`
   and re-judged from the stored raw output (`reverify_emb_*.json`; R10, R26
   small; R03, R04, R05 m3). Corrected: strict 26/28 vs 27/28, answers clean
   24 vs 25, traps strict safe 34 vs 35, bait safe 10 vs 11, 127 vs 111 s per
   question; one genuine unsupported claim each (small R26; m3 R18, a quote
   not in the source). Earlier runs since `bc0cba0` had no claim rejected
   through that bug. The pilot corpus's bge-small fixed-strategy index
   predates the PDF spacing fix (its chunks file was rewritten when the
   bge-m3 fixed index was built); rebuild it before using bge-small there.
6c. **Independent check of the hypothesis routes — done 2026-10-22.**
   (i) Completed-study controls rewritten to ask for measured outcomes
   (local `golden_set.json`, old set `golden_set.before-rewrite.json`; 7
   items, R07 the exploratory distance result). With bge-m3 the old R01/R04
   had retrieved only hypothesis and introduction passages (no results
   passage in the top 5); the new set has no hypothesis passage in any top 5
   (Hit@5 7/7). (ii) New corpus `frvir_vr_hrv_2024` (Pratviel, Bouny &
   Deschodt-Arsac 2024, CC BY; `papers/frvir_vr_hrv_2024.pdf`, 2.2 MB):
   numbered "1.5 Hypotheses" heading; PDF reader fixed for the 2024
   Frontiers layout (`08e820a`; the other PDFs parse identically). Local set:
   R01–R07 (R01–R03 worded like H1A/H1B/H2), T01–T03 false premise, T04–T06
   hypothesis bait, T07–T08 near-absent, T09 far-absent; Hit@5 7/7 (R01's
   results passage is not retrieved; the Discussion states it). Run
   2026-10-21 21:54 – 10-22 02:58 (4 threads; local
   `eval_results_hypval_{grounded,baseline}.json` in both report dirs).
   Completed study: grounded 7 answers all from results/methods passages,
   strict 7/8 (R06 figure-caption detail); baseline 17/18. Pratviel:
   grounded answers 7/7 clean, strict 9/9, traps 9/9 correct and strict
   safe, no rule fired; baseline 3/7, 18/26, traps 3/9 (strict 2/9). Rule
   behaviour in the baseline: hypothesis-match caught T03 (false H3 claim);
   structural route disputed a true R03 sentence (H2 supported) because it
   was judged against its two closest passages and the hypothesis came
   first. Fixes (this commit): `verify_statement` applies the plan rule per
   passage before choosing the best label; `_closest` adds the closest
   non-hypothesis passage when a Hypotheses-section passage is among the k
   closest; `REFUSAL_HINT_RE` covers "not possible to determine", "unclear
   from the provided context", "not … in the provided context", "does not
   … break down". Re-scored from the saved answers (scratchpad wrapper
   around `--reuse-from 9f0d6bf91aa376b3` that drops cached verdicts of
   sentences near a Hypotheses section; 08:20–09:05; local
   `eval_results_hypval_baseline_rescored.json`): R03 sentence supported,
   T03 still disputed, R01's true sentence still unsupported (its results
   passage is not retrieved), four abstentions no longer judged → baseline
   4/7, 19/26, traps 4/9 (strict 2/9). McNemar on traps: strict 7–0 (p =
   0.016), correct 5–0 (p = 0.063). Not shown by the metrics: grounded R02
   conflated two statistics and phi4 accepted it; grounded R01/R04 are true
   but do not answer the question (answer completeness is not measured).
   **Thesis baseline re-scored (owner approved; 2026-10-22 09:19–12:37,
   same wrapper, phi4, 4 threads):** main run `--reuse-from
   9ade0f6e045ad0b2 --ids` R01–R26, T01–T25, B01–B03, B06 (B04/B05 were
   replaced since, so not reused); bait run `--reuse-from bff0049629e4185d
   --traps-only --trap-types hypothesis_bait`; 43 sentences re-judged (local
   `eval_results_baseline_rescored2.json`,
   `eval_results_bait4_baseline_rescored2.json`). Outcome: one answerable
   sentence R19 partial → supported (strict 85 → 86 of 94, answers with a
   non-supported claim 7 → 6 of 26; grounded's one such answer is R21, so
   paired 5–0, p 0.031 → 0.063); traps 6/25 correct and 9/25 strict safe,
   bait 5/12 safe and 4/12 strict safe, unchanged item by item. Other label
   moves (e.g. unsupported → disputed on bait sentences) are judge variance
   plus the per-passage rule and change no item outcome. The all-passage
   sensitivity re-judge (2026-10-15) had already upgraded R19, so its
   lenient bound stands. The grounded arm is not affected.
6d. **Chinese questions end to end — done 2026-10-22 (owner approved, plan
   A).** `eval --lang zh` swaps each item's question for `question_zh` (the
   English one kept as `question_en`; items without one keep their
   question); traps were translated locally (thesis T01–T25, Pratviel
   T01–T09; backups `golden_set.before-zh-traps.json`). Grounded arm only,
   `--fresh`, 15:56–19:20, 4 threads (local `eval_results_zh_grounded.json`
   per report dir); compared with the 2026-10-21 English runs
   (`eval_results_emb_m3.json` + `reverify_emb_m3.json` for the thesis,
   `eval_results_hypval_grounded.json` for the two OA papers). Thesis: clean
   answers 25 → 23 (R16, R26 one partial claim each; R21 refused although
   its passage ranked first; R18 improved), strict 27/28 → 27/29, traps and
   bait identical item by item (36/37 correct, 35/37 strict safe), 0
   translated quotes, 128 s per question. Pratviel: clean 7 → 3, strict 9/9
   → 4/8, traps 9/9. Rockstroh: clean 6 → 5, strict 7/8 → 6/8. Pooled
   discordant 9 vs 2 (p ≈ 0.065). Failure pattern: on yes/no questions the
   generator echoes the Chinese question as an affirmative claim (Pratviel
   R01 true claim with an irrelevant quote → unsupported; R02 false claim
   "the conditions differed" → only partial); phi4, reasoning in Chinese,
   rejected a true claim with a correct quote (Pratviel R04); Rockstroh R07
   paraphrased its quote (fabricated-citation rejection of a true claim).
   Possible follow-ups (not started): a generator rule against restating a
   yes/no question as the claim, and judging Chinese claims with an English
   reasoning instruction.
6. **Human-labelled judge items** (`data/golden/judge_set_human.jsonl`,
   schema in `judge_set_human.example.jsonl`) — owner's task, not started.

## 8. Operational notes

- **Machine stability (cause found for the blue screens; mitigated
  2026-10-16):** five unexpected shutdowns in 60 days (Kernel-Power 41):
  2026-09-17 17:50 and 2026-10-12 23:54 with bugcheck 0x7E, and 2026-10-05
  12:52, 2026-10-13 ~01:13 and 2026-10-15 ~22:52 with bugcheck 0 (reset
  without a dump). WinDbg `!analyze -v` on the 2026-10-12 kernel dump:
  `AV_dxgmms2!VIDMM_GLOBAL::CreateOneAllocation` — a null-pointer read
  (address 0x15b) in `dxgmms2.sys` 10.0.19041.6456, Windows' DirectX video
  memory manager, while creating GPU page tables, with process `vmmem`, i.e.
  the WSL2 / Docker Desktop VM using the host GPU through GPU
  paravirtualisation (NVIDIA Quadro P620). The 2026-09-17 crash faulted at
  the same offset, so it is the same bug. Not this project (Ollama runs on
  Windows, CPU-only, not in WSL), not the Epson indirect-display driver (not
  on the stack), and a repeated fault site argues against bad RAM.
  Mitigation: `%USERPROFILE%\.wslconfig` with `[wsl2] gpuSupport=false`
  (owner's decision; both WSL distros, Ubuntu-24.04 and docker-desktop, were
  stopped, so it applies from their next start; set it back to true and run
  `wsl --shutdown` if a project needs the GPU inside WSL). Still advised:
  update the NVIDIA driver and `wsl --update`. The three dump-less resets
  are unexplained — if one recurs, note whether WSL/Docker was running; a
  memory test (`mdsched.exe`) remains useful. **A sixth reset (bugcheck 0, no
  dump) hit 2026-10-16 ~06:58, after the WSL GPU change, during this
  project's baseline run (CPU ~50%, RAM ~50%, swap flat, guard silent).** So
  the dump-less resets are not the WSL GPU bug; power, cooling or other
  hardware is suspected. Local runs are paused until the owner has checked
  CPU temperature under load, the power path (PSU, cables) and memory. Other read-only checks: no
  WHEA hardware-error events in 30 days, all four disks Healthy, BIOS 308
  (2019-03; ASUSPRO D840MB). Every eval run saves per question, so a reset
  loses at most the question in progress.
- **Ollama 0.40.5 (installed 2026-10-16 ~10:06, unplanned).** After the
  reset, an `ollama ps` status check started the Ollama app, which installed
  the pending update (0.35.1 → 0.40.5) and started the server without the
  CPU-only script (`library=CUDA`). Ollama was restarted through
  `start_ollama_cpu.cmd` (log: `CUDA_VISIBLE_DEVICES:-1`, `OLLAMA_VULKAN:false`,
  `library=cpu`); the startup shortcut still points at the script. Check
  Ollama's state with the HTTP API (`/api/version`, `/api/ps`), never with the
  `ollama` CLI while the server may be down. All results so far were produced
  on 0.35.1; later runs use 0.40.5, so compare within one version.
- **Owner's rule for local model runs:** before any test that runs models,
  measure current RAM / CPU / GPU, check that no other program is using
  Ollama, and submit an execution plan with a resource assessment (expected
  peak load, duration, guard thresholds); run only after the owner approves,
  step by step. A cheap calibration (a few items per new model) first is the
  accepted way to replace guessed timings with measured ones.
- **Shared Ollama server:** `D:\side_project\line_chat` runs long `src.community
  label/critique` jobs (qwen3:8b, phi4) started from other sessions (its
  `annotate --queue` process stays up and does not block),
  `D:\2026_manuscripts\Study1\scripts\32_vlm_triage.py` uses `gemma3:4b`, and
  `Alpha Machine\tg_bot.py` may call `llama3`. Check for them (processes and
  established connections to port 11434) right before launching, not only at
  planning time; Ollama holds at most three models, so a third project's model
  can start evictions. The guard now unloads only the models passed with
  `--models` (fixed 2026-10-07; it used to unload every loaded model,
  including other projects').
- **Limiting this project's CPU use:** `OLLAMA_NUM_THREAD=3` sends `num_thread`
  with each Ollama request (other projects on the shared server are
  unaffected); measured: the runner then uses exactly 3.0 cores, total CPU
  ~40%, and an answerable question takes ~110–400 s instead of ~95–280 s. Raise
  `OLLAMA_SECONDS_PER_TOKEN` (0.8 was used) so slower generation is not cut
  off. Other sessions start jobs at any time (`line_rumor_bot` ran a retrieval
  experiment mid-run on 2026-10-08 and the guard stopped our full-core run), so
  a capped run is the safer default when the machine is shared.
- **Long local runs:** launch detached (PowerShell `Start-Process`) under
  `eval.resource_guard`; tool-managed background tasks stop at ~30 min.
  Measured load with two judges: CPU ~50–80%, RAM ~45% of 64 GB; with three
  models resident (qwen2.5 + gemma3:12b + deepseek-r1:8b) RAM ~60%; Stage B
  step 2 with four judges peaked at CPU 85%, RAM 64%.
- **Ollama runs CPU-only so several models stay loaded (machine setup).** The
  machine's GPU (Quadro P620, 2 GB) holds only one model's compute buffer, so
  with the GPU visible Ollama evicted every other model on each switch —
  `OLLAMA_MAX_LOADED_MODELS` alone does not help (it already defaults to 3),
  and `OLLAMA_LLM_LIBRARY=cpu` does not change the scheduler. The fix hides
  the GPU from Ollama only:
  - `%LOCALAPPDATA%\OllamaCPU\start_ollama_cpu.cmd` sets
    `CUDA_VISIBLE_DEVICES=-1` and `OLLAMA_MAX_LOADED_MODELS=3` for the Ollama
    process and starts `ollama app.exe`; the login shortcut
    `Startup\Ollama.lnk` now runs this script (original backed up as
    `%LOCALAPPDATA%\OllamaCPU\Ollama.lnk.original`). `OLLAMA_MAX_LOADED_MODELS=3`
    is also a user environment variable. `CUDA_VISIBLE_DEVICES` is **not** set
    user- or machine-wide, so other GPU programs are unaffected.
  - Measured: 3 models co-resident; switching back to a loaded model costs
    0.1 s instead of a reload; a judgment on a resident model 9–11 s. Cost:
    the first, uncached prompt read is slower without the GPU (qwen2.5 ~8 s →
    ~35 s for ~680 tokens); later items reuse the cached instruction prefix.
  - **Ollama 0.35+ also finds GPUs through Vulkan.** Ollama auto-updated
    from 0.9.3 to 0.35.1 on 2026-10-06 (~10:40) and started evicting models
    again: the scheduler now counted the GPU via `library=Vulkan`, which
    `CUDA_VISIBLE_DEVICES` does not hide. The script therefore also sets
    `OLLAMA_VULKAN=0` and `GGML_VK_VISIBLE_DEVICES=-1`. Verified afterwards:
    phi4 and mistral-nemo stay co-resident, switching back to phi4 takes
    2.4 s, RAM 50% with both loaded.
  - Verify with the server log: the `server config` line should show
    `CUDA_VISIBLE_DEVICES:-1`, `OLLAMA_VULKAN:false` and
    `GGML_VK_VISIBLE_DEVICES:-1`, and `inference compute` should say
    `library=cpu`. A log line `predicted to exceed available memory,
    evicting` means a GPU is visible again. Starting Ollama from the tray or
    Start menu bypasses the script, and an Ollama update may rewrite the
    startup shortcut or add another GPU backend — if models start evicting
    each other again, check the log and re-point `Ollama.lnk` at the script.
- **Windows:** `cli.py` forces UTF-8 output; shell heredocs can mangle
  regex escapes and invisible characters (soft hyphens) — write edit scripts
  to files instead.
- **Reports are overwritten** each run (`eval_report.md`, `judge_report.md`);
  copy them before rerunning if you need the previous numbers.
- A private checklist of these steps is kept as a claude.ai artifact
  (owner's account).

## 9. Commit trail

| Commit | Change |
|---|---|
| `837d030` | `check` verifies the user's sentence; explicit refusal scoring; UTF-8 console |
| `0353c49` | Strict/lenient precision; micro-averaged claim metrics |
| `e670d75` | Over-refusal and answer-level hallucination rates |
| `42e05fe` | Verbatim-quote grounding; multi-model judge panel |
| `0dfa86b` | Judge validation set and judge/panel scoring |
| `9f6530a` | Prompt v2 (overclaim check); `<think>` stripping; prompt comparison |
| `271e5f8` | Validated defaults; `.env` loading; citation-id repair |
| `0430978` | Misplaced-refusal and wrong-passage handling; progress logging |
| `b9a3a6b` | Trap tiers; unclosed-tag parsing; golden-set copies ignored |
| `107fdb9` | Multi-corpus switch; journal PDF ingestion; page locators |
| `2e3b63c` | PDF hardening on a real article; resource guard |
| `4a0444f` | Prompt v3; plan→result and attribution items; regression items |
| `40461dc` | This handoff file (later updates: `e554507` Stage A, `9c85db5` Ollama fix) |
| `d789ff4` | Token budget for Ollama thinking models (deepseek-r1 empty answers) |
| `01e8580` | Scaled Ollama timeout; failed judge casts no vote; re-judge cached `unparsed` |
| `5eed641` | deepseek-r1 dropped as a judge |
| `a113f24`, `5345aee` | Guard unloads only its own models; default judge phi4 + prompt v3; Stage B step 1 results |
| `257ca4c`, `c29e747` | Per-claim detail in eval results; `--answerable-only`; `OLLAMA_NUM_THREAD`; system results under the new default and claim audit |
| `370f48b`, `a1e3f3d` | Quotes may span retrieved chunks; judges see the section heading; corrected diagnosis |
| `c93d081` | Confirmation rerun: strict precision 97% |
| `bf1c42b`, `8371906` | `judge-eval --with-heading`; headings do not make phi4 more lenient |
| `901bf30`, `2c004da` | Plan-as-result rule check; unit tests (`tests/`) |
| `846ec63`, `a8d8257` | Resumable `eval --with-llm` (`--fresh`) |
| `7e6bcdc`, `73da0a7` | Traps + pilot after fixes; timeout covers the prompt; judge failure → review |
| `e210680` | Timeout fix confirmed on the two claims that had timed out |
| `0a703ea`, `71545ee` | Confidence intervals in reports; hard plan-to-result judge items |
| `9bfba0e`, `621fecc` | phi4 on the hard set; structural plan-as-result route |
| `f9e7599`, `91aaf89`, `374353b`, `10dc3da`, `5ec514b` | Generator prompt v2 (opt-in) and its evaluation; v1 stays |
| `b401336`, `e7785a0`, `3de2902` | Judge prompts v4 and v5 (opt-in); v3 stays |
| `418a378`, `026ee60` | Baseline arm; `OLLAMA_NUM_CTX` |
| `4f3e1be`, `0835d39`, `da5e395` | Baseline abstention and splitting fixes; `--reuse-from` |
| `4b83c73`, `e8a03f3` | Grounded vs baseline results; trap audit fixes; `--trap-types`; bait run |
| `cf319f7`, `6733059` | Sensitivity re-judging tool (`eval.rejudge`) and results |
| `fe357ad` | Strict-safe trap rule |
| `ef06d21`, `7fa41d5` | Revised bait set; hypothesis-match rule route (on by default) |
| `697f890` | HANDOFF reorganised: short §7, history in §10 |
| `a5c1baf` | Crash dump analysed: WSL GPU paravirtualisation bug; mitigation |
| `2166c9d` | Sixth reset; unplanned Ollama 0.40.5; bait expansion paused |
| `a3bc628` | Run signature includes the Ollama server version |
| `ec2bce3`, `13ac317`, `d32aa5f` | 12-item bait results; Chinese abstention and advice patterns; README overview |
| `bf4eead` | check-cost measurement tool |
| `32d1ef4` | `CHECK_TOP_N` 3 → 2 from the measurement |
| `d90b293` | The two check false accepts were gold-label artefacts; rule unchanged |
| `82f91fb` | PDF ingestion: whitespace spans; body may start at an Introduction |
| `71bedc4` | Out-of-corpus check on two open-access papers |
| `36aab0a`, `dfc2122`, `812fb96`, `bc78857` | Generator prompt v2.1; `--ids`; raw XML kept; quote recovery; full-sentence rule |
| `44130e0` | v2.1 becomes the default generator prompt |
| `bc0cba0` | Parser: stray `</quote>` keeps its quote |
| `73e0ba0`, `6f8ec09` | Out-of-corpus rerun with generator v2.1; rule 7 tightened |
| `b1f6991` | Calibration 4 of v2.1 |
| `a9beec7`, `3567d4f` | Result-passage rule (tried) and the v2.1 length test |
| `6cea920` | Result-passage rule withdrawn after calibration 5 |
| `8da43f8` | `CONTEXT_ORDER` (results-first option) |
| `c7d1491` | Context order tried, not adopted |
| `f933c50` | Inline-hypothesis route of the plan-as-result rule |
| `802e99c`, `6c69964` | Cross-lingual retrieval: bge-m3 option, dense-only for Chinese queries |
| `0405f9d` | Parser: a stray `</quote>` no longer becomes the quote |
| `11735ed` | bge-m3 becomes the default embedder |
| `08e820a` | PDF reader: drop Frontiers 2024 sidebar, bare DOI headers and equation debris |
| `bdb3198` | Baseline judging: plan rule per passage, closest non-hypothesis passage; more abstention patterns; Pratviel 2024 results |
| `5ea9d0d` | Thesis baseline re-scored under the new judging (README table updated) |
| (this commit) | `eval --lang zh`; Chinese-question end-to-end results |

## 10. Completed work log

Moved from §7 on 2026-10-16 unchanged (the numbering is the original
order: 0000 is the newest). Each entry keeps its numbers, audits and
file names.

Done before item 1 (from the earlier handoff): deepseek-r1 dropped (§5 item
7); Stage B step 1 run (§4); default judge set to phi4 alone with prompt v3
(owner's decision, 2026-10-07; §2); system evaluation rerun on both corpora
under the new default, plus a per-claim rerun of the dissertation's answerable
questions and an audit of every non-supported claim (§4).

1. ~~Check that the section heading does not make the judge more lenient~~ —
   done 2026-10-09: no change in false-accept or false-reject (§4).
   `judge-eval --with-heading` shows judges the same `（章節：…）` prefix as the
   live verifier (shared `verify.with_heading`). The two audit fixes
   (implemented 2026-10-08, confirmed by a rerun, §4):
   - `verify.quote_span`: a quote not found in one passage is accepted when
     every sentence is (near-)verbatim in some retrieved passage; the claim is
     then judged against all spanned chunks (recorded as a citation repair).
     One sentence found nowhere still marks a fabrication; single-sentence
     quotes are unaffected.
   - The judge now sees each passage as `（章節：heading path）` + text
     (`Passage.section_path`, read from the chunk file; no index rebuild).
     Quote checks still use the bare text. The judge validation set
     (`judge-eval`) still passes bare passages, so its numbers are not directly
     comparable on items where the heading matters.
0000. **Grounded vs baseline comparison — done 2026-10-15 (results in README
   §Results); two follow-ups open (below).**
   `eval --with-llm --baseline`: the ungrounded `SYSTEM_BASELINE` prompt over
   the same top-5 context; the prose answer is split into sentences
   (`split_sentences`), abstentions ("does not mention", 未提及 …,
   `REFUSAL_HINT_RE`) are not claims, and every other sentence is judged by
   phi4 v3 against its 2 closest context passages (embedding similarity,
   early stop). Output has the grounded arm's shape, so metrics, intervals and
   resume are shared; the arm is part of the run signature. The grounded arm
   reuses existing v1 results. Plan: 2-0 calibration (3 answerable + 3 traps,
   audit splitting and refusal detection), then 2-1 the dissertation's 26
   answerable + 25 traps (~5 h). Caveats: not a paired comparison (generator
   randomness), baseline judged against 2 of 5 passages, refusal detection is a
   heuristic, off-topic true sentences count as unsupported.

   2-0 calibration (2026-10-12 15:12–17:36, phi4 shared with another
   project's phi4 job at matched `OLLAMA_NUM_THREAD=2`/`OLLAMA_NUM_CTX=8192`):
   splitting was clean on all 6 answers (decimals and statistics intact).
   One abstention was missed ("does not *explicitly* report") and judged
   unsupported → the regex now allows one adverb between negation and verb.
   All 3 far_absent traps opened by denying the premise, then added related,
   mostly supported content, so none counts as a refusal under the shared
   definition (abstains and makes no claim; same rule as the grounded arm).
   The baseline report now adds a secondary trap metric, "opens by denying
   the premise" (`leading_denial`). Speed while sharing phi4: ~24 min per
   question, ~7 min per judged sentence; the fix changes the run signature, so
   2-1 reruns the 6 calibration questions.

   2-1 (2026-10-12/13): stopped once by the guard (swap +2.6 GB from another
   program), then the machine crashed twice (bugcheck 0x7E at 23:54; a
   power-loss style reset ~01:13; cause not yet found — no crash since).
   Audit of the answerable set: 4 partials were split fragments (group labels
   like "X vs. Y" cut at "vs.") → the splitter skips common abbreviations; at
   least 2 unsupported sentences have their evidence in the context but not
   in the 2 closest passages the judge saw; 1 label contradicts its reason.
   Finished 2026-10-13 10:27–12:52 with `--reuse-from f5f1b8afca127666` (37
   answers reused, 2 re-joined sentences + 20 traps judged; run 86e12791…).
   Trap audit (all 31 answers by hand): abstentions phrased "does not
   explicitly address/detail" and advice sentences ("would need to review the
   full dissertation", "further analysis would be necessary") had been judged
   and rejected → `REFUSAL_HINT_RE` gained address/detail, and `META_RE`
   sentences are not judged. Re-scored without model calls (`--reuse-from
   86e12791e089c21f`; 57 answers, 189 verdicts reused; local
   `eval_results_baseline_rescored.json`; the pre-audit file is
   `eval_results_baseline.json`).

   Final numbers (grounded v1 vs baseline; Wilson 95%; exact McNemar on the
   same questions): strict precision 97% [84–99] vs 90% [83–95] (85/94);
   hallucination 0% vs 4% (4/94); answers with any non-supported claim 1/26 vs
   7/26 (p = 0.031; 6/26, p = 0.063 after the 2026-10-22 re-score, §7 6c);
   answers with an unsupported claim 0/26 vs 4/26 (p =
   0.125); must-refuse traps refused 18/19 vs 3/19 (p < 0.001), safe 18/19 vs
   14/19 (p = 0.125); all 25 traps correct 22/25 vs 6/25 (p < 0.001);
   false-premise correct 4/6 vs 3/6; baseline opens by denying the premise on
   13/19 must-refuse traps. Manual reading: of the 25 shared traps the
   baseline accepted the premise or invented content on 7 (one invented a
   participant group, one reported a planned analysis as a result); the rest
   denied or corrected the premise, usually followed by related content.
   Hypothesis-bait: baseline 0/6 (hypotheses or guesses stated as findings;
   one item may have a matching result in the results chapter — check);
   grounded v1 never ran these 6 items (v2 reference: 6/6).

   Hypothesis-bait, grounded v1 (2026-10-15 12:28–12:39, `eval --with-llm
   --traps-only --trap-types hypothesis_bait`, 4 threads): safe 5/6 vs
   baseline 0/6 (exact McNemar p = 0.063). Refused 3; answered 2 from
   passages that report results (abstract; results chapter) with verbatim
   quotes; restated the hypothesis once (B02) and phi4 rejected it. B04 and
   B05: the thesis reports a matching result, so these items do not test
   plan-as-result and the baseline's rejected sentences there were judged
   against a non-results passage — owner to review or replace the 2 items.
   Local file: `eval_results_bait_grounded_v1.json`.

   Bait set revised 2026-10-15 22:20 (local golden set; backup
   `golden_set.before-bait-swap.json`): B04 → B07 and B05 → B08, built
   on the two hypotheses whose predicted results are not reported (B07: the
   results report that measure only for other locations; B08: no matching
   analysis is reported); both retrieve the hypotheses chunk. B01 kept,
   though the results report related changes elsewhere (answering from those
   passes). Rerun 22:25–22:50 (grounded all 6; baseline
   `--reuse-from 86e12791e089c21f`, B07/B08 new; local
   `eval_results_bait2_*.json`): safe grounded 4/6 vs baseline 0/6 (p =
   0.125), strict safe 3/6 vs 0/6 (p = 0.25). Grounded generator: refused
   B01/B03/B06; restated the hypothesis on B02 (phi4 rejected), B08
   (re-attributed by verbatim quote to the hypotheses chunk → structural rule
   disputed) and B07 (cited the results chunk with a verbatim quote → no rule
   route fired; phi4 said partial, misreading the passage) — B07 is the open gap
   recorded in README Known limitations. Baseline restated the hypothesis on
   all 6. The machine reset again at ~22:52, after the run had finished
   (Kernel-Power 41, bugcheck 0; third unexplained reset, see §8 notes).

   Sensitivity tool (2026-10-15): `python -m eval.rejudge` re-judges, for
   both arms, every non-supported answerable claim and every unsupported /
   disputed trap claim against the context passages it was not judged on
   (baseline: all but its 2 closest; grounded: all but the cited one), keeps
   the better label and recomputes the paired comparison. Scope C: 36 claims
   (baseline 9 + 22, grounded 1 + 4), at most 113 phi4 calls; resumable
   (`rejudge_progress.jsonl`); `--limit N` for a trial. Selection checked on
   the saved results: every saved verdict's passage is among the recomputed
   ones (retrieval and embeddings deterministic).

   Sensitivity run (2026-10-15 13:46–15:22, scope C, 36 claims, phi4 4
   threads; local `rejudge_results.json`): 14 labels changed. Grounded →
   strict 31/31, answers not clean 0/26, traps correct 24/25, bait safe 6/6;
   baseline → strict 90/94, hallucination 2/94, answers unsupported 2/26,
   not clean 4/26, must-refuse safe 14/19 (unchanged), bait safe 1/6. Paired
   p: answers not clean 0.125, answers unsupported 0.5, must-refuse safe
   0.125, traps correct < 0.001, bait 0.063. Justified flips: baseline R05,
   R19, R21, R23, R26 (one sentence), B04 (one sentence) — evidence in an
   unseen passage. Lenient flips (unsupported → partial, counted safe):
   grounded B02 and baseline B02 (hypothesis stated as confirmed), grounded
   T25 (false premise accepted); grounded T24 → supported on a circular
   sentence. Baseline R26's other sentence → disputed by the plan rule (a
   methods passage in future tense). R06: phi4's reason reads as support but
   the label is unsupported (owner to check). Conclusion: report the main
   table as the strict bound and the sensitivity table as the lenient bound;
   the trap advantage holds under both, the answerable gap does not reach
   significance once the judge sees all passages.

   Open: (c) ~~bait items B04/B05~~ replaced (above); owner review of R06's
   label; (e) ~~close the B07 gap~~ — done 2026-10-15 as the
   hypothesis-match route (`verify.hypothesis_match`, `PLAN_RESULT_MATCH=on`
   by default, bge-m3, similarity >= 0.55, margin >= -0.02, partially
   supported claims only). Offline set (local scratchpad): 2 real + 10
   simulated positives, 7 hard negatives (hypotheses the results confirm,
   citing the result), 129 ordinary claims. Ordinary claims separate cleanly
   (in-context margins <= -0.11) but hard negatives overlap the positives
   (+0.01 to +0.21), so embeddings cannot tell confirmed from unconfirmed
   hypotheses — hence the partial-only gate. bge-m3 chosen over multilingual
   MiniLM (B07 margin +0.040 vs +0.013). Re-applied to all saved outputs (32
   partially supported claims): fired on grounded B07 and one baseline B02
   sentence, 0 of the other 30; answerable and non-bait trap metrics
   unchanged; grounded bait safe 4/6 -> 3/6. Not independently validated
   (thresholds set on the same outputs; one real positive). The baseline and
   `check` now pass all retrieved passages to the rule (`context`); (d) ~~stricter trap rule~~ — done
   2026-10-15 as *strict safe* (no unsupported, disputed or partially
   supported claim): `run_eval.strict_safe`, a column in the traps-by-type
   table, and `eval.rejudge` now reports both the original-label and the
   re-judged comparison with `premise_strict_safe` and `traps_correct_strict`.
   Detecting "restates the premise" directly was tried and dropped: lexical
   overlap with the question fails across languages (Chinese answers to
   English questions) and is as high for corrections as for acceptances.
   Premise traps (false_premise + hypothesis_bait, 12): strict safe grounded
   8/12 → 9/12 (re-judged) vs baseline 1/12 → 1/12, p = 0.016 / 0.008; plain
   safe 9/12 → 12/12 vs 3/12 → 4/12. Strict safe fails the partial lenient
   flips (grounded T25, B02) and grounded T21, not grounded T24 (upgraded to
   supported), and it also fails baseline T22/T23, which correct
   the premise and add one partially supported sentence — it favours short
   answers, so report it beside the plain safe rate.
000. **Judge prompt v5 — evaluated 2026-10-12; not adopted. Judge prompt work
   closed; default stays v3 (owner's decision).** v5 = v3 verbatim + v4's
   classification step and fields (phi4, with headings, 3 threads, run
   alongside another project's qwen3 job, 05:34–10:31).

   | | v3 | v4 | v5 | v5 + rule |
   |---|---|---|---|---|
   | Dissertation held-out FA / FR | 5% / 7% | 25% / 7% | 7% [3–17] / 4% [1–18] | same as v5 |
   | Pilot held-out FA / FR (v3 without heading) | 6% / 5% | 23% / 0% | 14% [6–29] / 0% | same as v5 |
   | Dissertation overclaim caught | 24/25 | 12/25 | 22/25 | — |
   | Hard plan items caught (cue-free) | 4/12 (1/5) | 9/12 (2/5) | 5/12 (0/5) | **12/12 (5/5)** |
   | Hard controls false-reject | 2/14 | 1/14 | 1/14 | 1/14 |

   Neither criterion met: no held-out regression (pilot +3 false accepts:
   a negation, a swapped passage, an overclaim) and a hard-set gain (5/12 vs
   4/12). v5 did not inherit v4's hard-set gain, so v4's gain came from a
   shift in leniency (more plan items rejected, many more overclaims
   accepted), not from better stage understanding. What works is v3 plus the
   two deterministic rules (wording + structural): held-out FA 5–6% and all 12
   hard plan items stopped, with no extra judge time (v4/v5 outputs are longer,
   ~20% slower). Lesson: for a 7–14B judge, restructuring the prompt moved its
   strictness, not its understanding; deterministic checks were more reliable
   than further prompt iterations. v4 and v5 stay in the code, opt-in, for
   comparison.
00. **Judge prompt v4 — evaluated 2026-10-11; not adopted (regression).**
   Stage 2a (hard set, development data): plan items caught 33% → 75% (with
   rule 100%), control false-reject 14% → 7%. **Stage 2b (held-out) showed a
   large regression**: false-accept dissertation 5% [2–15] → 25% [16–38]
   (14/55), pilot 6% → 23% [12–39] (8/35; pilot v3 votes lacked headings,
   a minor confound); false-reject unchanged (7%) / 5% → 0%. Cause:
   **overclaim detection collapsed** (dissertation 24/25 → 12/25) because v4
   dropped v3's explicit lists of hedge and overclaim words to shorten the
   prompt; attribution swaps also slipped. Synthetic plan→result improved
   (4/6 → 6/6), but v3 + the structural rule already caught all 12 hard plan
   items. Lesson: removing concrete cue lists from a 7–14B judge's prompt
   costs more than the shorter prompt gains, and a development set without
   overclaims could not show it. Next: v5 = v3 kept verbatim + v4's
   classification fields appended.
   Original v4 description: (opt-in, `judge-eval --prompts v4`; default stays v3). The judge first classifies
   the best supporting sentence (`source_stage`: result / hypothesis /
   plan_or_method / background / other_study), the claim (`claim_stage`:
   finding / hypothesis_or_plan / method / background) and the claim's
   attribution; `verify.stage_verdict` then forces `unsupported` for a finding
   drawn from a hypothesis/plan/method sentence or another study's result
   claimed for this study, instead of trusting the judge's own flag (phi4
   rarely set v3's). Output limit 400 tokens for v4. Plan: stage 2a on the hard
   set (development data — the failure modes shaped v4), then 2b regression on
   both held-out splits if 2a helps.
0. **Generator prompt v2 — evaluated 2026-10-10; not adopted, default stays
   v1 (owner's decision).** Full rerun of both corpora (06:20–12:16, 3 threads,
   alongside another project's job; `eval_results_v2.json`, local). No false
   claim shown as supported under either prompt; all differences are inside
   the 95% intervals.

   | | Dissertation v1 → v2 | Pilot v1 → v2 |
   |---|---|---|
   | Strict / lenient precision | 97% / 100% → 90% / 90% (26/29) | 89% / 95% → 89% / 100% |
   | Unsupported claims | 0 → 3 | 1 → 0 |
   | Misplaced refusal markers | 13 → **25** | 12 → 15 |
   | Translated quotes | 0 → 0 | 1 → 0 |
   | Near-absent / false-premise traps | 89% / 67% → 100% / 83% | 100% / 67% → 83% / 83% |
   | Hypothesis-bait traps (6) | — → 6/6 safe | — |

   Audit: the three new dissertation rejections are citation/format errors on
   true claims — a result cited to a methods passage (rule 7 not followed),
   the template placeholder "段落編號" copied as the citation id (new failure),
   and two sentences merged into one quote (stitching, though forbidden). All
   other non-supported claims were judged correctly by phi4. Per rule: the
   source-language rule removed the one translated quote; attribution (rule 8)
   had no visible effect; negative facts (rule 9) did not lower over-refusal;
   the emphatic refusal rule backfired (more markers inside <claim>). The bait
   traps never cited a hypotheses section, so the structural rule's recall on
   real outputs is still unmeasured. If revisited (v2.1): a real example id in
   the template instead of the placeholder, a positive or separate refusal tag
   instead of the "never put it in <claim>" wording, and drop rules 8–9 to
   shorten the prompt for the 7B generator.

   What v2 contains (`GENERATOR_PROMPT=v2`): quotes in the source language; each passage is
   labelled with its heading path and a type (研究假設 / 研究方法 / 研究結果 /
   討論 / 背景 — from the chapter, except an innermost "Hypotheses" heading),
   and hypothesis/methods passages may only be reported as plans; findings of
   cited studies attributed to them; negative facts answered; the refusal
   marker only in `<unsupported_note>`. Evaluation adds a translated-quote
   count (v1 baseline: dissertation 0/35 claims, pilot 1/21; misplaced
   refusals 13 / 12) and 6 local `hypothesis_bait` traps (dissertation;
   reported separately) that tempt restating a hypothesis as a finding, to
   test the structural rule on real outputs. Watch: rule 7 may raise refusals
   when only methods passages are retrieved; rule 9 may turn "not mentioned"
   into "did not happen".
2. ~~Close the plan→result gap~~ — rule check done (§4); traps and the pilot
   paper rerun with all fixes (§4); timeout fix confirmed on the two claims
   that had timed out (§4). Optional: harder plan→result items.
3. `check` cost under phi4: it judges each sentence against up to five
   passages (45 s – ~4 min per sentence on CPU). Consider judging only the top
   one or two passages; measure the effect on its verdicts first.
4. Generator prompt: answer negative facts; attribute cited studies
   explicitly; keep quotes in the source language; put refusals only in
   `<unsupported_note>`.
5. ~~Resumable `run_eval`~~ — done 2026-10-09: each scored question is
   appended to `eval/reports/eval_progress.jsonl` with a run signature (models,
   prompt, rule settings, top-k, embedding model, hash of generate/verify/
   retrieve code); rerunning the same command skips saved questions; the file
   is renamed when the run completes; `--fresh` ignores it. Aggregates were
   checked identical to the previous implementation on fake data
   (`tests/test_run_eval_resume.py`). Long runs are still best launched
   detached.
6. Human-labelled judge items (`data/golden/judge_set_human.jsonl`, schema in
   `judge_set_human.example.jsonl`) — owner's task, not started.
7. ~~Confidence intervals~~ — done 2026-10-10: `eval/stats.py` (Wilson 95%);
   both reports print rate [CI] (k/n); results JSON carry `counts` /
   `false_accept` / `false_reject` with intervals. Headline intervals are in
   the README ("How certain are these numbers?"). Main reading: judge
   comparisons and the 84% → 97% gain overlap within their intervals (the gain
   is paired, 4/4 discordant claims in one direction, exact p = 0.125).
   Still open: grounded vs baseline prompt comparison.
