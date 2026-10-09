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
this set (the only real test of it). phi4 scoring on the set is the next step.

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

## 7. Next steps (in order)

Every step that runs local models needs the owner's approval of a plan and
resource assessment first (§8).

Done since the last handoff: deepseek-r1 dropped (§5 item 7); Stage B step 1
run (§4); default judge set to phi4 alone with prompt v3 (owner's decision,
2026-10-07; §2); system evaluation rerun on both corpora under the new default,
plus a per-claim rerun of the dissertation's answerable questions and an
audit of every non-supported claim (§4).

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

## 8. Operational notes

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
| (this commit) | Timeout fix confirmed on the two claims that had timed out |
