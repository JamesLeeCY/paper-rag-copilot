# Handoff — Dissertation RAG Copilot

State as of **2026-10-06**. Read this first when picking the project up. The
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
| `src/llm.py` | Claude / Ollama backends; strips `<think>`; checks the model is pulled |
| `src/pipeline.py` | `ask`, `check` (verifies a user's own sentences), CLI rendering |
| `eval/run_eval.py` | Main harness: retrieval, grounding, refusal, traps by tier |
| `eval/judge_set.py` | Builds the known-answer judge validation set (rule-based perturbations) |
| `eval/judge_eval.py` | Scores judges / panels / prompt versions; vote cache makes it resumable |
| `eval/resource_guard.py` | Wraps long local-LLM jobs; stops them on RAM/CPU/swap overload |

### Default configuration (validated)

| Setting | Value | Why |
|---|---|---|
| Generator | `qwen2.5:7b-instruct` | Follows the XML/quote format far better than llama3 |
| Judges | `qwen2.5:7b-instruct` + `gemma3:12b`, unanimous | Two families; 0% false-accept / 4% false-reject on the held-out judge split (v2) |
| Judge prompt | `v2` | `v3` helps on synthetic items but its false-reject rate is unmeasured and it still misses the real regression case (see §6) |
| Chunking | section-aware | Best on the dissertation; see §4 for the pilot paper |

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

## 5. Known weaknesses and open findings

1. **Plan read as result — still open.** The generator restated a planned
   assessment (future tense) as a reported finding. All three judges approve
   it under v2 **and under v3**. v3 does catch the synthetic version
   (deepseek-r1: 91%), so the synthetic items are easier than the real
   error: the real claim is a Chinese paraphrase of an English future-tense
   passage, not "The results showed that" + the source sentence. Options:
   harder plan→result items shaped like real outputs (paraphrased, Chinese
   claim / English passage), human-labelled items, and a rule-based check
   (cited passage has planning/future markers + claim asserts a result →
   flag) that does not depend on the judge noticing.
2. **Cited study attributed to this study — partly addressed.** v3 helps on
   "Author et al. found → This study found" swaps, but the pilot paper's
   background-statement swaps stay at ≤ 40% detection.
3. **Self-judging — and qwen2.5 is the weakest judge here.** qwen2.5 is both
   generator and judge, has approved its own errors twice, and v3 barely
   changes its votes on the new error types (87% false-accept). Candidates to
   replace it: `deepseek-r1:8b` (best on the new types, but built on Qwen3
   weights), `gemma3:12b`, and the newly installed `phi4` (Phi family) and
   `mistral-nemo` (Mistral) — not yet evaluated.
4. **Negative facts refused.** "The paper says X was not done" was answered
   with the refusal marker (over-refusal).
5. **Format drift.** Refusal marker placed inside `<claim>` (3× dissertation,
   12× pilot), translated quotes, occasional unclosed tags. All handled and
   counted; worth reducing in the generator prompt.
6. **Judges are not deterministic.** The same claim/passage can get different
   votes across runs; decide on larger sets or repeated runs, not one replay.

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

**Stage B — not started.** v3 on the full held-out split (all item types, so
false-reject is measured), adding `phi4` and `mistral-nemo`.

**Ollama contention (fixed 2026-10-06, see §8).** Stage A was paused mid-run
because another local project (`D:\side_project\line_chat`, a labelling job on
`qwen3:8b`) shared the Ollama server, and Ollama kept only one model loaded:
the two jobs evicted each other's model on every request (~2.5 min per
judgment instead of ~8 s). A detached waiter resumed Stage A once that job
exited. Ollama now keeps up to three models loaded, so this should not recur.

## 7. Next steps (in order)

1. **Stage B** — v3 on the full held-out judge split (all item types) to
   measure false-reject; add `phi4` and `mistral-nemo`; choose a judge panel
   **without the generator model** (qwen2.5 is also the weakest judge on the
   new error types). Leading candidate so far: gemma3:12b + deepseek-r1:8b
   under v3 (7% false-accept on the dissertation's new error types).
2. **Close the plan→result gap.** The real case is still approved under v3:
   add harder items shaped like real generator outputs, and/or a rule-based
   check (planning/future markers in the cited passage + a result-asserting
   claim → flag) that does not rely on the judge.
3. Make the chosen prompt + panel the default and rerun both corpora.
4. Generator prompt: answer negative facts; attribute cited studies
   explicitly; keep quotes in the source language; put refusals only in
   `<unsupported_note>`.
5. Resumable `run_eval` (save per question) — the tool's background tasks are
   killed after ~30 min; long runs are launched detached for now.
6. Human-labelled judge items (`data/golden/judge_set_human.jsonl`, schema in
   `judge_set_human.example.jsonl`) — owner's task, not started.
7. Confidence intervals in reports; grounded vs baseline prompt comparison.

## 8. Operational notes

- **Long local runs:** launch detached (PowerShell `Start-Process`) under
  `eval.resource_guard`; tool-managed background tasks stop at ~30 min.
  Measured load with two judges: CPU ~50–80%, RAM ~45% of 64 GB; with three
  models resident (qwen2.5 + gemma3:12b + deepseek-r1:8b) RAM ~60%.
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
  - Verify with the server log: the `server config` line should show
    `CUDA_VISIBLE_DEVICES:-1` and `inference compute` should say
    `library=cpu`. Starting Ollama from the tray or Start menu bypasses the
    script, and an Ollama update may rewrite the startup shortcut — if models
    start evicting each other again, re-point `Ollama.lnk` at the script.
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
| `40461dc` | This handoff file |
