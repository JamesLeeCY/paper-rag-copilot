# Handoff — Dissertation RAG Copilot

State as of **2026-10-05**. Read this first when picking the project up. The
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
| Judge prompt | `v2` | `v3` is written but **not yet validated** (see §5) |
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

## 5. Known weaknesses and open findings

1. **Plan read as result.** The generator restated a planned assessment
   (future tense) as a reported finding. In a replay with the v2 prompt,
   **all three judges approved it** (qwen2.5, gemma3:12b, deepseek-r1:8b) —
   a shared blind spot, not self-bias. → prompt v3 + `plan_to_result` items.
2. **Cited study attributed to this study.** Journal Discussions describe
   other studies; the generator presented one as the paper's own result.
   qwen2.5 approved it; gemma3:12b and deepseek-r1 did not. → v3 +
   `attribution_swap` items.
3. **Self-judging.** qwen2.5 is both generator and judge and has approved
   its own errors twice. Replacement candidates now installed: `phi4`
   (Phi family), `mistral-nemo` (Mistral), plus `deepseek-r1:8b` (built on
   Qwen3 weights — same family as the generator, so a weak substitute).
4. **Negative facts refused.** "The paper says X was not done" was answered
   with the refusal marker (over-refusal).
5. **Format drift.** Refusal marker placed inside `<claim>` (3× dissertation,
   12× pilot), translated quotes, occasional unclosed tags. All handled and
   counted; worth reducing in the generator prompt.
6. **Judges are not deterministic.** The same claim/passage can get different
   votes across runs; decide on larger sets or repeated runs, not one replay.

## 6. In flight — Stage A of the v3 validation (PAUSED)

Goal: v2 vs v3 on the new error types, judges qwen2.5 / gemma3:12b /
deepseek-r1:8b, both corpora (`--types plan_to_result attribution_swap`,
both splits; v3 is not tuned on these items). Saved votes so far: qwen2.5 v2
14, v3 4 (cache: `eval/reports/judge_votes.jsonl`).

**Why paused:** another local project (`D:\side_project\line_chat`, a labelling
job using `qwen3:8b`) shares the Ollama server. On this machine Ollama keeps
only one model loaded (small 2 GB GPU), so the two jobs evicted each other's
model on every request (~2.5 min per judgment instead of ~8 s). Stage A was
stopped; the other job was left alone.

**Auto-resume:** a detached waiter (`eval/reports/stage_a_resume.ps1`, local)
waits for that job's process to exit, then reruns Stage A under the resource
guard; progress goes to `eval/reports/stage_a.log`. To resume by hand:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -u -m eval.resource_guard --log-every-s 300 -- cmd /c eval\reports\stage_a.cmd
```

`stage_a.cmd` runs, for `CORPUS=dissertation` and then `nature_walking_2026`:

```
python -u cli.py judge-eval --judges ollama:qwen2.5:7b-instruct ollama:gemma3:12b ollama:deepseek-r1:8b --prompts v2 v3 --split all --types plan_to_result attribution_swap
```

Lasting fix for contention (owner's call — restarts Ollama): set
`OLLAMA_MAX_LOADED_MODELS=3`; with 64 GB RAM several models fit at once.

## 7. Next steps (in order)

1. **Finish Stage A.** If v3 closes the plan/attribution blind spot:
2. **Stage B** — v3 on the full held-out judge split (all item types) to
   confirm no rise in false-reject; add `phi4` and `mistral-nemo`; choose a
   judge panel **without the generator model**; then make v3 + that panel
   the default and rerun both corpora.
3. Generator prompt: answer negative facts; attribute cited studies
   explicitly; keep quotes in the source language; put refusals only in
   `<unsupported_note>`.
4. Resumable `run_eval` (save per question) — the tool's background tasks are
   killed after ~30 min; long runs are launched detached for now.
5. Human-labelled judge items (`data/golden/judge_set_human.jsonl`, schema in
   `judge_set_human.example.jsonl`) — owner's task, not started.
6. Confidence intervals in reports; grounded vs baseline prompt comparison.

## 8. Operational notes

- **Long local runs:** launch detached (PowerShell `Start-Process`) under
  `eval.resource_guard`; tool-managed background tasks stop at ~30 min.
  Measured load with two judges: CPU ~50–80%, RAM ~45% of 64 GB.
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
