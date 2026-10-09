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

**Current default:** generator `qwen2.5:7b-instruct`, judge **`phi4`** alone,
judge prompt **`v3`**, section-aware chunking — chosen on the judge validation
below (2026-10-07). "Previous" below means judges `qwen2.5:7b-instruct` +
`gemma3:12b` (unanimous) with prompt `v2`.

### At a glance

| | Dissertation, previous | **Dissertation, current** | Pilot, previous | **Pilot, current** | Target |
|---|---|---|---|---|---|
| Retrieval Hit@5 (section-aware) | 100% (MRR 0.952) | 100% (MRR 0.952) | 95% | 95% | ≥ 90% |
| Strict citation precision | 96% (26/27) | 84% (26/31) → **97% (30/31)**¹ | 90% (18/20) | 90% (18/20) | ≥ 95% |
| Lenient citation precision | 96% | 97% → **100%**¹ | 90% | **95%** | — |
| Claim / answer hallucination rate | 0% / 0% | 3% / 4% → **0% / 0%**¹ | 10% / 11% | **5% / 5%** | low |
| Must-refuse traps refused | 95% | 95% | 100% | 100% | ≥ 90% |
| False-premise traps safe | 83% | **100%** | 67% | **83%** | — |
| Over-refusal on answerable questions | 0% | 0% | 5% | 5% | low |
| Parse failures | 1 | **0** | 0 | 0 | — |

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

### Retrieval — chunking ablation (dissertation)

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| fixed-size (baseline) | 69% | 88% | 88% | 0.794 |
| **section-aware** | **92%** | **96%** | **100%** | **0.952** |

*26 hand-authored golden questions, `bge-small-en-v1.5`, hybrid (dense + BM25) +
RRF fusion, no reranker.* Section-aware chunking clears the ≥ 90% Hit@k target
and beats the fixed baseline by 12 points at Hit@5. On the pilot paper the order
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

### Judge prompts

Set with `VERIFY_PROMPT`; compare with `judge-eval --prompts v1 v2 v3`.

| Version | Adds | Status |
|---|---|---|
| `v1` | Plain entailment: supported / partially_supported / unsupported | Superseded |
| `v2` | Explicit evidence-strength field; a claim stronger than its source (overclaim) cannot be `supported` | Previous default |
| `v3` | Two extra checks: a plan, hypothesis or planned measure restated as a result (`plan_as_result`), and another study's finding claimed for this study (`misattributed`); either one forces `unsupported` | **Default** (validated with phi4) |

Judges answer in JSON. A self-contradicting vote (`claim_stronger` but labelled
`supported`) is downgraded to `partially_supported`.

---

## Setup

```bash
pip install -r requirements.txt          # torch is assumed preinstalled
```

Place the manuscript at the repo root as
`2026_Manuscripts_BXF+NTSEC_v2.1.docx` (or point `DISSERTATION_DOCX` at it).
Embeddings are always local; models download on first run (~130 MB for
`bge-small-en-v1.5`). Swap to a stronger model via
`EMBED_MODEL=BAAI/bge-large-en-v1.5`.

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
against up to five passages and stops at the first that supports it, so a
sentence costs 45 s to about 4 min. Adding gemma3:12b as a second judge roughly
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
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers embedding model |
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
| `PLAN_RESULT_RULE` | `flag` | Plan or prediction read as a result: `flag` → disputed, `reject` → unsupported, `off` |

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
python cli.py eval --with-llm --answerable-only      # answerable questions only
python cli.py eval --with-llm --fresh                # ignore an interrupted run's saved questions

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
which of your own statements the source does not support.

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
   from 3/55 to 1/55. Caveats: one real positive only; the synthetic items share
   the rule's cues; a true finding cited from a future-tense methods passage
   could still be flagged, which is why the default is review, not rejection.
2. **Background statements attributed to this study.** v3 catches most
   "Author et al. found → This study found" swaps, but on the pilot paper at
   most 40% of swaps of background statements.
3. **Self-judging — resolved in the default.** qwen2.5 was both generator and
   judge, and the weakest judge; the default judge is now phi4.
4. **Negative facts.** "The paper says X was not done" tends to be refused
   (over-refusal).
5. **Format drift.** The refusal marker inside `<claim>`, translated quotes and
   unclosed tags occur; all are handled and counted, not hidden.
6. **Judges are not deterministic.** The same pair can get different votes
   across runs; decide on larger sets or repeated runs.
7. **Reasoning-model judges are too slow on CPU.** With room for its thinking,
   `deepseek-r1:8b` needs 4–8 minutes per judgment (one exceeded 5 minutes), so
   it was dropped as a judge; its Stage B result above is invalid.
8. **Small samples.** Judge rates rest on 55–83 items and system rates on
   20–51 questions; no confidence intervals are reported yet.
9. **Verification is slow on CPU.** Each phi4 judgment takes 38–50 s (longer
   with `OLLAMA_NUM_THREAD=3`), so `check` on a long paragraph can take several
   minutes (see Setup). A judgment that still times out sends the claim to
   review rather than guessing; `n_unjudged` in the eval results counts these.
10. **Chunk boundaries.** A chunk judged alone can lack context a true claim
    relies on, and a quote can run across two chunks. Both are now mitigated
    (section heading shown to the judge; quotes checked sentence by sentence
    across retrieved chunks), but a claim whose support is split across chunks
    is still judged one chunk at a time.

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
8. Human-labelled judge items; confidence intervals; grounded vs baseline prompt
   comparison.

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
data/golden/           example schemas (real golden and judge sets are local)
tests/                 unit tests for the deterministic verification rules
docs/METHODOLOGY.md    design write-up
HANDOFF.md             current state, open findings and next steps
```
