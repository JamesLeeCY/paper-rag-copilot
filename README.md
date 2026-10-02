# 論文寫作 RAG Copilot — Dissertation RAG Copilot

A **strictly citation-grounded** retrieval-augmented assistant over a PhD
dissertation's literature and body, with a **quantified evaluation harness** that
measures whether its hallucination rate is low enough to trust.

Corpus: *Lee (2025), "The Effects of Nature-Based Interventions on Psychological,
Cognitive, and Neural Functioning Across the Lifespan"* — Introduction through
Discussion, plus its 84-entry reference list. Every answer traces back to the
exact section / paragraph span it came from.

> Built to demonstrate three things end-to-end: **prompt & context engineering**
> (citation-forcing prompts, chunking ablation), **RAG system design** (hybrid
> search + rerank + verification layer), and **evaluation methodology** (a golden
> set and a multi-metric benchmark, not vibes).

---

## Headline result

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| fixed-size (baseline) | 69% | 88% | 88% | 0.794 |
| **section-aware** | **92%** | **96%** | **100%** | **0.952** |

*26 hand-authored golden questions, `bge-small-en-v1.5`, hybrid (dense+BM25) +
RRF fusion, no reranker.* Section-aware chunking clears the spec's **≥90% Hit@k**
target and beats the fixed baseline by 12 points at Hit@5 — the chunking ablation
the spec asked for. The full report is generated locally at
`eval/reports/eval_report.md` (not committed; see Data & privacy below).

---

## Architecture

```
1. Ingestion   docx → section-hierarchy walk → chunk (fixed | section-aware)
               → metadata (section №, paragraph span, inline citations→refs)
2. Indexing    local bge/e5 embeddings → ChromaDB  +  BM25 sparse index
3. Retrieval   query → dense top-k ⊕ BM25 top-k → RRF fusion → (rerank) → top-k
4. Generation  Claude **or local Ollama**, citation-forcing XML prompt: every
               claim carries a chunk_id, or the model must say "查無直接支持此說法的段落"
5. Verification cascade, cheapest first:
               (0) quote grounding — each claim's verbatim <quote> must be found
                   in its cited passage, else it is rejected as fabricated
               (1) judge panel — independent LLM judges, ideally from different
                   model families, vote supported / partial / unsupported;
                   disagreement → "disputed" (+ optional NLI / lexical fallback)
6. Evaluation  golden set → pipeline → Hit@k / MRR / citation precision /
               hallucination rate / refusal correctness + chunking ablation
```

Why these choices:
- **Hybrid search** — academic prose is full of exact-match terms (`DiFuMo`,
  `ISFC`, `TFCE`, `dISFC`) that pure semantic embeddings blur; BM25 anchors them.
- **RRF fusion** — merges dense (cosine) and BM25 (unbounded) rankings without
  score calibration.
- **Separate verification pass** — a fresh LLM call that never sees the
  generator's reasoning, avoiding self-confirmation bias. This is what makes
  "strict" mean something measurable.

---

## Setup

```bash
pip install -r requirements.txt          # torch is assumed preinstalled
```

Place the manuscript at the repo root as
`2026_Manuscripts_BXF+NTSEC_v2.1.docx` (or point `DISSERTATION_DOCX` at it).
Embeddings are always local; models download on first run (~130 MB for
`bge-small-en-v1.5`). Swap to a stronger model via `EMBED_MODEL=BAAI/bge-large-en-v1.5`.

### Choosing the LLM backend (generation + verification)

The LLM layer is pluggable via `LLM_BACKEND`. It auto-selects: **Claude** if an
API key is present, otherwise **local Ollama** — so it runs fully offline out of
the box.

**Option A — Local Ollama (free, offline, default):**
```bash
ollama serve                       # if not already running
ollama pull qwen2.5:7b-instruct    # generator + judge
ollama pull gemma3:12b             # second judge (different model family)
#   defaults: OLLAMA_MODEL=qwen2.5:7b-instruct
#             JUDGES=ollama:qwen2.5:7b-instruct,ollama:gemma3:12b
```
A judge whose model is not pulled is skipped with a message; with no judge
left, the generator's own model judges.

**Option B — Claude API (higher quality on the structured prompts):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...          # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export LLM_BACKEND=claude                     # optional; auto-selected when key is set
```

> Why these defaults: on the judge validation set (below), `llama3` let 35% of
> planted errors through as `supported` and `gemma3:4b` rejected most correct
> claims, while `qwen2.5` + `gemma3:12b` under the v2 prompt let none through
> on the held-out split (4% false-reject). `qwen2.5` also attached verbatim
> quotes far more reliably as the generator. Settings can go in a `.env` file
> (see `.env.example`); real environment variables override it.

---

## Data & privacy

The corpus is an **unpublished PhD dissertation**, so nothing derived from it is
committed to this repository. `.gitignore` excludes the manuscript (`*.docx`),
the generated chunks/indexes, the golden set (`data/golden/golden_set.json`), and
the evaluation reports (`eval/reports/`). What ships is the **code, methodology,
architecture, and aggregate retrieval metrics** only.

To run it on your own corpus: drop a `.docx` at the repo root (or set
`DISSERTATION_DOCX`), then author a golden set following
[`data/golden/golden_set.example.json`](data/golden/golden_set.example.json) and
save it as `data/golden/golden_set.json`.

## Usage

```bash
# 1. Ingest + index both chunking strategies
python cli.py build --all

# 2. Ask a citation-grounded question
python cli.py ask "Which theory explains how natural environments restore directed attention?"

# 3. Reverse hallucination check — paste something you wrote, get per-sentence verdicts
python cli.py check "Forest therapy reduced salivary cortisol and improved sleep quality."

# 4. Run the evaluation harness (retrieval only — fast, no LLM)
python cli.py eval
#    ...with grounding / hallucination / refusal via the LLM backend:
python cli.py eval --with-llm
#    ...quick sample (e.g. 3 questions + 3 traps) for a slow local model:
python cli.py eval --with-llm --llm-limit 3
#    ...with cross-encoder reranker (downloads bge-reranker-base):
python cli.py eval --rerank
```

### Multi-model cross-check (judge panel)

Judges from the same model family tend to make the same mistakes, so the
verifier can poll several models and only accept a verdict they agree on:

```bash
ollama pull qwen2.5:7b-instruct
# PowerShell: $env:JUDGES="ollama:llama3:latest,ollama:qwen2.5:7b-instruct"
export JUDGES="ollama:llama3:latest,ollama:qwen2.5:7b-instruct"
export PANEL_RULE=unanimous      # or "majority" (useful with 3+ judges)
python cli.py eval --with-llm
```

Under `unanimous`, judges that disagree on *whether to accept* the claim
yield `disputed` (⚖), which counts as neither support nor hallucination and
is the queue for human review. Judges that all reject and differ only on
severity (partial vs unsupported) are resolved by majority, ties going to
`unsupported`.

Pick judges from **different model families** — check `ollama show <model>`:
`deepseek-r1:8b`, for instance, reports architecture `qwen3` (a Qwen3 model
distilled from DeepSeek-R1), so pairing it with `qwen2.5` is not a
cross-family panel. Reasoning models' `<think>` blocks are stripped
automatically. The judge prompt version is set by `VERIFY_PROMPT` (`v2`,
default, adds an explicit evidence-strength / overclaim check; `v1` is the
original) — compare them with `judge-eval --prompts v1 v2`. Leave
`JUDGES` unset for a single judge on the generator's backend. Set
`QUOTE_REQUIRED=1` to also reject claims that come without a supporting quote.

If **no** LLM backend is reachable (no API key **and** no Ollama server),
`ask`/`check` fall back to **mock mode**: retrieval is real, but generation and
verification are stand-ins (pipeline shape only). With either Claude or a running
Ollama, the grounding / hallucination / refusal numbers are fully live.

---

## Evaluation methodology

The golden set (`data/golden/golden_set.json`, kept local — schema in
[`golden_set.example.json`](data/golden/golden_set.example.json)) has two parts:

- **26 retrieval questions** — natural-language information needs, each tied to
  the docx paragraph index(es) that answer it. A retrieved chunk "hits" if its
  paragraph span covers a target — so the metric is **strategy-agnostic** and
  fairly compares fixed vs section-aware chunking.
- **10 trap questions** — topics genuinely *absent* from the dissertation
  (psilocybin, EEG, cortisol, blue-space, HRV, actigraphy…). A trustworthy system
  must refuse these, not confabulate. Because the retrieval questions are all
  answerable, they double as the over-refusal check: refusing everything would
  ace the traps but score 100% over-refusal.

| Metric | Definition | Target (spec §1.3) |
|---|---|---|
| Retrieval Hit Rate@k | correct chunk in top-k | ≥ 90% |
| Citation Precision (strict) | claims judged `supported` / total claims | ≥ 95% |
| Citation Precision (lenient) | claims judged `supported` or `partially_supported` / total claims | — |
| Hallucination Rate | unsupported claims / total claims | low |
| Answer Hallucination Rate | non-refused answers with ≥1 unsupported claim / non-refused answers | low |
| Over-refusal Rate | answerable (retrieval) questions wrongly refused | low |
| Refusal Correctness | trap questions correctly refused | ≥ 90% |
| Chunking Ablation | fixed vs section-aware on the above | — |

Claim-level metrics are **micro-averaged**: claims are pooled across all
questions before dividing, so a question with many claims is not under-weighted.

### Validating the judges themselves

Every grounding number above is only as good as the verifier's judges, so they
get their own benchmark — a **judge validation set** of (claim, passage) pairs
with known gold labels, built without any LLM by perturbing real sentences:

| Perturbation | Example edit | Gold |
|---|---|---|
| original | verbatim sentence vs its own passage | supported |
| negation | "increased" → "decreased", "was" → "was not" | unsupported |
| number | a sample size / duration / statistic changed | unsupported |
| overclaim | "may" → "will always", "suggests" → "proves" | partially_supported |
| conjunction | sentence + an unrelated claim appended | partially_supported |
| swap_passage | sentence vs an unrelated passage | unsupported |

```bash
python cli.py judge-build                       # -> data/golden/judge_set_synthetic.jsonl (local)
python cli.py judge-eval --judges ollama:llama3:latest ollama:qwen2.5:7b-instruct
```

The headline metric is the **false-accept rate** (a hallucination judged
`supported`), alongside false-reject rate, 3-class accuracy, Cohen's κ, and a
per-perturbation breakdown; panels are scored from the same cached votes, so
adding a judge only costs that judge's calls. Items are split dev/test by
source sentence, so a judge with a tunable threshold can be calibrated on dev
and reported on test. Synthetic positives are verbatim and therefore easy — add
your own labelled pairs to `data/golden/judge_set_human.jsonl` (schema:
[`judge_set_human.example.jsonl`](data/golden/judge_set_human.example.jsonl)).

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the design write-up.

---

## Layout

```
config.py              all tunable knobs (chunking, models, retrieval, LLM)
cli.py                 build | ask | check | eval
src/
  ingest.py            docx → chunks + references + inline-citation resolution
  embedder.py          local sentence-transformers wrapper (bge/e5 prefixes)
  index.py             ChromaDB dense index + BM25 sparse index
  retrieve.py          hybrid search, RRF fusion, optional rerank
  generate.py          citation-forcing prompt + baseline prompt + XML parsing
  verify.py            independent entailment verification (LLM / NLI / lexical)
  pipeline.py          ask() and check() orchestration
  llm.py               Claude wrapper with graceful no-key degradation
eval/run_eval.py       the evaluation harness → eval/reports/
eval/judge_set.py      builds the known-answer judge validation set
eval/judge_eval.py     scores judges / panels on it → eval/reports/judge_report.md
data/golden/           golden set (retrieval + trap questions)
```
