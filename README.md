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
ollama serve                 # if not already running
ollama pull llama3           # or a stronger instruct model, e.g. qwen2.5:7b-instruct
#   OLLAMA_MODEL=llama3:latest   OLLAMA_HOST=http://localhost:11434  (defaults)
```

**Option B — Claude API (higher quality on the structured prompts):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...          # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export LLM_BACKEND=claude                     # optional; auto-selected when key is set
```

> Note on local models: Llama 3 8B handles the citation-forcing prompt and the
> refusal behaviour well, but a larger instruct model (e.g. `qwen2.5:7b/14b`) is
> more reliable on the strict XML/JSON structure and on borderline entailment
> calls. Set `OLLAMA_MODEL` accordingly.

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

Any disagreement under `unanimous` yields `disputed` (⚖), which counts as
neither support nor hallucination and is the queue for human review. Leave
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
data/golden/           golden set (retrieval + trap questions)
```
