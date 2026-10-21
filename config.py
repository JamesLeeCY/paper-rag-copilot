"""Central configuration for the dissertation RAG Copilot.

All tunable knobs live here so experiments (chunking ablation, model swaps,
retrieval parameters) are reproducible and diff-able.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines, # comments) — no dependency.

    Variables already set in the real environment win, so a shell export
    always overrides the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Which document collection to work on. "dissertation" (default) keeps the
# original layout; any other name gets its own data/index/report folders and
# reads papers/<name>.pdf (or SOURCE_PATH), so corpora never mix.
CORPUS = os.environ.get("CORPUS", "dissertation")

if CORPUS == "dissertation":
    DATA_DIR = ROOT / "data"
    REPORT_DIR = ROOT / "eval" / "reports"
    SOURCE_PATH = Path(
        os.environ.get(
            "DISSERTATION_DOCX",
            ROOT / "2026_Manuscripts_BXF+NTSEC_v2.1.docx",
        )
    )
    # Short human-readable handle used in citations.
    SOURCE_LABEL = "Lee (2025) Dissertation"
else:
    DATA_DIR = ROOT / "data" / "corpora" / CORPUS
    REPORT_DIR = ROOT / "eval" / "reports" / CORPUS
    SOURCE_PATH = Path(os.environ.get("SOURCE_PATH", ROOT / "papers" / f"{CORPUS}.pdf"))
    SOURCE_LABEL = os.environ.get("SOURCE_LABEL", CORPUS)

SOURCE_DOCX = SOURCE_PATH               # back-compat name
CHUNK_DIR = DATA_DIR / "chunks"
GOLDEN_DIR = DATA_DIR / "golden"
INDEX_DIR = DATA_DIR / "index"          # ChromaDB persistent store lives here

for _d in (DATA_DIR, CHUNK_DIR, GOLDEN_DIR, INDEX_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Two strategies are implemented so we can run the ablation the spec asks for.
#   "fixed"   -> fixed-size sliding window (baseline)
#   "section" -> section/structure-aware chunking
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "section")

FIXED_CHUNK_TOKENS = 320       # ~ target size for the baseline chunker
FIXED_CHUNK_OVERLAP = 64       # token overlap between adjacent fixed chunks
SECTION_MAX_TOKENS = 380       # a section chunk is split if it exceeds this
SECTION_MIN_TOKENS = 40        # tiny trailing paragraphs get merged forward

# Rough token estimate: words * this factor. Avoids a tokenizer dependency for
# chunk sizing (embedding model still tokenizes internally).
TOKENS_PER_WORD = 1.3

# --------------------------------------------------------------------------
# Embedding model (local, via sentence-transformers)
# --------------------------------------------------------------------------
# bge-m3 (multilingual, ~2.3 GB) is the default since 2026-10-21: on the four
# corpora it matches bge-small on English questions (Hit@5 52 vs 53 of 56) and
# lifts Chinese questions from 52% to 88% Hit@5 (with CJK_QUERY_DENSE_ONLY);
# a full thesis run was as good or better on every measure. Results reported
# before that date used "BAAI/bge-small-en-v1.5" (smaller, English only), which
# is still available. Each model has its own index (`cli.py build`).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH = 32
EMBED_NORMALIZE = True
# Retrieval query instruction. bge-v1.5 recommends this prefix on queries only.
# e5 models instead want "query: " / "passage: " prefixes (see _embed_prefixes).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# BM25 cannot match a Chinese question against an English source, and fusing
# its noise into the ranking hurt a multilingual embedder (bge-m3, Chinese
# questions, Hit@5 73% hybrid vs 88% dense-only over four corpora,
# 2026-10-21). "auto": a query containing CJK characters uses dense retrieval
# only when EMBED_MODEL is multilingual (bge-m3 / multilingual); "on" always,
# "off" never.
CJK_QUERY_DENSE_ONLY = os.environ.get("CJK_QUERY_DENSE_ONLY", "auto")

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DENSE_TOP_K = 20               # candidates from dense search
SPARSE_TOP_K = 20             # candidates from BM25
RRF_K = 60                    # reciprocal-rank-fusion constant
FUSED_TOP_K = 20             # candidates kept after fusion, fed to reranker
FINAL_TOP_K = 5              # passages returned to the generator
# Passages each sentence is judged against in `check` (retrieval order, early
# stop on support). Measured on 40 labelled sentences (eval.check_cost,
# 2026-10-17): 2 instead of 3 kept false accepts at 0/18, added one false
# reject (1/22; its evidence was ranked third) and cut ~146 s to ~111 s per
# sentence; 1 passage cost 3/22 false rejects. Embedding-similarity order was
# identical to retrieval order on all 40, so it is not used.
CHECK_TOP_N = int(os.environ.get("CHECK_TOP_N", "2"))

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "0") == "1"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")

# --------------------------------------------------------------------------
# LLM (generation + verification) — pluggable backend
# --------------------------------------------------------------------------
# Backend selection:
#   "claude" -> Anthropic API (needs ANTHROPIC_API_KEY)
#   "ollama" -> local Ollama server (free, offline)
# Default: use Claude if a key is present, otherwise fall back to local Ollama.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_BACKEND = os.environ.get(
    "LLM_BACKEND", "claude" if ANTHROPIC_API_KEY else "ollama"
)

LLM_MAX_TOKENS = 1600
LLM_TEMPERATURE = 0.0

# -- Claude backend --
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# -- Ollama (local) backend --
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# qwen2.5 follows the citation/quote XML far more reliably than llama3
# (5/5 verbatim quotes vs 2/4 in the comparison run).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
# Ollama 0.35+ returns a reasoning model's thinking in a separate field, and
# those tokens count toward num_predict: deepseek-r1 spent all 300 judge tokens
# thinking and returned empty content (`think: false` is ignored by it). Models
# reporting the "thinking" capability get this many extra tokens. Measured:
# ~1,200 tokens (~4 min on CPU) for one deepseek-r1:8b judgment.
OLLAMA_THINK_BUDGET = int(os.environ.get("OLLAMA_THINK_BUDGET", "2000"))
# Request timeout scales with the work: max(300 s, 60 s + (output limit +
# half the estimated prompt tokens) x this); see llm.request_timeout.
# CPU-only generation measured ~5 tokens/s (0.2 s/token); 0.4 leaves a 2x
# margin. A fixed 300 s cut off a deepseek-r1 judgment still thinking, and an
# output-only rule let phi4 time out reading long passages on 3 threads.
OLLAMA_SECONDS_PER_TOKEN = float(os.environ.get("OLLAMA_SECONDS_PER_TOKEN", "0.4"))
# CPU threads Ollama may use for this project's requests (sent per request as
# the num_thread option, so other projects sharing the server are unaffected).
# 0 = Ollama's default (all physical cores). Set e.g. 3 to leave room for other
# jobs; generation and judging slow down roughly in proportion.
OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "0"))
# Context window per request (num_ctx); 0 = Ollama's default. Ollama keeps one
# runner per model file and reloads it whenever a request's num_thread or
# num_ctx differs from the loaded runner's, so when another project uses the
# same model, matching its settings lets requests queue instead of reloading.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "0"))

# Back-compat: some modules refer to LLM_MODEL as the active model name.
LLM_MODEL = CLAUDE_MODEL if LLM_BACKEND == "claude" else OLLAMA_MODEL

# -- Verifier panel (multi-model cross-check) --
# Comma-separated "backend:model" judges, e.g.
#   JUDGES="ollama:llama3:latest,ollama:qwen2.5:7b-instruct"
# Empty -> a single judge on the generator's own backend (original behaviour).
# Judges from different model families make correlated errors less likely.
# Default on the Ollama backend: phi4 alone, selected on the judge validation
# set under prompt v3 (held-out split: dissertation 5% false-accept / 7%
# false-reject, pilot paper 6% / 5%, 0% disputed). It also takes the generator
# model (qwen2.5) off the judge seat; qwen2.5 was the weakest judge and had
# approved its own errors. For fewer false-accepts at the cost of a human-review
# queue (13-24% disputed), use JUDGES="ollama:phi4:latest,ollama:gemma3:12b".
# Judges whose model is not pulled are skipped.
DEFAULT_OLLAMA_JUDGES = "ollama:phi4:latest"
JUDGES = [
    j.strip()
    for j in os.environ.get(
        "JUDGES", DEFAULT_OLLAMA_JUDGES if LLM_BACKEND == "ollama" else ""
    ).split(",")
    if j.strip()
]
# How panel votes combine: "unanimous" -> any disagreement is "disputed";
# "majority" -> a strict majority label wins, otherwise "disputed".
PANEL_RULE = os.environ.get("PANEL_RULE", "unanimous")
# Judge prompt version (see src/verify.py): "v1" is the original prompt, "v2"
# adds an explicit evidence-strength (overclaim) check, "v3" adds the
# plan-as-result and misattribution checks. v3 is the default because the
# default judge (phi4) was validated under it.
VERIFY_PROMPT = os.environ.get("VERIFY_PROMPT", "v3")

# -- Quote grounding --
# The generator attaches a verbatim <quote> from the cited passage to every
# claim; quotes are string-matched against the passage before any LLM judge.
# Share of quote characters that must align with the passage to count as
# near-verbatim (tolerates small whitespace/punctuation drift).
QUOTE_MATCH_THRESHOLD = 0.9
# If True, a claim without a <quote> is marked unsupported outright.
QUOTE_REQUIRED = os.environ.get("QUOTE_REQUIRED", "0") == "1"
# Rule-based check for a plan or prediction read as a result (source sentence
# in future / hypothesis form, claim asserting a finding). Judges have approved
# this error, so it is checked without them. "flag" turns an accepted claim
# into "disputed" (human review), "reject" into "unsupported", "off" disables.
PLAN_RESULT_RULE = os.environ.get("PLAN_RESULT_RULE", "flag")
# Hypothesis-match route of that rule (opt-in): a claim the judge rated only
# partially supported is flagged when it is closer to a hypothesis sentence
# among the retrieved passages than to its cited passage -- a hypothesis
# restated as a finding that cites a results passage, which neither the
# structural nor the wording route sees. Uses a multilingual embedder (claims
# are often Chinese, hypotheses English), loaded only when a Hypotheses
# section is in the context. Thresholds were set on the dissertation's saved
# outputs: it flags the one real case and none of the other 30 partially
# supported claims; not yet checked on new outputs. "off" disables it.
PLAN_RESULT_MATCH = os.environ.get("PLAN_RESULT_MATCH", "on")
PLAN_MATCH_MODEL = os.environ.get("PLAN_MATCH_MODEL", "BAAI/bge-m3")
PLAN_MATCH_MIN_SIM = float(os.environ.get("PLAN_MATCH_MIN_SIM", "0.55"))
PLAN_MATCH_MARGIN = float(os.environ.get("PLAN_MATCH_MARGIN", "-0.02"))
# Generator prompt version (src/generate.py). "v2" adds: quotes in the source
# language, hypotheses/methods never stated as findings (with passage-type
# labels in the context), cited studies attributed explicitly, negative facts
# answered, the refusal marker only in <unsupported_note>. "v1" is the
# validated original; v2 was not adopted (more format errors). "v2.1" keeps
# v2's source-language quotes and passage-type rule, drops its attribution
# and negative-fact rules, refuses with a separate <refusal> tag, and shows a
# two-example template with an obviously fake id instead of a placeholder.
# v2.1 became the default on 2026-10-19 after a paired full run against v1
# (dissertation, Ollama 0.40.5): no misplaced refusal markers (v1: 16), traps
# strict-safe 36/37 vs 30/37, hypothesis restated by the generator on 1 bait
# item vs 4, no unsupported answerable claim (v1: 1), ~15% faster; strict
# precision 25/28 vs 29/30 (three partially supported claims). Results
# reported before that date were produced with v1.
GENERATOR_PROMPT = os.environ.get("GENERATOR_PROMPT", "v2.1")
# Order of the passages shown to the grounded generator. "rank" keeps the
# retrieval order; "results_first" puts results and discussion passages
# first and hypothesis passages last (within a type, retrieval order), to
# test whether a small generator answers result questions from a retrieved
# hypothesis passage because it comes first. Retrieval, citations and
# verification are unaffected.
CONTEXT_ORDER = os.environ.get("CONTEXT_ORDER", "rank")

# The exact string the model must emit when nothing supports a claim. The
# evaluation harness matches on this to score refusal correctness.
REFUSAL_MARKER = "查無直接支持此說法的段落"

# --------------------------------------------------------------------------
# Chroma collection naming (strategy-scoped so both indexes can coexist)
# --------------------------------------------------------------------------
def collection_name(strategy: str | None = None) -> str:
    strategy = strategy or CHUNK_STRATEGY
    safe_model = EMBED_MODEL.split("/")[-1].replace(".", "_")
    return f"diss_{strategy}_{safe_model}"


def chunks_path(strategy: str | None = None) -> Path:
    strategy = strategy or CHUNK_STRATEGY
    return CHUNK_DIR / f"chunks_{strategy}.jsonl"
