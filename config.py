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
# bge-small keeps first-run download small (~130MB) and is fast on CPU.
# Swap to "BAAI/bge-large-en-v1.5" or "intfloat/e5-large-v2" for higher quality.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH = 32
EMBED_NORMALIZE = True
# Retrieval query instruction. bge-v1.5 recommends this prefix on queries only.
# e5 models instead want "query: " / "passage: " prefixes (see _embed_prefixes).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DENSE_TOP_K = 20               # candidates from dense search
SPARSE_TOP_K = 20             # candidates from BM25
RRF_K = 60                    # reciprocal-rank-fusion constant
FUSED_TOP_K = 20             # candidates kept after fusion, fed to reranker
FINAL_TOP_K = 5              # passages returned to the generator
CHECK_TOP_N = 3              # passages each sentence is judged against in `check`

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
# Request timeout scales with the token limit: max(300 s, 60 s + limit x this).
# CPU-only generation measured ~5 tokens/s (0.2 s/token); 0.4 leaves a 2x
# margin. A fixed 300 s cut off a deepseek-r1 judgment still thinking.
OLLAMA_SECONDS_PER_TOKEN = float(os.environ.get("OLLAMA_SECONDS_PER_TOKEN", "0.4"))
# CPU threads Ollama may use for this project's requests (sent per request as
# the num_thread option, so other projects sharing the server are unaffected).
# 0 = Ollama's default (all physical cores). Set e.g. 3 to leave room for other
# jobs; generation and judging slow down roughly in proportion.
OLLAMA_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", "0"))

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
