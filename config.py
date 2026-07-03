"""Central configuration for the dissertation RAG Copilot.

All tunable knobs live here so experiments (chunking ablation, model swaps,
retrieval parameters) are reproducible and diff-able.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHUNK_DIR = DATA_DIR / "chunks"
GOLDEN_DIR = DATA_DIR / "golden"
INDEX_DIR = DATA_DIR / "index"          # ChromaDB persistent store lives here
REPORT_DIR = ROOT / "eval" / "reports"

# Source document (the dissertation manuscript).
SOURCE_DOCX = Path(
    os.environ.get(
        "DISSERTATION_DOCX",
        ROOT / "2026_Manuscripts_BXF+NTSEC_v2.1.docx",
    )
)
# Short human-readable handle used in citations.
SOURCE_LABEL = "Lee (2025) Dissertation"

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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

# Back-compat: some modules refer to LLM_MODEL as the active model name.
LLM_MODEL = CLAUDE_MODEL if LLM_BACKEND == "claude" else OLLAMA_MODEL

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
