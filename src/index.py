"""Indexing Layer — embed chunks into ChromaDB + build a BM25 sparse index.

Dense (semantic) and sparse (lexical/BM25) indexes are kept side by side so the
retrieval layer can run true hybrid search. BM25 matters here because academic
prose is full of exact-match terms — "DiFuMo", "ISFC", "TFCE", "dISFC" — that a
semantic embedding will happily blur together.
"""
from __future__ import annotations

import json
import pickle
import re

import chromadb

import config
from src.embedder import Embedder
from src.ingest import Chunk, load_chunks


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _chroma_client():
    return chromadb.PersistentClient(path=str(config.INDEX_DIR))


def _bm25_path(strategy: str):
    return config.INDEX_DIR / f"bm25_{strategy}.pkl"


def build_index(strategy: str, embedder: Embedder | None = None) -> dict:
    chunks: list[Chunk] = load_chunks(strategy)
    embedder = embedder or Embedder()

    # ---- Dense index (Chroma) -------------------------------------------
    client = _chroma_client()
    name = config.collection_name(strategy)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [
        {
            "strategy": c.strategy,
            "section": c.section,
            "section_number": c.section_number,
            "para_start": c.para_start,
            "para_end": c.para_end,
            "est_tokens": c.est_tokens,
            "source": c.source,
            "citations": json.dumps(c.citations, ensure_ascii=False),
            "page_start": c.page_start,
            "page_end": c.page_end,
        }
        for c in chunks
    ]
    embeddings = embedder.encode_passages(docs)
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    # ---- Sparse index (BM25) --------------------------------------------
    from rank_bm25 import BM25Okapi

    corpus_tokens = [bm25_tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    with _bm25_path(strategy).open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)

    return {"strategy": strategy, "n_chunks": len(chunks), "collection": name}


def load_bm25(strategy: str):
    with _bm25_path(strategy).open("rb") as f:
        return pickle.load(f)


def get_collection(strategy: str):
    return _chroma_client().get_collection(config.collection_name(strategy))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build dense + sparse indexes")
    ap.add_argument("--strategy", choices=["fixed", "section"], default=config.CHUNK_STRATEGY)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    embedder = Embedder()
    strategies = ["fixed", "section"] if args.all else [args.strategy]
    for strat in strategies:
        info = build_index(strat, embedder)
        print(f"[index] {info}")
