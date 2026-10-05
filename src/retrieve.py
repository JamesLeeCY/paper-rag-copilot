"""Retrieval Layer — hybrid (dense + BM25) search, RRF fusion, optional rerank.

Pipeline: query -> {dense top-k, sparse top-k} -> reciprocal-rank fusion ->
optional cross-encoder rerank -> final top-k passages with full metadata.

Reciprocal Rank Fusion (RRF) is used to merge the two candidate lists because
it needs no score calibration between the (cosine) dense scores and the
(unbounded) BM25 scores — it only looks at ranks.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass

import config
from src.embedder import Embedder
from src.index import get_collection, load_bm25, bm25_tokenize


@dataclass
class Passage:
    chunk_id: str
    text: str
    section: str
    section_number: str
    para_start: int
    para_end: int
    source: str
    citations: list
    score: float          # fused (or rerank) score
    dense_rank: int | None = None
    sparse_rank: int | None = None
    page_start: int = 0
    page_end: int = 0

    def locator(self) -> str:
        sec = f"§{self.section_number}" if self.section_number else self.section[:40]
        if self.page_start:
            pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                     else f"pp. {self.page_start}-{self.page_end}")
            return f"{self.source}, {sec} ({pages}) [{self.chunk_id}]"
        return f"{self.source}, {sec} (paras {self.para_start}-{self.para_end}) [{self.chunk_id}]"


class Retriever:
    def __init__(self, strategy: str | None = None, embedder: Embedder | None = None):
        self.strategy = strategy or config.CHUNK_STRATEGY
        self.embedder = embedder or Embedder()
        self.coll = get_collection(self.strategy)
        bm = load_bm25(self.strategy)
        self.bm25 = bm["bm25"]
        self.bm25_ids = bm["ids"]
        self._reranker = None

    # -- candidate generation ------------------------------------------------
    def _dense(self, query: str, k: int) -> list[str]:
        qvec = self.embedder.encode_queries([query])[0]
        res = self.coll.query(query_embeddings=[qvec], n_results=k)
        return res["ids"][0]

    def _sparse(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25_ids[i] for i in order[:k]]

    # -- fusion --------------------------------------------------------------
    @staticmethod
    def _rrf(dense: list[str], sparse: list[str], k: int) -> dict[str, dict]:
        fused: dict[str, dict] = {}
        for rank, cid in enumerate(dense):
            fused.setdefault(cid, {"score": 0.0, "dense_rank": None, "sparse_rank": None})
            fused[cid]["score"] += 1.0 / (config.RRF_K + rank + 1)
            fused[cid]["dense_rank"] = rank
        for rank, cid in enumerate(sparse):
            fused.setdefault(cid, {"score": 0.0, "dense_rank": None, "sparse_rank": None})
            fused[cid]["score"] += 1.0 / (config.RRF_K + rank + 1)
            fused[cid]["sparse_rank"] = rank
        return fused

    # -- rerank --------------------------------------------------------------
    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def _hydrate(self, ids: list[str]) -> dict[str, dict]:
        got = self.coll.get(ids=ids, include=["documents", "metadatas"])
        out = {}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            out[cid] = {"text": doc, "meta": meta}
        return out

    # -- public --------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_hybrid: bool = True,
        use_rerank: bool | None = None,
    ) -> list[Passage]:
        top_k = top_k or config.FINAL_TOP_K
        use_rerank = config.RERANK_ENABLED if use_rerank is None else use_rerank

        dense = self._dense(query, config.DENSE_TOP_K)
        sparse = self._sparse(query, config.SPARSE_TOP_K) if use_hybrid else []
        fused = self._rrf(dense, sparse, config.RRF_K)

        ranked = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)
        cand_ids = [cid for cid, _ in ranked[: config.FUSED_TOP_K]]
        hydrated = self._hydrate(cand_ids)

        if use_rerank and cand_ids:
            reranker = self._get_reranker()
            pairs = [(query, hydrated[cid]["text"]) for cid in cand_ids]
            rr_scores = reranker.predict(pairs)
            order = sorted(range(len(cand_ids)), key=lambda i: rr_scores[i], reverse=True)
            cand_ids = [cand_ids[i] for i in order]
            score_by_id = {cand_ids[i]: float(rr_scores[order[i]]) for i in range(len(cand_ids))}
        else:
            score_by_id = {cid: fused[cid]["score"] for cid in cand_ids}

        passages: list[Passage] = []
        for cid in cand_ids[:top_k]:
            meta = hydrated[cid]["meta"]
            passages.append(
                Passage(
                    chunk_id=cid,
                    text=hydrated[cid]["text"],
                    section=meta.get("section", ""),
                    section_number=meta.get("section_number", ""),
                    para_start=meta.get("para_start", -1),
                    para_end=meta.get("para_end", -1),
                    source=meta.get("source", config.SOURCE_LABEL),
                    citations=json.loads(meta.get("citations", "[]")),
                    score=score_by_id[cid],
                    dense_rank=fused[cid]["dense_rank"],
                    sparse_rank=fused[cid]["sparse_rank"],
                    page_start=meta.get("page_start", 0),
                    page_end=meta.get("page_end", 0),
                )
            )
        return passages


@functools.lru_cache(maxsize=2)
def get_retriever(strategy: str) -> Retriever:
    return Retriever(strategy)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ad-hoc retrieval test")
    ap.add_argument("query")
    ap.add_argument("--strategy", default=config.CHUNK_STRATEGY)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    r = Retriever(args.strategy)
    for i, p in enumerate(r.search(args.query, top_k=args.k), 1):
        print(f"\n#{i}  score={p.score:.4f}  {p.locator()}")
        print("   ", p.text[:220].replace("\n", " "))
