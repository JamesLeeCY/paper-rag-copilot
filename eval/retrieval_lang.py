"""Retrieval hit rate for English vs Chinese questions with the current EMBED_MODEL.

Retrieval items in a corpus's golden set may carry a `question_zh` (the same
information need in Traditional Chinese). This runs the retrieval benchmark
twice — original questions, then the Chinese ones — so embedding models can be
compared on cross-lingual queries over an English source. The index for
EMBED_MODEL must exist (`cli.py build` with that EMBED_MODEL).

Run: CORPUS=<name> EMBED_MODEL=<model> python -m eval.retrieval_lang
"""
from __future__ import annotations

import copy
import json

import config
from eval.run_eval import eval_retrieval, load_golden


def main() -> None:
    golden = load_golden()
    zh = copy.deepcopy(golden)
    zh["retrieval"] = [dict(it, question=it["question_zh"]) for it in golden["retrieval"]
                       if it.get("question_zh")]
    out = {"corpus": config.CORPUS, "model": config.EMBED_MODEL}
    for lang, g in (("en", golden), ("zh", zh)):
        if not g["retrieval"]:
            continue
        r = eval_retrieval(config.CHUNK_STRATEGY, g, use_rerank=False)
        out[lang] = {"n": len(g["retrieval"]), "hit_at": r["hit_at"], "mrr": r["mrr"],
                     "misses": r["misses"]}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
