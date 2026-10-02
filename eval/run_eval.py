"""Evaluation Harness — the artefact that turns this from a toy into a tool.

Produces quantified, reproducible metrics over the golden set:

  * Retrieval Hit Rate@k   — did the passage that answers the question appear
                             in the top-k results? (strategy-agnostic: matches
                             on docx paragraph spans)
  * MRR                    — mean reciprocal rank of the first correct hit
  * Chunking ablation      — fixed-size vs section-aware, same golden set
  * Citation Precision     — (needs API key) share of generated claims whose
                             cited passage actually entails them
  * Hallucination Rate     — (needs API key) share of unsupported claims
  * Refusal Correctness    — (needs API key) share of trap questions correctly
                             answered with the refusal marker

Run:  python -m eval.run_eval            # all strategies, retrieval metrics
      python -m eval.run_eval --with-llm # also grounding/hallucination/refusal
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import config
from src.retrieve import Retriever
from src.llm import LLMClient

GOLDEN = config.GOLDEN_DIR / "golden_set.json"
K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)


def load_golden() -> dict:
    with GOLDEN.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_hit_rank(passages, target_paras: list[int]) -> int | None:
    """1-based rank of the first passage whose paragraph span covers a target."""
    tset = set(target_paras)
    for rank, p in enumerate(passages, 1):
        span = range(p.para_start, p.para_end + 1)
        if any(t in span for t in tset):
            return rank
    return None


# --------------------------------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------------------------------
def eval_retrieval(strategy: str, golden: dict, use_rerank: bool) -> dict:
    retriever = Retriever(strategy)
    items = golden["retrieval"]
    ranks: list[int | None] = []
    per_item = []
    for it in items:
        passages = retriever.search(it["question"], top_k=MAX_K, use_rerank=use_rerank)
        rank = _first_hit_rank(passages, it["target_paras"])
        ranks.append(rank)
        per_item.append(
            {
                "id": it["id"],
                "rank": rank,
                "top1": passages[0].chunk_id if passages else None,
                "top1_section": passages[0].section_number if passages else None,
            }
        )

    n = len(items)
    hit_at = {
        k: round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
        for k in K_VALUES
    }
    rr = [1.0 / r if r else 0.0 for r in ranks]
    return {
        "strategy": strategy,
        "rerank": use_rerank,
        "n": n,
        "hit_at": hit_at,
        "mrr": round(statistics.mean(rr), 4),
        "misses": [p["id"] for p in per_item if p["rank"] is None],
        "per_item": per_item,
    }


# --------------------------------------------------------------------------
# Generation-grounded evaluation (needs LLM)
# --------------------------------------------------------------------------
def eval_grounding(strategy: str, golden: dict, llm: LLMClient, limit: int | None = None) -> dict:
    from src.pipeline import ask
    from src.verify import Verifier

    retrieval_items = golden["retrieval"][:limit] if limit else golden["retrieval"]
    trap_items = golden["traps"][:limit] if limit else golden["traps"]

    verifier = Verifier(llm=llm)
    precisions, hallucs, n_claims = [], [], 0
    parse_failures = 0
    for it in retrieval_items:
        bundle = ask(it["question"], strategy=strategy, llm=llm, verifier=verifier)
        parse_failures += int(bundle.generation.parse_failed)
        v = bundle.verification
        if v.n_claims:
            precisions.append(v.citation_precision())
            hallucs.append(v.hallucination_rate())
            n_claims += v.n_claims

    # Trap questions: system should refuse.
    refused = 0
    trap_detail = []
    for it in trap_items:
        bundle = ask(it["question"], strategy=strategy, llm=llm, verifier=verifier)
        # Malformed output (no claims, no marker) is NOT a correct refusal.
        did_refuse = bundle.generation.refused
        parse_failures += int(bundle.generation.parse_failed)
        refused += int(did_refuse)
        trap_detail.append({
            "id": it["id"],
            "refused": did_refuse,
            "parse_failed": bundle.generation.parse_failed,
        })

    n_traps = len(trap_items)
    return {
        "strategy": strategy,
        "n_retrieval_scored": len(retrieval_items),
        "n_traps_scored": n_traps,
        "n_claims_total": n_claims,
        "citation_precision": round(statistics.mean(precisions), 4) if precisions else None,
        "hallucination_rate": round(statistics.mean(hallucs), 4) if hallucs else None,
        "refusal_correctness": round(refused / n_traps, 4) if n_traps else None,
        "parse_failures": parse_failures,
        "trap_detail": trap_detail,
    }


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------
def write_report(results: dict, path: Path) -> None:
    lines = []
    A = lines.append
    A("# Dissertation RAG Copilot — Evaluation Report")
    A("")
    A(f"- Generated: {results['generated_at']}")
    A(f"- Embedding model: `{results['embed_model']}`")
    A(f"- Rerank enabled: `{results['rerank']}`")
    A(f"- Golden set: {results['n_retrieval']} retrieval questions, "
      f"{results['n_traps']} trap questions")
    A("")

    A("## 1. Retrieval Hit Rate (chunking ablation)")
    A("")
    header = "| Strategy | Rerank | " + " | ".join(f"Hit@{k}" for k in K_VALUES) + " | MRR |"
    A(header)
    A("|" + "---|" * (len(K_VALUES) + 3))
    for r in results["retrieval"]:
        cells = " | ".join(f"{r['hit_at'][k]:.0%}" for k in K_VALUES)
        A(f"| {r['strategy']} | {r['rerank']} | {cells} | {r['mrr']:.3f} |")
    A("")
    # Target line from the spec
    A("> Target (spec §1.3): **Hit Rate@k ≥ 90%**.")
    A("")

    best = max(results["retrieval"], key=lambda r: r["hit_at"][5])
    A(f"**Best configuration by Hit@5:** `{best['strategy']}` "
      f"(rerank={best['rerank']}) → Hit@5 = {best['hit_at'][5]:.0%}, MRR = {best['mrr']:.3f}")
    A("")
    if best["misses"]:
        A(f"Misses (not in top-{MAX_K}): {', '.join(best['misses'])}")
        A("")

    if results.get("grounding"):
        A(f"## 2. Grounding / Hallucination / Refusal (LLM: `{results.get('llm_backend','?')}`)")
        A("")
        g0 = results["grounding"][0]
        A(f"_Scored on {g0.get('n_retrieval_scored','?')} retrieval questions and "
          f"{g0.get('n_traps_scored','?')} trap questions"
          + (" (sampled subset)." if results.get("llm_limit") else ".") + "_")
        A("")
        A("| Strategy | Citation Precision | Hallucination Rate | Refusal Correctness "
          "| Claims | Parse Failures |")
        A("|---|---|---|---|---|---|")
        for g in results["grounding"]:
            cp = f"{g['citation_precision']:.0%}" if g["citation_precision"] is not None else "n/a"
            hr = f"{g['hallucination_rate']:.0%}" if g["hallucination_rate"] is not None else "n/a"
            rc = f"{g['refusal_correctness']:.0%}" if g["refusal_correctness"] is not None else "n/a"
            A(f"| {g['strategy']} | {cp} | {hr} | {rc} | {g['n_claims_total']} "
              f"| {g.get('parse_failures', 0)} |")
        A("")
        A("> Parse failures = outputs with neither a `<claim>` nor the refusal marker. "
          "They are not counted as refusals.")
        A("")
        A("> Targets (spec §1.3): Citation grounding ≥ **95%** supported; "
          "trap refusal rate ≥ **90%**.")
        A("")
    else:
        A("## 2. Grounding / Hallucination / Refusal (LLM)")
        A("")
        A("_Skipped — run `python -m eval.run_eval --with-llm` with an LLM backend "
          "(Claude via `ANTHROPIC_API_KEY`, or a running local Ollama) to populate "
          "citation precision, hallucination rate, and trap refusal correctness._")
        A("")

    A("## 3. Per-question retrieval detail (best config)")
    A("")
    A("| ID | First-hit rank | Top-1 section | Top-1 chunk |")
    A("|---|---|---|---|")
    for p in best["per_item"]:
        A(f"| {p['id']} | {p['rank'] if p['rank'] else 'MISS'} | "
          f"{p['top1_section'] or ''} | `{p['top1'] or ''}` |")
    A("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run the evaluation harness")
    ap.add_argument("--strategies", nargs="+", default=["fixed", "section"])
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder rerank")
    ap.add_argument("--with-llm", action="store_true",
                    help="also run grounding/hallucination/refusal (needs an LLM backend)")
    ap.add_argument("--llm-strategy", default="section",
                    help="chunking strategy used for the (slow) LLM grounding pass")
    ap.add_argument("--llm-limit", type=int, default=None,
                    help="cap #retrieval and #trap questions for the LLM pass (quick runs)")
    args = ap.parse_args()

    golden = load_golden()
    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embed_model": config.EMBED_MODEL,
        "rerank": args.rerank,
        "n_retrieval": len(golden["retrieval"]),
        "n_traps": len(golden["traps"]),
        "retrieval": [],
    }

    for strat in args.strategies:
        r = eval_retrieval(strat, golden, use_rerank=args.rerank)
        results["retrieval"].append(r)
        print(f"[retrieval:{strat}] Hit@1={r['hit_at'][1]:.0%} "
              f"Hit@3={r['hit_at'][3]:.0%} Hit@5={r['hit_at'][5]:.0%} MRR={r['mrr']:.3f} "
              f"misses={r['misses']}")

    if args.with_llm:
        llm = LLMClient()
        if not llm.available:
            print(f"[warn] --with-llm requested but LLM backend "
                  f"'{llm.backend}' unavailable ({llm.describe()}); skipping.")
        else:
            print(f"[grounding] using LLM backend = {llm.describe()} "
                  f"(strategy={args.llm_strategy}, limit={args.llm_limit})")
            results["llm_backend"] = llm.describe()
            results["llm_limit"] = args.llm_limit
            g = eval_grounding(args.llm_strategy, golden, llm, limit=args.llm_limit)
            results["grounding"] = [g]
            print(f"[grounding:{args.llm_strategy}] citation_precision={g['citation_precision']} "
                  f"halluc={g['hallucination_rate']} refusal={g['refusal_correctness']} "
                  f"parse_failures={g['parse_failures']}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(results, config.REPORT_DIR / "eval_report.md")
    print(f"\nReport written to {config.REPORT_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main()
