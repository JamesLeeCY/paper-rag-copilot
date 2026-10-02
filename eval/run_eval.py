"""Evaluation Harness — the artefact that turns this from a toy into a tool.

Produces quantified, reproducible metrics over the golden set:

  * Retrieval Hit Rate@k   — did the passage that answers the question appear
                             in the top-k results? (strategy-agnostic: matches
                             on docx paragraph spans)
  * MRR                    — mean reciprocal rank of the first correct hit
  * Chunking ablation      — fixed-size vs section-aware, same golden set
  * Citation Precision     — (needs LLM) share of generated claims whose cited
                             passage entails them; strict = supported only,
                             lenient = supported + partially_supported
  * Hallucination Rate     — (needs LLM) unsupported claims / total claims

  Grounding metrics are micro-averaged over all claims pooled across
  questions, so every claim carries equal weight.
  * Refusal Correctness    — (needs LLM) share of trap questions correctly
                             answered with the refusal marker
  * Over-refusal Rate      — (needs LLM) share of answerable questions wrongly
                             refused; read together with refusal correctness
  * Answer Hallucination   — (needs LLM) share of non-refused answers with at
                             least one unsupported claim

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


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


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
    print(f"[grounding] verifier {verifier.describe()}")
    # Pool claim counts across questions (micro-average) rather than averaging
    # per-question rates, which would down-weight questions with many claims.
    n_claims = n_supported = n_partial = n_unsupported = 0
    # Cross-check tallies: disputed verdicts, quote-grounding outcomes, and how
    # often panel judges agreed (claims judged by >= 2 judges).
    n_disputed = n_panel_judged = n_panel_agreed = 0
    n_repaired = 0  # mangled citation ids recovered via the verbatim quote
    quote_counts = {s: 0 for s in ("verbatim", "near", "not_found", "missing")}
    parse_failures = 0
    # Answer-level counts on answerable questions. Every retrieval question has
    # a known answer in the corpus, so refusing one is an over-refusal — the
    # counterweight that stops "refuse everything" from scoring perfectly.
    n_over_refused = n_answered = n_answers_with_halluc = 0
    answer_detail = []
    for it in retrieval_items:
        bundle = ask(it["question"], strategy=strategy, llm=llm, verifier=verifier)
        g, v = bundle.generation, bundle.verification
        parse_failures += int(g.parse_failed)
        n_over_refused += int(g.refused)
        if v.n_claims:
            n_answered += 1
            n_answers_with_halluc += int(v.n_unsupported > 0)
        n_claims += v.n_claims
        n_supported += v.n_supported
        n_partial += v.n_partial
        n_unsupported += v.n_unsupported
        n_disputed += v.n_disputed
        n_repaired += v.n_citation_repaired
        n_panel_judged += v.n_panel_judged
        n_panel_agreed += v.n_panel_agreed
        for status in quote_counts:
            quote_counts[status] += v.n_quote(status)
        answer_detail.append({
            "id": it["id"],
            "refused": g.refused,
            "parse_failed": g.parse_failed,
            "n_claims": v.n_claims,
            "n_unsupported": v.n_unsupported,
            "n_disputed": v.n_disputed,
        })

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
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "n_disputed": n_disputed,
        "verifier": verifier.describe(),
        "quote_counts": quote_counts,
        # Claims whose quote is absent from the cited passage: fabricated citations.
        "fabricated_quote_rate": _ratio(quote_counts["not_found"], n_claims),
        "panel_agreement_rate": _ratio(n_panel_agreed, n_panel_judged),
        "n_citation_repaired": n_repaired,
        "citation_precision_strict": _ratio(n_supported, n_claims),
        "citation_precision_lenient": _ratio(n_supported + n_partial, n_claims),
        "hallucination_rate": _ratio(n_unsupported, n_claims),
        "refusal_correctness": round(refused / n_traps, 4) if n_traps else None,
        "over_refusal_rate": _ratio(n_over_refused, len(retrieval_items)),
        "n_answered": n_answered,
        # Share of non-refused answers containing at least one unsupported claim.
        "answer_hallucination_rate": _ratio(n_answers_with_halluc, n_answered),
        "parse_failures": parse_failures,
        "answer_detail": answer_detail,
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
        def pct(x):
            return f"{x:.0%}" if x is not None else "n/a"

        A("### 2a. Claim level")
        A("")
        A("| Strategy | Citation Precision (strict) | Citation Precision (lenient) "
          "| Hallucination Rate | Claims (✔/◐/⚖/✘) |")
        A("|---|---|---|---|---|")
        for g in results["grounding"]:
            counts = (f"{g['n_claims_total']} ({g['n_supported']}/{g['n_partial']}/"
                      f"{g['n_disputed']}/{g['n_unsupported']})")
            A(f"| {g['strategy']} | {pct(g['citation_precision_strict'])} "
              f"| {pct(g['citation_precision_lenient'])} | {pct(g['hallucination_rate'])} "
              f"| {counts} |")
        A("")
        A("> Strict = `supported` only; lenient = `supported` + `partially_supported`. "
          "`disputed` (judges disagree) counts as neither support nor hallucination. "
          "Micro-averaged over claims pooled across answerable questions.")
        A("")
        A("### 2a′. Cross-check (quote grounding + judge panel)")
        A("")
        A("| Strategy | Verifier | Quotes (verbatim/near/not found/missing) "
          "| Fabricated Quote Rate | Panel Agreement | Repaired Citation IDs |")
        A("|---|---|---|---|---|---|")
        for g in results["grounding"]:
            q = g["quote_counts"]
            A(f"| {g['strategy']} | {g['verifier']} "
              f"| {q['verbatim']}/{q['near']}/{q['not_found']}/{q['missing']} "
              f"| {pct(g['fabricated_quote_rate'])} | {pct(g['panel_agreement_rate'])} "
              f"| {g.get('n_citation_repaired', 0)} |")
        A("")
        A("> A quote not found in its cited passage rejects the claim before any LLM "
          "judge runs. Panel agreement = share of claims judged by ≥ 2 judges on "
          "which all votes matched (n/a with a single judge).")
        A("")
        A("### 2b. Answer level (refusal behaviour)")
        A("")
        A("| Strategy | Refusal Correctness (traps) | Over-refusal Rate (answerable) "
          "| Answer Hallucination Rate | Answered | Parse Failures |")
        A("|---|---|---|---|---|---|")
        for g in results["grounding"]:
            A(f"| {g['strategy']} | {pct(g['refusal_correctness'])} "
              f"| {pct(g['over_refusal_rate'])} | {pct(g['answer_hallucination_rate'])} "
              f"| {g['n_answered']}/{g['n_retrieval_scored']} | {g['parse_failures']} |")
        A("")
        A("> Refusal correctness and over-refusal must be read together: a system "
          "that refuses everything scores 100% on the first and 100% (worst) on the "
          "second. Answer hallucination rate = share of non-refused answers with at "
          "least one unsupported claim.")
        A("")
        A("> Parse failures = outputs with neither a `<claim>` nor the refusal marker. "
          "They are not counted as refusals.")
        A("")
        A("> Targets (spec §1.3): Citation grounding ≥ **95%** supported (strict); "
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
            print(f"[grounding:{args.llm_strategy}] precision strict/lenient="
                  f"{g['citation_precision_strict']}/{g['citation_precision_lenient']} "
                  f"halluc={g['hallucination_rate']} "
                  f"answer_halluc={g['answer_hallucination_rate']} "
                  f"refusal={g['refusal_correctness']} over_refusal={g['over_refusal_rate']} "
                  f"parse_failures={g['parse_failures']}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(results, config.REPORT_DIR / "eval_report.md")
    print(f"\nReport written to {config.REPORT_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main()
