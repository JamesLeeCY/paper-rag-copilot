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
import hashlib
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import config
from eval import stats
from src.retrieve import Retriever
from src.llm import LLMClient

GOLDEN = config.GOLDEN_DIR / "golden_set.json"
# Trap types whose only correct answer is a refusal (see eval_grounding).
MUST_REFUSE = {"far_absent", "near_absent"}
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
def _claim_detail(g, v) -> list[dict]:
    """Per-claim record (claim, quote, cited passage, verdict, votes, reason) so
    a verdict can be audited after the run. Verdicts align with claims by index.
    Contains corpus text: eval_results.json stays local (reports are gitignored)."""
    from src.verify import _passage_text

    out = []
    for claim, verdict in zip(g.claims, v.verdicts):
        out.append({
            "statement": claim.statement,
            "quote": claim.quote,
            "cited": claim.citation_ids,
            "judged_against": verdict.citation_ids,
            "passages": {cid: (_passage_text(g, cid) or "")[:2000]
                         for cid in verdict.citation_ids},
            "label": verdict.label,
            "method": verdict.method,
            "quote_check": verdict.quote_check,
            "votes": verdict.votes,
            "reason": verdict.reason,
        })
    return out


PROGRESS_NAME = "eval_progress.jsonl"
_QUOTE_STATUSES = ("verbatim", "near", "not_found", "missing")


def _run_signature(strategy: str, llm: LLMClient, verifier) -> str:
    """Identifies a grounding run. Saved per-question records are reused only
    by a run with the same signature: same models, prompts, knobs and the same
    generation / verification / retrieval code, so a resumed run never mixes
    answers produced under different conditions."""
    src = Path(__file__).resolve().parents[1] / "src"
    code = "".join((src / f).read_text(encoding="utf-8")
                   for f in ("generate.py", "verify.py", "retrieve.py"))
    parts = [strategy, llm.describe(), verifier.describe(), config.VERIFY_PROMPT,
             config.GENERATOR_PROMPT,
             config.PLAN_RESULT_RULE, str(config.QUOTE_REQUIRED), str(config.FINAL_TOP_K),
             config.EMBED_MODEL, str(config.RERANK_ENABLED), code]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def _question_key(kind: str, it: dict) -> str:
    return f"{kind}|{it['id']}|" + hashlib.md5(it["question"].encode("utf-8")).hexdigest()[:10]


def _load_progress(path: Path, signature: str) -> dict[str, dict]:
    """{question key: record} saved by an interrupted run with this signature."""
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:      # a line cut off by a hard stop
                continue
            if row.get("run") == signature:
                done[row["key"]] = row["record"]
    return done


_CJK_RE = re.compile(r"[㐀-鿿]")


def is_translated_quote(quote: str, passages: list[str]) -> bool:
    """A quote in Chinese while every cited passage is (almost) Chinese-free:
    the generator translated the source sentence instead of copying it."""
    if not _CJK_RE.search(quote or ""):
        return False
    text = "".join(passages)
    return bool(text) and len(_CJK_RE.findall(text)) / len(text) < 0.05


def _question_record(g, v) -> dict:
    """Everything the aggregate metrics need from one question, so they can be
    recomputed from saved records when a run resumes."""
    return {
        "refused": g.refused,
        "parse_failed": g.parse_failed,
        "misplaced_refusal": g.misplaced_refusal,
        "n_claims": v.n_claims,
        "n_supported": v.n_supported,
        "n_partial": v.n_partial,
        "n_unsupported": v.n_unsupported,
        "n_disputed": v.n_disputed,
        "n_citation_repaired": v.n_citation_repaired,
        "n_panel_judged": v.n_panel_judged,
        "n_panel_agreed": v.n_panel_agreed,
        # Claims no judge could vote on (timeout, error): sent to review.
        "n_unjudged": sum(x.method == "none" for x in v.verdicts),
        # Quotes translated instead of copied (format error; the quote check
        # then rejects them as fabricated).
        "n_translated_quotes": sum(
            is_translated_quote(c.quote, [p.text for p in g.passages if p.chunk_id in c.citation_ids])
            for c in g.claims),
        "quote_counts": {s: v.n_quote(s) for s in _QUOTE_STATUSES},
        "claims": _claim_detail(g, v),
        # Keep the raw output of malformed generations for diagnosis
        # (reports are gitignored, so this never leaves the machine).
        **({"raw": g.raw[:2000]} if g.parse_failed else {}),
    }


def eval_grounding(
    strategy: str, golden: dict, llm: LLMClient, limit: int | None = None,
    traps_only: bool = False, answerable_only: bool = False, fresh: bool = False,
) -> dict:
    """Grounding / refusal metrics. Each question's record is appended to
    eval_progress.jsonl as soon as it is scored, so an interrupted run (guard
    stop, crash, closed session) resumes where it stopped when started again
    with the same settings; ``fresh`` ignores saved records. The progress file
    is renamed once the run completes, so the next run starts fresh."""
    from src.pipeline import ask
    from src.verify import Verifier

    retrieval_items = golden["retrieval"][:limit] if limit else golden["retrieval"]
    if traps_only:
        retrieval_items = []
    trap_items = golden["traps"][:limit] if limit else golden["traps"]
    if answerable_only:
        trap_items = []

    verifier = Verifier(llm=llm)
    print(f"[grounding] verifier {verifier.describe()}")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.REPORT_DIR / PROGRESS_NAME
    signature = _run_signature(strategy, llm, verifier)
    done = {} if fresh else _load_progress(progress_path, signature)
    todo = [("answerable", it) for it in retrieval_items] + [("trap", it) for it in trap_items]
    n_resumed = sum(_question_key(k, it) in done for k, it in todo)
    if n_resumed:
        print(f"[grounding] resuming: {n_resumed}/{len(todo)} questions already scored "
              f"(run {signature}; --fresh to start over)")

    def scored(kind: str, it: dict) -> tuple[dict, bool]:
        key = _question_key(kind, it)
        if key in done:
            return done[key], True
        t0 = time.time()
        bundle = ask(it["question"], strategy=strategy, llm=llm, verifier=verifier)
        rec = _question_record(bundle.generation, bundle.verification)
        rec["seconds"] = round(time.time() - t0)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run": signature, "key": key, "record": rec},
                               ensure_ascii=False) + "\n")
        done[key] = rec
        return rec, False

    answer_detail = []
    for qi, it in enumerate(retrieval_items, 1):
        rec, resumed = scored("answerable", it)
        print(f"  [answerable {qi}/{len(retrieval_items)}] {it['id']}: "
              f"{rec['n_claims']} claim(s), refused={rec['refused']} "
              + ("(resumed)" if resumed else f"({rec['seconds']}s)"), flush=True)
        answer_detail.append({"id": it["id"], **rec})

    # Trap questions. What counts as correct depends on the trap type:
    #   far_absent / near_absent -> the topic is not in the corpus: must refuse.
    #   false_premise            -> the question presupposes something the
    #     corpus contradicts: correct if no claim is let through that the
    #     judges reject or dispute (refusing, or correcting the premise with
    #     supported claims, both pass).
    # "safe" (no unsupported or disputed claim) is recorded for every type.
    trap_detail = []
    for qi, it in enumerate(trap_items, 1):
        rec, resumed = scored("trap", it)
        kind = it.get("type", "far_absent")
        # Malformed output (no claims, no marker) is NOT a correct refusal.
        safe = rec["n_unsupported"] == 0 and rec["n_disputed"] == 0 and not rec["parse_failed"]
        correct = rec["refused"] if kind in MUST_REFUSE else safe
        print(f"  [trap {qi}/{len(trap_items)}] {it['id']} ({kind}): refused={rec['refused']} "
              f"claims={rec['n_claims']} correct={correct} "
              + ("(resumed)" if resumed else f"({rec['seconds']}s)"), flush=True)
        trap_detail.append({"id": it["id"], "type": kind, "correct": correct, "safe": safe, **rec})

    # Pool claim counts across questions (micro-average) rather than averaging
    # per-question rates, which would down-weight questions with many claims.
    def total(rows, field):
        return sum(r[field] for r in rows)

    every = answer_detail + trap_detail
    n_claims = total(answer_detail, "n_claims")
    n_supported = total(answer_detail, "n_supported")
    n_partial = total(answer_detail, "n_partial")
    n_unsupported = total(answer_detail, "n_unsupported")
    quote_counts = {s: sum(r["quote_counts"][s] for r in answer_detail) for s in _QUOTE_STATUSES}
    # Answer-level counts on answerable questions. Every retrieval question has
    # a known answer in the corpus, so refusing one is an over-refusal — the
    # counterweight that stops "refuse everything" from scoring perfectly.
    answered = [r for r in answer_detail if r["n_claims"]]

    must_refuse = [d for d in trap_detail if d["type"] in MUST_REFUSE]
    traps_by_type = {}
    for kind in sorted({d["type"] for d in trap_detail}):
        sub = [d for d in trap_detail if d["type"] == kind]
        traps_by_type[kind] = {
            "n": len(sub),
            "correct_rate": _ratio(sum(d["correct"] for d in sub), len(sub)),
            "refusal_rate": _ratio(sum(d["refused"] for d in sub), len(sub)),
            "safe_rate": _ratio(sum(d["safe"] for d in sub), len(sub)),
        }

    # Complete: retire the progress file so the next run starts fresh.
    if progress_path.exists():
        progress_path.replace(progress_path.with_name(f"eval_progress.done-{signature}.jsonl"))

    n_partial_ok = n_supported + n_partial
    must_refused = sum(d["refused"] for d in must_refuse)
    # Counts with 95% Wilson intervals (eval/stats.py) for the headline rates.
    counts = {
        "citation_precision_strict": stats.rate(n_supported, n_claims),
        "citation_precision_lenient": stats.rate(n_partial_ok, n_claims),
        "hallucination_rate": stats.rate(n_unsupported, n_claims),
        "refusal_correctness": stats.rate(must_refused, len(must_refuse)),
        "over_refusal_rate": stats.rate(total(answer_detail, "refused"), len(retrieval_items)),
        "answer_hallucination_rate": stats.rate(sum(r["n_unsupported"] > 0 for r in answered),
                                                len(answered)),
        **{f"trap_correct:{k}": stats.rate(sum(d["correct"] for d in trap_detail if d["type"] == k),
                                           v["n"])
           for k, v in traps_by_type.items()},
    }

    return {
        "strategy": strategy,
        "counts": counts,
        "n_retrieval_scored": len(retrieval_items),
        "n_traps_scored": len(trap_items),
        "n_resumed": n_resumed,
        "run_signature": signature,
        "n_claims_total": n_claims,
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "n_disputed": total(answer_detail, "n_disputed"),
        "verifier": verifier.describe(),
        "quote_counts": quote_counts,
        # Claims whose quote is absent from the cited passage: fabricated citations.
        "fabricated_quote_rate": _ratio(quote_counts["not_found"], n_claims),
        "panel_agreement_rate": _ratio(total(answer_detail, "n_panel_agreed"),
                                       total(answer_detail, "n_panel_judged")),
        "n_citation_repaired": total(answer_detail, "n_citation_repaired"),
        "citation_precision_strict": _ratio(n_supported, n_claims),
        "citation_precision_lenient": _ratio(n_supported + n_partial, n_claims),
        "hallucination_rate": _ratio(n_unsupported, n_claims),
        # Refusal correctness covers only traps that must be refused.
        "refusal_correctness": _ratio(sum(d["refused"] for d in must_refuse), len(must_refuse)),
        "traps_by_type": traps_by_type,
        "over_refusal_rate": _ratio(total(answer_detail, "refused"), len(retrieval_items)),
        "n_answered": len(answered),
        # Share of non-refused answers containing at least one unsupported claim.
        "answer_hallucination_rate": _ratio(sum(r["n_unsupported"] > 0 for r in answered),
                                            len(answered)),
        # Across answerable and trap questions; these count as disputed above.
        "n_unjudged": sum(r.get("n_unjudged", 0) for r in every),
        "n_translated_quotes": sum(r.get("n_translated_quotes", 0) for r in every),
        "generator_prompt": config.GENERATOR_PROMPT,
        "parse_failures": total(every, "parse_failed"),
        "misplaced_refusals": total(every, "misplaced_refusal"),
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

        def ci(g, key):
            """Rate with its 95% interval and counts when stored, else the plain rate."""
            c = g.get("counts", {}).get(key)
            return stats.fmt(c) if c else pct(g.get(key))

        A("### 2a. Claim level")
        A("")
        A("| Strategy | Citation Precision (strict) | Citation Precision (lenient) "
          "| Hallucination Rate | Claims (✔/◐/⚖/✘) |")
        A("|---|---|---|---|---|")
        for g in results["grounding"]:
            counts = (f"{g['n_claims_total']} ({g['n_supported']}/{g['n_partial']}/"
                      f"{g['n_disputed']}/{g['n_unsupported']})")
            A(f"| {g['strategy']} | {ci(g, 'citation_precision_strict')} "
              f"| {ci(g, 'citation_precision_lenient')} | {ci(g, 'hallucination_rate')} "
              f"| {counts} |")
        A("")
        A("> Strict = `supported` only; lenient = `supported` + `partially_supported`. "
          "`disputed` (judges disagree) counts as neither support nor hallucination. "
          "Micro-averaged over claims pooled across answerable questions. "
          "Brackets: 95% Wilson confidence interval in percentage points, then counts. "
          "Claims are pooled, so the interval treats them as independent although "
          "claims from one answer are correlated; read it as a lower bound on the "
          "uncertainty.")
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
          "| Answer Hallucination Rate | Answered | Parse Failures | Misplaced Refusals "
          "| Translated Quotes |")
        A("|---|---|---|---|---|---|---|---|")
        for g in results["grounding"]:
            A(f"| {g['strategy']} | {ci(g, 'refusal_correctness')} "
              f"| {ci(g, 'over_refusal_rate')} | {ci(g, 'answer_hallucination_rate')} "
              f"| {g['n_answered']}/{g['n_retrieval_scored']} | {g['parse_failures']} "
              f"| {g.get('misplaced_refusals', 0)} | {g.get('n_translated_quotes', 'n/a')} |")
        A("")
        A("> Refusal correctness and over-refusal must be read together: a system "
          "that refuses everything scores 100% on the first and 100% (worst) on the "
          "second. Answer hallucination rate = share of non-refused answers with at "
          "least one unsupported claim.")
        A("")
        A("> Parse failures = outputs with neither a `<claim>` nor the refusal marker. "
          "They are not counted as refusals. Misplaced refusals = the refusal marker "
          "wrapped in a `<claim>`; it is counted as a refusal but reported here as a "
          "format error.")
        A("")
        tbt = results["grounding"][0].get("traps_by_type")
        if tbt:
            A("### 2c. Traps by type")
            A("")
            A("| Type | n | Correct | Refused | Safe (no rejected/disputed claim) |")
            A("|---|---|---|---|---|")
            g0c = results["grounding"][0].get("counts", {})
            for kind, t in tbt.items():
                correct = (stats.fmt(g0c[f"trap_correct:{kind}"]) if f"trap_correct:{kind}" in g0c
                           else pct(t["correct_rate"]))
                A(f"| {kind} | {t['n']} | {correct} | {pct(t['refusal_rate'])} "
                  f"| {pct(t['safe_rate'])} |")
            A("")
            A("> far_absent / near_absent: correct = refused (the topic is not in the corpus; "
              "near_absent questions are about the corpus's own studies). false_premise: "
              "correct = safe — refusing or correcting the premise with supported claims "
              "both pass. Refusal correctness above covers only the must-refuse types.")
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
    ap.add_argument("--traps-only", action="store_true",
                    help="LLM pass on trap questions only (skip the answerable ones)")
    ap.add_argument("--answerable-only", action="store_true",
                    help="LLM pass on answerable questions only (skip the traps)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore questions saved by an interrupted run and start over")
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
            g = eval_grounding(args.llm_strategy, golden, llm, limit=args.llm_limit,
                               traps_only=args.traps_only,
                               answerable_only=args.answerable_only, fresh=args.fresh)
            results["grounding"] = [g]
            print(f"[grounding:{args.llm_strategy}] precision strict/lenient="
                  f"{g['citation_precision_strict']}/{g['citation_precision_lenient']} "
                  f"halluc={g['hallucination_rate']} "
                  f"answer_halluc={g['answer_hallucination_rate']} "
                  f"refusal={g['refusal_correctness']} over_refusal={g['over_refusal_rate']} "
                  f"parse_failures={g['parse_failures']} unjudged={g['n_unjudged']} "
                  f"translated_quotes={g['n_translated_quotes']} prompt={g['generator_prompt']}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(results, config.REPORT_DIR / "eval_report.md")
    print(f"\nReport written to {config.REPORT_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main()
