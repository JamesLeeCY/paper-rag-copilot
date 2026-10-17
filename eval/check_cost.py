"""How many passages does `check` need to judge? Measure once, simulate offline.

`check` judges each sentence against its top CHECK_TOP_N retrieved passages
in retrieval order and stops at the first that supports it. Each judgment
costs ~45-75 s with phi4 on CPU. This tool judges every sentence of a labelled
set against each candidate passage once (no early stop), records the verdict
and time per passage, and then simulates selection strategies offline:
how many passages, in which order (retrieval rank or embedding similarity),
with early stop. It reports accuracy against the gold labels, agreement with
the current strategy, judge calls and seconds per sentence.

Items: dev-split statements of the synthetic judge set (local), as if a user
had written them: originals (gold supported) and perturbations (negation,
number, plan-to-result, attribution swap, overclaim; gold not supported).
Each judged (sentence, passage) pair is appended to check_cost_progress.jsonl
(local), so an interrupted run resumes.

Caveat: a gold label is relative to the passage the item was built from. A
plan_to_result item restates a hypothesis as a finding, which that passage
does not support; if the source reports the hypothesis as a result elsewhere,
`check` rightly finds it, so read such "false accepts" before counting them
(2026-10-17: 2 of the 20 perturbed items).

Run: python -m eval.check_cost [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import config
from eval import stats

PROGRESS = "check_cost_progress.jsonl"
RESULTS = "check_cost_results.json"
MIX = {"original": 20, "negation": 5, "number": 4, "plan_to_result": 4,
       "attribution_swap": 3, "overclaim": 4}
POOL = 5          # passages retrieved; candidates come from these


def pick_items(seed: int = 7) -> list[dict]:
    rows = [json.loads(l) for l in (config.GOLDEN_DIR / "judge_set_synthetic.jsonl")
            .open(encoding="utf-8")]
    rng = random.Random(seed)
    out = []
    for kind, n in MIX.items():
        pool = sorted((r for r in rows if r.get("split") == "dev" and r["perturbation"] == kind),
                      key=lambda r: r["id"])
        out += rng.sample(pool, min(n, len(pool)))
    return out


def similarity_order(sentence: str, passages: list, embedder) -> list:
    from eval.run_eval import _closest
    return _closest(sentence, passages, embedder, k=len(passages))


def strategies(passages: list, by_sim: list) -> dict[str, list]:
    """Passage ids each strategy would judge, in order."""
    ret = [p.chunk_id for p in passages]
    sim = [p.chunk_id for p in by_sim]
    return {"retrieval_top3 (current)": ret[:3], "retrieval_top2": ret[:2],
            "retrieval_top1": ret[:1], "similarity_top3": sim[:3],
            "similarity_top2": sim[:2], "similarity_top1": sim[:1]}


def simulate(order: list[str], verdicts: dict[str, dict]) -> tuple[str, int, float]:
    """Best label with early stop on support: (label, judge calls, seconds)."""
    from src.verify import _RANK
    best, calls, secs = None, 0, 0.0
    for cid in order:
        v = verdicts[cid]
        calls += 1
        secs += v["seconds"]
        if best is None or _RANK[v["label"]] > _RANK[best]:
            best = v["label"]
        if best == "supported":
            break
    return best or "unsupported", calls, secs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="judge at most N new pairs (trial)")
    args = ap.parse_args(argv)

    from src.llm import LLMClient, ollama_version
    from src.retrieve import get_retriever
    from src.verify import Verifier

    report = config.REPORT_DIR
    verifier = Verifier(llm=LLMClient())
    sig = hashlib.sha256("\x00".join([verifier.describe(), config.VERIFY_PROMPT,
                                      config.PLAN_RESULT_RULE, config.PLAN_RESULT_MATCH,
                                      ollama_version()]).encode()).hexdigest()[:16]
    print(f"[check-cost] verifier {verifier.describe()} (run {sig})")
    retriever = get_retriever(config.CHUNK_STRATEGY)
    done = {}
    path = report / PROGRESS
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run") == sig:
                done[row["key"]] = row["record"]

    items, judged = pick_items(), 0
    per_item = []
    for it in items:
        sent = it["claim"]
        passages = retriever.search(sent)[:POOL]
        by_sim = similarity_order(sent, passages, retriever.embedder)
        strat = strategies(passages, by_sim)
        needed = list(dict.fromkeys(cid for ids in strat.values() for cid in ids))
        verdicts = {}
        for cid in needed:
            key = f"{it['id']}|{cid}"
            rec = done.get(key)
            if rec is None:
                if args.limit is not None and judged >= args.limit:
                    break
                p = next(x for x in passages if x.chunk_id == cid)
                t0 = time.time()
                v = verifier.verify_statement(sent, [p], context=passages)
                rec = {"label": v.label, "seconds": round(time.time() - t0, 1),
                       "reason": v.reason[:300]}
                judged += 1
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"run": sig, "key": key, "record": rec},
                                       ensure_ascii=False) + "\n")
                print(f"  [{it['id']} {cid}] {rec['label']} ({rec['seconds']}s)", flush=True)
            verdicts[cid] = rec
        if len(verdicts) < len(needed):
            break
        per_item.append({"id": it["id"], "perturbation": it["perturbation"],
                         "gold": it["gold_label"], "strategies": strat, "verdicts": verdicts})

    if args.limit is not None or len(per_item) < len(items):
        print(f"[check-cost] {judged} pair(s) judged; {len(per_item)}/{len(items)} items complete")
        return

    summary = {}
    current = {r["id"]: simulate(r["strategies"]["retrieval_top3 (current)"], r["verdicts"])[0]
               for r in per_item}
    for name in per_item[0]["strategies"]:
        rows = [(r, *simulate(r["strategies"][name], r["verdicts"])) for r in per_item]
        true = [x for x in rows if x[0]["gold"] == "supported"]
        false = [x for x in rows if x[0]["gold"] != "supported"]
        summary[name] = {
            "false_reject": stats.rate(sum(lab != "supported" for _, lab, _, _ in true), len(true)),
            "false_accept": stats.rate(sum(lab == "supported" for _, lab, _, _ in false), len(false)),
            "agree_with_current": stats.rate(sum(lab == current[r["id"]] for r, lab, _, _ in rows),
                                             len(rows)),
            "mean_calls": round(sum(c for *_, c, _ in rows) / len(rows), 2),
            "mean_seconds": round(sum(s for *_, s in rows) / len(rows), 1),
        }
        s = summary[name]
        print(f"  {name:26} FR {stats.fmt(s['false_reject'])}  FA {stats.fmt(s['false_accept'])}  "
              f"agree {stats.fmt(s['agree_with_current'])}  calls {s['mean_calls']}  "
              f"{s['mean_seconds']} s/sentence")
    (report / RESULTS).write_text(json.dumps({"run": sig, "summary": summary, "items": per_item},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[check-cost] written {report / RESULTS}")


if __name__ == "__main__":
    main()
