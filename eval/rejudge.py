"""Sensitivity analysis: re-judge non-supported claims against all context passages.

The baseline arm judges each sentence against only its 2 closest context
passages, and the audit found sentences whose evidence sat in one of the other
3. This script takes the saved results of both arms, picks the claims a wider
view could change (answerable questions: every non-supported claim; traps:
unsupported or disputed claims, the only labels that affect a trap's "safe"
verdict), judges each against the context passages it has not yet been judged
against (early stop on support), keeps the better of the old and new labels,
and recomputes the comparison. Both arms get the same treatment, since a wider
view can only make the judge more lenient.

Each re-judged claim is appended to ``rejudge_progress.jsonl`` (local), so an
interrupted run resumes. Outputs (``rejudge_results.json``) hold corpus text
and stay local like every report.

Run: python -m eval.rejudge [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import config
from eval import stats

PROGRESS = "rejudge_progress.jsonl"
RESULTS = "rejudge_results.json"
MUST_REFUSE = {"far_absent", "near_absent"}
TRAP_RELEVANT = {"unsupported", "disputed"}

# (arm, results file, part). The grounded traps come from two runs: the 25
# shared traps and the 6 hypothesis-bait items added later.
SOURCES = [
    ("baseline", "eval_results_baseline_rescored.json", "answer_detail"),
    ("baseline", "eval_results_baseline_rescored.json", "trap_detail"),
    ("grounded", "eval_results_phi4v3_fix.json", "answer_detail"),
    ("grounded", "eval_results_traps_postfix.json", "trap_detail"),
    ("grounded", "eval_results_bait_grounded_v1.json", "trap_detail"),
]


def needs_rejudge(label: str, part: str) -> bool:
    if part == "answer_detail":
        return label != "supported"
    return label in TRAP_RELEVANT


def _signature(verifier) -> str:
    code = (Path(__file__).resolve().parents[1] / "src" / "verify.py").read_text(encoding="utf-8")
    parts = [verifier.describe(), config.VERIFY_PROMPT, config.PLAN_RESULT_RULE,
             str(config.FINAL_TOP_K), config.EMBED_MODEL, code]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def claim_key(arm: str, part: str, qid: str, statement: str) -> str:
    return f"{arm}|{part}|{qid}|" + hashlib.md5(statement.encode("utf-8")).hexdigest()[:10]


def already_judged(arm: str, claim: dict, passages: list, embedder) -> set[str]:
    """Passage ids the original verdict looked at: the baseline's 2 closest
    passages (recomputed; retrieval and embeddings are deterministic), or the
    grounded claim's cited and judged passages."""
    seen = set(claim.get("judged_against") or [])
    if arm == "baseline":
        from eval.run_eval import _closest
        seen |= {p.chunk_id for p in _closest(claim["statement"], passages, embedder)}
    else:
        seen |= set(claim.get("cited") or [])
    return seen


def better(old: str, new: str) -> str:
    from src.verify import _RANK
    return new if _RANK[new] > _RANK[old] else old


def rejudge_claim(arm: str, claim: dict, passages: list, embedder, verifier) -> dict:
    seen = already_judged(arm, claim, passages, embedder)
    rest = [p for p in passages if p.chunk_id not in seen]
    if not rest:
        return {"new": claim["label"], "extra": [], "reason": "no unseen passage", "votes": {}}
    v = verifier.verify_statement(claim["statement"], rest)
    return {"new": better(claim["label"], v.label), "extra": [p.chunk_id for p in rest],
            "judged_new": v.label, "against": v.citation_ids, "reason": v.reason,
            "votes": v.votes}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired booleans."""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p": round(p, 6)}


def summarize(rows: dict) -> dict:
    """Metrics per arm from {arm: {"answer": [...], "trap": [...]}} records whose
    claims carry the final label."""
    out = {}
    for arm, parts in rows.items():
        labels = [c for d in parts["answer"] for c in d["labels"]]
        n = len(labels)
        traps = parts["trap"]

        def safe(d):
            return not any(lab in TRAP_RELEVANT for lab in d["labels"]) and not d["parse_failed"]

        must = [d for d in traps if d["type"] in MUST_REFUSE]
        out[arm] = {
            "strict": stats.rate(labels.count("supported"), n),
            "lenient": stats.rate(labels.count("supported") + labels.count("partially_supported"), n),
            "hallucination": stats.rate(labels.count("unsupported"), n),
            "answers_unsupported": stats.rate(
                sum("unsupported" in d["labels"] for d in parts["answer"]), len(parts["answer"])),
            "answers_not_clean": stats.rate(
                sum(any(x != "supported" for x in d["labels"]) for d in parts["answer"]),
                len(parts["answer"])),
            "must_refuse_safe": stats.rate(sum(safe(d) for d in must), len(must)),
            "by_question": {
                **{f"A|{d['id']}": {"unsupported": "unsupported" in d["labels"],
                                    "not_clean": any(x != "supported" for x in d["labels"])}
                   for d in parts["answer"]},
                **{f"T|{d['id']}": {"type": d["type"], "safe": safe(d),
                                    "correct": d["refused"] if d["type"] in MUST_REFUSE else safe(d)}
                   for d in traps}},
        }
    return out


def compare(summary: dict) -> dict:
    """Paired tests, grounded vs baseline, on the questions both arms answered."""
    g, b = summary["grounded"]["by_question"], summary["baseline"]["by_question"]
    shared = [k for k in g if k in b]
    ans = [k for k in shared if k.startswith("A|")]
    traps = [k for k in shared if k.startswith("T|")]
    must = [k for k in traps if g[k]["type"] in MUST_REFUSE]
    bait = [k for k in traps if g[k]["type"] == "hypothesis_bait"]
    core = [k for k in traps if g[k]["type"] != "hypothesis_bait"]

    def pair(keys, field, flip=False):
        a = [g[k][field] != flip for k in keys]
        bb = [b[k][field] != flip for k in keys]
        return {"grounded": stats.rate(sum(a), len(keys)),
                "baseline": stats.rate(sum(bb), len(keys)), **mcnemar(a, bb)}

    return {
        "answers_unsupported": pair(ans, "unsupported"),
        "answers_not_clean": pair(ans, "not_clean"),
        "must_refuse_safe": pair(must, "safe"),
        "traps_correct": pair(core, "correct"),
        "bait_safe": pair(bait, "safe"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="re-judge at most N claims (a trial run); results are not written")
    ap.add_argument("--strategy", default="section")
    args = ap.parse_args(argv)

    from src.llm import LLMClient
    from src.retrieve import get_retriever
    from src.verify import Verifier
    from eval.run_eval import load_golden

    report = config.REPORT_DIR
    golden = load_golden()
    questions = {it["id"]: it["question"] for it in golden["retrieval"] + golden["traps"]}
    verifier = Verifier(llm=LLMClient())
    sig = _signature(verifier)
    print(f"[rejudge] verifier {verifier.describe()} (run {sig})")
    retriever = get_retriever(args.strategy)

    progress = report / PROGRESS
    done = {}
    if progress.exists():
        for line in progress.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run") == sig:
                done[row["key"]] = row["record"]

    rows = {"baseline": {"answer": [], "trap": []}, "grounded": {"answer": [], "trap": []}}
    flips, judged_now = [], 0
    contexts = {}
    for arm, fname, part in SOURCES:
        res = json.loads((report / fname).read_text(encoding="utf-8"))["grounding"][0]
        for d in res[part]:
            labels = []
            for c in d["claims"]:
                lab = c["label"]
                if needs_rejudge(lab, part):
                    key = claim_key(arm, part, d["id"], c["statement"])
                    rec = done.get(key)
                    if rec is None:
                        if args.limit is not None and judged_now >= args.limit:
                            labels.append(lab)
                            continue
                        if d["id"] not in contexts:
                            contexts[d["id"]] = retriever.search(questions[d["id"]])
                        rec = rejudge_claim(arm, c, contexts[d["id"]], retriever.embedder, verifier)
                        rec["old"] = lab
                        judged_now += 1
                        with progress.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"run": sig, "key": key, "record": rec},
                                               ensure_ascii=False) + "\n")
                        print(f"  [{arm} {d['id']}] {lab} -> {rec['new']} "
                              f"(judged {rec.get('judged_new', '-')} on {len(rec['extra'])} "
                              f"unseen passage(s))", flush=True)
                    if rec["new"] != lab:
                        flips.append({"arm": arm, "id": d["id"], "old": lab, "new": rec["new"],
                                      "statement": c["statement"], "against": rec.get("against"),
                                      "reason": rec.get("reason")})
                    lab = rec["new"]
                labels.append(lab)
            rows[arm]["answer" if part == "answer_detail" else "trap"].append(
                {"id": d["id"], "type": d.get("type"), "labels": labels,
                 "refused": d["refused"], "parse_failed": d.get("parse_failed", False)})

    if args.limit is not None:
        print(f"[rejudge] trial: {judged_now} claim(s) judged; rerun without --limit for results")
        return
    summary = summarize(rows)
    result = {"run": sig, "verifier": verifier.describe(), "flips": flips,
              "comparison": compare(summary),
              "metrics": {arm: {k: v for k, v in m.items() if k != "by_question"}
                          for arm, m in summary.items()}}
    (report / RESULTS).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rejudge] {judged_now} claim(s) judged this run; {len(flips)} label(s) changed")
    for name, c in result["comparison"].items():
        print(f"  {name}: grounded {stats.fmt(c['grounded'])} | baseline "
              f"{stats.fmt(c['baseline'])} | McNemar p = {c['p']}")
    print(f"[rejudge] written {report / RESULTS}")


if __name__ == "__main__":
    main()
