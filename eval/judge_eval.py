"""Judge evaluation — score verifier judges against known-answer items.

For each judge in ``--judges`` (default: ``config.JUDGES``, else the generator's
own backend), every item in the judge validation set is judged once with the
production prompt (``Verifier._llm_vote``). Votes are cached in
``eval/reports/judge_votes.jsonl``, so re-runs are free and adding a judge
(e.g. a new model or service) only costs that judge's calls. Panels are then
scored offline by combining cached votes with each ``PANEL_RULE``.

Metrics (per judge / panel, per split):
  * False-accept rate  — gold NOT supported, judged supported. The dangerous
                         error: a hallucination waved through.
  * False-reject rate  — gold supported, judged not supported (feeds
                         over-refusal / needless "unsupported" flags).
  * 3-class accuracy and Cohen's kappa vs gold.
  * Detection rate per perturbation type — which kinds of error a judge misses.

Run:  python -m eval.judge_eval --judges ollama:llama3:latest ollama:qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from itertools import combinations

import config
from eval.judge_set import load_items
from src.verify import VERIFY_PROMPTS, Verifier, _parse_judge_spec, aggregate_votes

VOTES_PATH = config.REPORT_DIR / "judge_votes.jsonl"
UNPARSED = "unparsed"   # judge output could not be parsed into a label


def _key(prompt: str, judge_name: str, item: dict) -> str:
    h = hashlib.md5((item["claim"] + "\x00" + item["passage"]).encode()).hexdigest()[:12]
    return f"{prompt}|{judge_name}|{item['id']}|{h}"


def _load_cache() -> dict[str, str]:
    cache = {}
    if VOTES_PATH.exists():
        for line in VOTES_PATH.open(encoding="utf-8"):
            row = json.loads(line)
            key = row["key"]
            if key.count("|") == 2:      # cached before prompt versions existed
                key = "v1|" + key
            cache[key] = row["label"]
    return cache


def collect_votes(
    judge_specs: list[str], items: list[dict], prompts: list[str]
) -> dict[str, dict[str, str]]:
    """Return {"judge (prompt)": {item_id: label}}, calling judges only on cache misses."""
    cache = _load_cache()
    votes: dict[str, dict[str, str]] = {}
    with VOTES_PATH.open("a", encoding="utf-8") as out:
        for spec in judge_specs:
            judge = _parse_judge_spec(spec)
            if not judge.available:
                print(f"[judge-eval] {judge.describe()} unavailable; skipped")
                continue
            for prompt in prompts:
                name = f"{judge.describe()} ({prompt})"
                votes[name] = {}
                misses = [it for it in items if _key(prompt, judge.describe(), it) not in cache]
                print(f"[judge-eval] {name}: {len(items) - len(misses)} cached, "
                      f"{len(misses)} to judge")
                for i, it in enumerate(items, 1):
                    k = _key(prompt, judge.describe(), it)
                    if k not in cache:
                        vote = Verifier._llm_vote(judge, it["claim"], it["passage"],
                                                  prompt=VERIFY_PROMPTS[prompt])
                        cache[k] = vote[0] if vote else UNPARSED
                        out.write(json.dumps({"key": k, "label": cache[k]},
                                             ensure_ascii=False) + "\n")
                        out.flush()
                        if i % 10 == 0:
                            print(f"             {i}/{len(items)}")
                    votes[name][it["id"]] = cache[k]
    return votes


def panel_votes(votes: dict[str, dict[str, str]], members: tuple[str, ...], rule: str) -> dict[str, str]:
    out = {}
    for item_id in votes[members[0]]:
        labels = [votes[m][item_id] for m in members if votes[m][item_id] != UNPARSED]
        out[item_id] = aggregate_votes(labels, rule) if labels else UNPARSED
    return out


def _kappa(gold: list[str], pred: list[str]) -> float | None:
    n = len(gold)
    if not n:
        return None
    po = sum(g == p for g, p in zip(gold, pred)) / n
    cg, cp = Counter(gold), Counter(pred)
    pe = sum(cg[l] * cp[l] for l in set(cg) | set(cp)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def _r(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(items: list[dict], preds: dict[str, str]) -> dict:
    gold = [it["gold_label"] for it in items]
    pred = [preds[it["id"]] for it in items]
    neg = [(g, p) for g, p in zip(gold, pred) if g != "supported"]
    pos = [(g, p) for g, p in zip(gold, pred) if g == "supported"]
    by_type = {}
    for kind in sorted({it["perturbation"] for it in items}):
        sub = [(it, preds[it["id"]]) for it in items if it["perturbation"] == kind]
        if kind == "original":
            ok = sum(p == "supported" for _, p in sub)
        else:   # detected = anything but a wrongful "supported"
            ok = sum(p != "supported" if it["gold_label"] != "supported" else p == "supported"
                     for it, p in sub)
        by_type[kind] = {"n": len(sub), "correct": _r(ok, len(sub))}
    return {
        "n": len(items),
        "false_accept_rate": _r(sum(p == "supported" for _, p in neg), len(neg)),
        "false_reject_rate": _r(sum(p != "supported" for _, p in pos), len(pos)),
        "accuracy_3class": _r(sum(g == p for g, p in zip(gold, pred)), len(items)),
        "kappa": _kappa(gold, pred),
        "disputed_rate": _r(pred.count("disputed"), len(items)),
        "unparsed_rate": _r(pred.count(UNPARSED), len(items)),
        "by_type": by_type,
    }


def write_report(results: dict, path) -> None:
    def pct(x):
        return f"{x:.0%}" if x is not None else "n/a"

    L = []
    A = L.append
    A("# Judge Validation Report")
    A("")
    A(f"- Generated: {results['generated_at']}")
    A(f"- Items: {results['n_items']} ({results['composition']})")
    A(f"- Split scored: `{results['split']}`")
    A("")
    A("## 1. Overall")
    A("")
    A("| Judge / Panel | n | False-accept ↓ | False-reject ↓ | Accuracy (3-class) | κ | Disputed | Unparsed |")
    A("|---|---|---|---|---|---|---|---|")
    for name, s in results["scores"].items():
        A(f"| {name} | {s['n']} | **{pct(s['false_accept_rate'])}** | {pct(s['false_reject_rate'])} "
          f"| {pct(s['accuracy_3class'])} | {s['kappa'] if s['kappa'] is not None else 'n/a'} "
          f"| {pct(s['disputed_rate'])} | {pct(s['unparsed_rate'])} |")
    A("")
    A("> **False-accept** = gold not supported but judged `supported` — a hallucination "
      "let through; the number that matters most. **False-reject** = gold supported "
      "but judged otherwise. A `disputed` panel verdict is not an accept, so panels "
      "trade false-accepts for disputed items routed to human review.")
    A("")
    A("## 2. Correct rate by perturbation type")
    A("")
    kinds = sorted({k for s in results["scores"].values() for k in s["by_type"]})
    A("| Judge / Panel | " + " | ".join(kinds) + " |")
    A("|---|" + "---|" * len(kinds))
    for name, s in results["scores"].items():
        cells = [f"{pct(s['by_type'][k]['correct'])} (n={s['by_type'][k]['n']})"
                 if k in s["by_type"] else "—" for k in kinds]
        A(f"| {name} | " + " | ".join(cells) + " |")
    A("")
    A("> `original` = verbatim sentence judged supported. Other columns = share of "
      "perturbed claims NOT waved through as `supported` (for human items: the gold "
      "label's side of supported / not supported).")
    A("")
    A("> Caveat: synthetic positives are verbatim sentences, which are easier than "
      "real paraphrased claims. Add human-labelled items to "
      "`data/golden/judge_set_human.jsonl` for a realistic estimate.")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score verifier judges on the judge validation set")
    ap.add_argument("--judges", nargs="+", default=None,
                    help='judge specs "backend:model" (default: config.JUDGES or the generator backend)')
    ap.add_argument("--prompts", nargs="+", choices=list(VERIFY_PROMPTS),
                    default=[config.VERIFY_PROMPT],
                    help="judge prompt version(s) to score, e.g. --prompts v1 v2 to compare")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N items (quick run)")
    args = ap.parse_args(argv)

    specs = args.judges or config.JUDGES or [f"{config.LLM_BACKEND}:{config.LLM_MODEL}"]
    items = load_items()
    if args.split != "all":
        items = [it for it in items if it["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    votes = collect_votes(specs, items, args.prompts)
    preds = dict(votes)
    # Panels combine judges that ran the same prompt version.
    for prompt in args.prompts:
        names = [n for n in votes if n.endswith(f"({prompt})")]
        for size in range(2, len(names) + 1):
            for members in combinations(names, size):
                for rule in ("unanimous", "majority") if size >= 3 else ("unanimous",):
                    short = " + ".join(m.removesuffix(f" ({prompt})") for m in members)
                    preds[f"panel[{short}] {rule} ({prompt})"] = panel_votes(votes, members, rule)

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "n_items": len(items),
        "composition": ", ".join(f"{k}={v}" for k, v in Counter(it["perturbation"] for it in items).items()),
        "scores": {name: score(items, p) for name, p in preds.items()},
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "judge_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, config.REPORT_DIR / "judge_report.md")
    for name, s in results["scores"].items():
        print(f"[judge-eval] {name}: false_accept={s['false_accept_rate']} "
              f"false_reject={s['false_reject_rate']} acc3={s['accuracy_3class']} kappa={s['kappa']}")
    print(f"\nReport written to {config.REPORT_DIR / 'judge_report.md'}")


if __name__ == "__main__":
    main()
