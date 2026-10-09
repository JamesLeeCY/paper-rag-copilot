"""Unified command-line entry point for the dissertation RAG Copilot.

Examples
--------
    python cli.py build --all
    python cli.py ask "What is the EAR model?"
    python cli.py check "Nature exposure raised cortisol and improved sleep."
    python cli.py eval --with-llm
"""
from __future__ import annotations

import argparse
import sys

import config


def cmd_build(args):
    from src.ingest import ingest, load_references
    from src.index import build_index
    from src.embedder import Embedder

    strategies = ["fixed", "section"] if args.all else [args.strategy]
    embedder = Embedder()
    for strat in strategies:
        chunks = ingest(strat)
        info = build_index(strat, embedder)
        print(f"[build:{strat}] {len(chunks)} chunks indexed -> {info['collection']}")
    print(f"References parsed: {len(load_references())}")


def cmd_ask(args):
    from src.pipeline import ask, format_answer
    from src.llm import LLMClient

    llm = LLMClient()
    print(f"[llm] backend = {llm.describe()}\n")
    bundle = ask(args.query, strategy=args.strategy, top_k=args.k,
                 llm=llm, grounded=not args.baseline)
    if not llm.available:
        print("[note] LLM backend unavailable — showing retrieved passages "
              "(generation runs in mock mode).\n")
        for i, p in enumerate(bundle.passages, 1):
            print(f"#{i} {p.locator()}\n    {p.text[:200].strip()}\n")
    print(format_answer(bundle))


def cmd_check(args):
    from src.pipeline import check, format_answer

    text = args.text or sys.stdin.read()
    bundles = check(text, strategy=args.strategy)
    print(f"Reverse hallucination check — {len(bundles)} sentence(s)\n" + "=" * 60)
    for b in bundles:
        print(format_answer(b))
        print("-" * 60)


def cmd_eval(args):
    from eval.run_eval import main as eval_main

    sys.argv = ["run_eval"]
    if args.with_llm:
        sys.argv.append("--with-llm")
    if args.rerank:
        sys.argv.append("--rerank")
    if args.llm_limit is not None:
        sys.argv += ["--llm-limit", str(args.llm_limit)]
    if args.llm_strategy:
        sys.argv += ["--llm-strategy", args.llm_strategy]
    if args.traps_only:
        sys.argv.append("--traps-only")
    if args.answerable_only:
        sys.argv.append("--answerable-only")
    if args.fresh:
        sys.argv.append("--fresh")
    eval_main()


def cmd_judge_build(args):
    from eval.judge_set import main as build_main

    sys.argv = ["judge_set", "--n-sentences", str(args.n_sentences), "--seed", str(args.seed)]
    build_main()


def cmd_judge_eval(args):
    from eval.judge_eval import main as judge_main

    argv = ["--split", args.split]
    if args.types:
        argv += ["--types", *args.types]
    if args.judges:
        argv += ["--judges", *args.judges]
    if args.prompts:
        argv += ["--prompts", *args.prompts]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.with_heading:
        argv.append("--with-heading")
    judge_main(argv)


def build_parser():
    ap = argparse.ArgumentParser(prog="cli.py", description="Dissertation RAG Copilot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ingest + index the dissertation")
    b.add_argument("--strategy", choices=["fixed", "section"], default=config.CHUNK_STRATEGY)
    b.add_argument("--all", action="store_true", help="build both strategies")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="ask a question (citation-grounded)")
    a.add_argument("query")
    a.add_argument("--strategy", default=config.CHUNK_STRATEGY)
    a.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    a.add_argument("--baseline", action="store_true", help="ungrounded baseline prompt")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("check", help="reverse-check a paragraph you wrote")
    c.add_argument("text", nargs="?", help="text to verify (or pipe via stdin)")
    c.add_argument("--strategy", default=config.CHUNK_STRATEGY)
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--with-llm", action="store_true")
    e.add_argument("--rerank", action="store_true")
    e.add_argument("--llm-limit", type=int, default=None,
                   help="cap #questions for the LLM pass (e.g. 3 for a quick run)")
    e.add_argument("--llm-strategy", default="section")
    e.add_argument("--traps-only", action="store_true",
                   help="LLM pass on trap questions only")
    e.add_argument("--answerable-only", action="store_true",
                   help="LLM pass on answerable questions only")
    e.add_argument("--fresh", action="store_true",
                   help="ignore questions saved by an interrupted run and start over")
    e.set_defaults(func=cmd_eval)

    jb = sub.add_parser("judge-build", help="build the synthetic judge validation set")
    jb.add_argument("--n-sentences", type=int, default=60)
    jb.add_argument("--seed", type=int, default=42)
    jb.set_defaults(func=cmd_judge_build)

    je = sub.add_parser("judge-eval", help="score verifier judges on the validation set")
    je.add_argument("--judges", nargs="+", default=None,
                    help='judge specs "backend:model" (default: JUDGES env / generator backend)')
    je.add_argument("--prompts", nargs="+", default=None,
                    help="judge prompt version(s), e.g. --prompts v1 v2 to compare")
    je.add_argument("--split", choices=["dev", "test", "all"], default="all")
    je.add_argument("--types", nargs="+", default=None, help="only these item types")
    je.add_argument("--limit", type=int, default=None)
    je.add_argument("--with-heading", action="store_true",
                    help="show judges each passage's section heading (as the live verifier does)")
    je.set_defaults(func=cmd_judge_eval)

    return ap


if __name__ == "__main__":
    # Windows consoles default to a legacy codepage (e.g. cp950) that cannot
    # encode the ✔/◐/✘ verdict badges; force UTF-8 so output never crashes.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    ns = parser.parse_args()
    ns.func(ns)
