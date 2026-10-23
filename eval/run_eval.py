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


def _run_signature(strategy: str, llm: LLMClient, verifier, arm: str = "grounded",
                   reuse_from: str = "") -> str:
    """Identifies a grounding run. Saved per-question records are reused only
    by a run with the same signature: same models, prompts, knobs and the same
    generation / verification / retrieval code, so a resumed run never mixes
    answers produced under different conditions."""
    src = Path(__file__).resolve().parents[1] / "src"
    code = "".join((src / f).read_text(encoding="utf-8")
                   for f in ("generate.py", "verify.py", "retrieve.py"))
    if arm == "baseline":   # sentence splitting and refusal detection live here
        code += Path(__file__).read_text(encoding="utf-8")
    parts = [arm, strategy, llm.describe(), verifier.describe(), config.VERIFY_PROMPT,
             config.GENERATOR_PROMPT,
             config.PLAN_RESULT_RULE, str(config.QUOTE_REQUIRED), str(config.FINAL_TOP_K),
             config.EMBED_MODEL, str(config.RERANK_ENABLED), code]
    if config.CONTEXT_ORDER != "rank":     # (absent at the default: older runs still resume)
        parts.append(f"order:{config.CONTEXT_ORDER}")
    if config.PLAN_RESULT_MATCH == "on":   # (absent when off: older runs still resume)
        parts.append(f"match:{config.PLAN_MATCH_MODEL}:{config.PLAN_MATCH_MIN_SIM}:"
                     f"{config.PLAN_MATCH_MARGIN}")
    if config.ZH_OVERCLAIM_RULE != "off":  # (absent when off: older runs still resume)
        parts.append(f"zh-overclaim:{config.ZH_OVERCLAIM_RULE}")
    if config.ECHO_RULE != "off":          # (absent when off: older runs still resume)
        parts.append(f"echo:{config.ECHO_RULE}:{config.ECHO_MIN_SIM}:{config.ECHO_MIN_OVERLAP}")
    if reuse_from:          # answers and verdicts carried over from that run
        parts.append(f"reuse:{reuse_from}")
    if "ollama" in llm.describe() or "ollama" in verifier.describe():
        from src.llm import ollama_version
        parts.append(f"ollama-server:{ollama_version()}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def in_language(golden: dict, lang: str) -> dict:
    """The golden set with each question in ``lang``: "zh" takes an item's
    ``question_zh`` (the English one is kept as ``question_en``); items without
    one keep their question (some bait items are written in Chinese). The
    question text is part of each progress key, so runs in different
    languages never share saved records."""
    if lang == "en":
        return golden
    out = dict(golden)
    for part in ("retrieval", "traps"):
        out[part] = [dict(it, question=it[f"question_{lang}"], question_en=it["question"])
                     if it.get(f"question_{lang}") else it
                     for it in golden.get(part, [])]
    return out


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


# --------------------------------------------------------------------------
# Baseline arm: an ungrounded prose answer, judged sentence by sentence
# --------------------------------------------------------------------------
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# No break after a common abbreviation: "A vs. B" in a sentence was cut into
# fragments the judge could only rate as partial (2-1 run, answerable set).
_ABBREV = ("vs", "Vs", "cf", "Cf", "al", "Fig", "fig", "Figs", "figs", "Eq", "eq",
           "No", "no", "Dr", "Mr", "Ms", "approx", "e.g", "E.g", "i.e", "I.e")
_NOT_ABBREV = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREV)
_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)" + _NOT_ABBREV
                       + r"\s+(?=[A-Z0-9\"'(（「])|\n+")
# An abstention rather than a claim ("the text does not mention X"). One
# adverb may sit between the negation and the verb ("does not explicitly
# report"); missing it sent a denial to the judge in calibration.
# The 2-1 trap audit added "address" and "detail" ("does not explicitly
# address X" had been judged as a claim and rejected).
REFUSAL_HINT_RE = re.compile(
    r"not (?:\w+ly )?(?:mentioned|stated|reported|provided|specified|discussed|available|described"
    r"|addressed|detailed)"
    r"|(?:does|do|did)(?:n't| not) (?:\w+ly )?"
    r"(?:mention|provide|contain|specify|discuss|report|state|describe|include|address|detail"
    r"|break down)"
    r"|no (?:information|mention|data|details)|cannot (?:be )?(?:determined|answered|determine|answer)"
    r"|unable to|there is no (?:information|mention|evidence)"
    # Pratviel 2024 baseline run (2026-10-22): "it is not possible to determine
    # if ...", "it is unclear from the provided context whether ...", "... were
    # not directly compared ... in the provided context".
    r"|not possible to (?:determine|tell|say|answer)"
    r"|unclear (?:from|in|based on) the (?:provided|given) (?:context|text|passages|information)"
    r"|\bnot\b[^.]{0,60}\b(?:in|from) the (?:provided|given) (?:context|text|passages|information)"
    r"|未提及|沒有提到|没有提到|無法|无法|未說明|未说明|沒有提供|没有提供|並未提到|并未提到|資料中沒有|未有相關"
    # Chinese abstentions with words between the negation and the verb ("並未在
    #文獻中明確提及", "尚未在提供的文獻片段中得到詳細報告"): up to 15 characters
    # inside one clause. Found in the 12-item bait run (2026-10-16).
    r"|(?:並|并|尚|皆|均)?未[^。！？，,]{0,15}(?:提及|提到|報告|报告|說明|说明|提供|涉及|載明|载明)"
    r"|(?:沒有|没有|並無|并无)[^。！？，,]{0,12}(?:提及|提到|相關資料|相关资料|相關數據|相关数据|資訊|信息|報告|报告)",
    re.I)
# Advice about where to look ("one would need to review the full
# dissertation", "further analysis would be necessary"): neither a claim
# about the source nor an abstention, so it is not judged. The judge had
# rejected such sentences as unsupported.
META_RE = re.compile(
    r"(?:would|will|might|may) (?:need|have) to (?:review|refer|consult|look|examine|read)"
    r"|need to (?:review|refer to|consult|look (?:at|for)|read)"
    r"|it would be necessary to|for a (?:comprehensive|complete|full) understanding"
    r"|further (?:analysis|research|investigation|study|data)(?: or [\w ]+)? (?:would be|is|are|will be)"
    r" (?:needed|necessary|required)"
    r"|refer to the (?:full|original|complete)"
    # Chinese advice ("具體的改善程度需要通過實驗數據來確認", "可能需要查阅更多…").
    r"|需要(?:通過|通过|透過|透过|參考|参考|查閱|查阅|更多|進一步|进一步)[^。！？]{0,25}"
    r"(?:確認|确认|驗證|验证|資料|资料|數據|数据|研究|分析|圖表|图表)"
    r"|有待(?:進一步|进一步)",
    re.I)


def split_sentences(text: str) -> list[str]:
    """Sentences of a prose answer (Chinese or English punctuation, lines,
    list bullets). Fragments under 10 characters are dropped."""
    out = []
    for part in _SPLIT_RE.split(text or ""):
        s = _BULLET_RE.sub("", part or "").strip().strip("*").strip()
        if len(s) >= 10:
            out.append(s)
    return out


def opens_with_denial(text: str) -> bool:
    """The answer's first sentence abstains ("the text does not report X"),
    whatever follows. Secondary trap metric for the baseline arm: a strict
    refusal also requires no further claims, and the baseline often denies
    the premise, then adds related content."""
    sentences = split_sentences(text)
    return bool(sentences) and bool(REFUSAL_HINT_RE.search(sentences[0]))


def _closest(sentence: str, passages: list, embedder, k: int = 2) -> list:
    """The k context passages most similar to the sentence (cosine).

    When one of them sits in a hypotheses section, the closest passage outside
    such a section is added: a true finding worded like its hypothesis is
    closest to the hypothesis, and judging it there alone rejected it while
    the results passage that supports it went unseen (Pratviel 2024 run)."""
    if len(passages) <= k:
        return list(passages)
    import numpy as np

    from src.verify import in_hypotheses_section

    q = np.asarray(embedder.encode_queries([sentence])[0], dtype=float)
    P = np.asarray(embedder.encode_passages([p.text for p in passages]), dtype=float)
    sims = P @ q / (np.linalg.norm(P, axis=1) * np.linalg.norm(q) + 1e-9)
    ranked = [passages[i] for i in np.argsort(-sims)]
    out = ranked[:k]
    if any(in_hypotheses_section(p) for p in out):
        out += next(([p] for p in ranked[k:] if not in_hypotheses_section(p)), [])
    return out


def baseline_bundle(question: str, strategy: str, llm, verifier, k: int = 2,
                    raw: str | None = None, cached: dict | None = None):
    """Answer with the ungrounded baseline prompt over the same top-k context
    the grounded arm gets, then judge every non-abstaining sentence against
    its k closest context passages (early stop on support), as `check` does.

    ``raw`` reuses a saved answer instead of generating one; ``cached`` maps a
    sentence to its saved verdict, so only sentences not judged before go to
    the judge (see ``reused_verdicts``).

    Returns (generation, verification) shaped like the grounded arm's, so the
    same per-question record and metrics apply. ``refused`` = the answer
    abstains (refusal phrase) and makes no claim."""
    from src.generate import Claim, GenerationResult, generate
    from src.retrieve import get_retriever
    from src.verify import VerificationReport

    retriever = get_retriever(strategy)
    passages = retriever.search(question)
    if raw is None:
        raw = generate(question, passages, llm=llm, grounded=False).raw
    cached = cached or {}
    sentences = split_sentences(raw)
    claims_text = [s for s in sentences
                   if not REFUSAL_HINT_RE.search(s) and not META_RE.search(s)]
    verdicts = [cached[s] if s in cached else
                verifier.verify_statement(s, _closest(s, passages, retriever.embedder, k),
                                          context=passages)
                for s in claims_text]
    gen = GenerationResult(
        query=question, raw=raw, mode="baseline", passages=passages,
        refused=bool(REFUSAL_HINT_RE.search(raw)) and not claims_text,
        claims=[Claim(statement=s, citation_ids=v.citation_ids) for s, v in zip(claims_text, verdicts)],
    )
    return gen, VerificationReport(query=question, verdicts=verdicts)


_RAW_LIMIT = 3000   # saved baseline answers are cut here (see eval_grounding)


def reused_verdicts(record: dict) -> dict:
    """{sentence: ClaimVerdict} from a saved baseline record, for re-scoring
    its answer after a change to splitting or abstention detection without
    judging unchanged sentences again."""
    from src.verify import ClaimVerdict

    return {c["statement"]: ClaimVerdict(c["statement"], list(c["judged_against"]), c["label"],
                                         c["reason"], c["method"], dict(c.get("votes") or {}),
                                         c.get("quote_check", ""))
            for c in record.get("claims", [])}


def load_reusable(report_dir: Path, signature: str) -> dict[str, dict]:
    """Saved baseline records of run ``signature`` whose full answer was kept,
    from the live progress file or a retired one."""
    rows = {}
    for path in (report_dir / PROGRESS_NAME,
                 report_dir / f"eval_progress.done-{signature}.jsonl"):
        rows.update(_load_progress(path, signature))
    return {k: r for k, r in rows.items() if "raw" in r and len(r["raw"]) < _RAW_LIMIT}


def strict_safe(rec: dict) -> bool:
    """Safe with no partially supported claim either. The plain safe rate lets
    a partially supported claim through; the sensitivity re-judging showed
    that on premise traps such a claim can restate a hypothesis as confirmed
    or half-accept a false premise."""
    return (rec["n_unsupported"] == 0 and rec["n_disputed"] == 0 and rec["n_partial"] == 0
            and not rec["parse_failed"])


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
    baseline: bool = False, reuse_from: str = "", trap_types: list[str] | None = None,
    ids: list[str] | None = None,
) -> dict:
    """Grounding / refusal metrics. Each question's record is appended to
    eval_progress.jsonl as soon as it is scored, so an interrupted run (guard
    stop, crash, closed session) resumes where it stopped when started again
    with the same settings; ``fresh`` ignores saved records. The progress file
    is renamed once the run completes, so the next run starts fresh.

    ``reuse_from`` (baseline arm): take each question's answer from that
    earlier run's saved records and reuse its verdicts for sentences that
    split the same way; only new sentences are judged and only questions the
    earlier run did not reach are generated."""
    from src.pipeline import ask
    from src.verify import Verifier

    retrieval_items = golden["retrieval"][:limit] if limit else golden["retrieval"]
    if traps_only:
        retrieval_items = []
    trap_items = golden["traps"]
    if trap_types:      # e.g. score one trap tier that an earlier run predates
        trap_items = [it for it in trap_items if it.get("type", "far_absent") in trap_types]
    trap_items = trap_items[:limit] if limit else trap_items
    if answerable_only:
        trap_items = []
    if ids:             # a handful of questions, e.g. to inspect raw outputs
        retrieval_items = [it for it in retrieval_items if it["id"] in ids]
        trap_items = [it for it in trap_items if it["id"] in ids]

    verifier = Verifier(llm=llm)
    print(f"[grounding] verifier {verifier.describe()}")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.REPORT_DIR / PROGRESS_NAME
    arm = "baseline" if baseline else "grounded"
    if reuse_from and not baseline:
        raise ValueError("reuse_from applies to the baseline arm only")
    signature = _run_signature(strategy, llm, verifier, arm, reuse_from)
    reusable = load_reusable(config.REPORT_DIR, reuse_from) if reuse_from else {}
    if reuse_from:
        print(f"[grounding] reusing answers of run {reuse_from}: {len(reusable)} saved")
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
        old = reusable.get(key)
        if baseline and old:
            g, v = baseline_bundle(it["question"], strategy, llm, verifier,
                                   raw=old["raw"], cached=reused_verdicts(old))
        elif baseline:
            g, v = baseline_bundle(it["question"], strategy, llm, verifier)
        else:
            bundle = ask(it["question"], strategy=strategy, llm=llm, verifier=verifier)
            g, v = bundle.generation, bundle.verification
        rec = _question_record(g, v)
        if not baseline:
            # The raw XML, for format audits (missing quotes, misplaced tags).
            rec["raw"] = g.raw[:_RAW_LIMIT]
        if baseline:
            rec["raw"] = g.raw[:_RAW_LIMIT]   # the prose answer, for the refusal audit (local only)
            rec["leading_denial"] = opens_with_denial(g.raw)
            if old:
                cached = reused_verdicts(old)
                rec["reused_from"] = reuse_from
                rec["n_reused_verdicts"] = sum(c.statement in cached for c in g.claims)
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
        trap_detail.append({"id": it["id"], "type": kind, "correct": correct, "safe": safe,
                            "strict_safe": strict_safe(rec), **rec})

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
            "strict_safe_rate": _ratio(sum(d["strict_safe"] for d in sub), len(sub)),
        }

    # Complete: retire the progress file so the next run starts fresh. A
    # sampled run (--llm-limit) keeps it, so the full run reuses its questions.
    if progress_path.exists() and not limit:
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
    if baseline:
        counts["trap_leading_denial"] = stats.rate(
            sum(d.get("leading_denial", False) for d in must_refuse), len(must_refuse))

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
        **({"reuse_from": reuse_from,
            "n_reused_answers": sum("reused_from" in r for r in every),
            "n_reused_verdicts": sum(r.get("n_reused_verdicts", 0) for r in every)}
           if reuse_from else {}),
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
        arm = results["grounding"][0].get("arm", "grounded")
        A(f"## 2. Grounding / Hallucination / Refusal (LLM: `{results.get('llm_backend','?')}`, "
          f"arm: `{arm}`)")
        if arm == "baseline":
            A("")
            A("> Baseline arm: ungrounded prose answers over the same context passages, split "
              "into sentences; each non-abstaining sentence is judged against its 2 closest "
              "context passages. \"Refused\" = the answer only abstains. Claim counts are "
              "sentences; quote columns do not apply.")
            g0r = results["grounding"][0]
            if g0r.get("reuse_from"):
                A("")
                A(f"> Re-scored from run `{g0r['reuse_from']}`: {g0r['n_reused_answers']} saved "
                  f"answers reused, {g0r['n_reused_verdicts']} sentence verdicts carried over "
                  "unchanged; the remaining sentences and questions were judged in this run.")
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
            A("| Type | n | Correct | Refused | Safe (no rejected/disputed claim) "
              "| Strict safe (also no partial) |")
            A("|---|---|---|---|---|---|")
            g0c = results["grounding"][0].get("counts", {})
            for kind, t in tbt.items():
                correct = (stats.fmt(g0c[f"trap_correct:{kind}"]) if f"trap_correct:{kind}" in g0c
                           else pct(t["correct_rate"]))
                A(f"| {kind} | {t['n']} | {correct} | {pct(t['refusal_rate'])} "
                  f"| {pct(t['safe_rate'])} | {pct(t.get('strict_safe_rate'))} |")
            A("")
            A("> far_absent / near_absent: correct = refused (the topic is not in the corpus; "
              "near_absent questions are about the corpus's own studies). false_premise: "
              "correct = safe — refusing or correcting the premise with supported claims "
              "both pass. Refusal correctness above covers only the must-refuse types. "
              "Strict safe also fails a partially supported claim: on a trap built to "
              "elicit overreach, partial support means the claim goes beyond the source "
              "(e.g. a hypothesis stated as confirmed, or a premise half-accepted).")
            A("")
            if "trap_leading_denial" in g0c:
                A(f"Opens by denying the premise (must-refuse traps, baseline): "
                  f"{stats.fmt(g0c['trap_leading_denial'])}")
                A("")
                A("> Secondary metric: the first sentence abstains, whatever follows. It is "
                  "not a refusal under the shared definition (abstains and makes no claim), "
                  "which both arms are scored by.")
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
    ap.add_argument("--baseline", action="store_true",
                    help="ungrounded baseline arm: prose answers over the same context, "
                         "judged sentence by sentence (for the grounded-vs-baseline comparison)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore questions saved by an interrupted run and start over")
    ap.add_argument("--ids", nargs="+", default=None, metavar="ID",
                    help="LLM pass on these question ids only (e.g. R01 R04)")
    ap.add_argument("--trap-types", nargs="+", default=None, metavar="TYPE",
                    help="LLM pass on these trap types only (e.g. hypothesis_bait)")
    ap.add_argument("--reuse-from", default="", metavar="SIGNATURE",
                    help="baseline arm: reuse the answers and unchanged sentence verdicts "
                         "saved by that earlier run (after a splitting/abstention change)")
    ap.add_argument("--lang", choices=["en", "zh"], default="en",
                    help="ask each question in this language (zh: the items' question_zh)")
    ap.add_argument("--llm-limit", type=int, default=None,
                    help="cap #retrieval and #trap questions for the LLM pass (quick runs)")
    args = ap.parse_args()

    golden = in_language(load_golden(), args.lang)
    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lang": args.lang,
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
                               answerable_only=args.answerable_only, fresh=args.fresh,
                               baseline=args.baseline, reuse_from=args.reuse_from,
                               trap_types=args.trap_types, ids=args.ids)
            g["arm"] = "baseline" if args.baseline else "grounded"
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
