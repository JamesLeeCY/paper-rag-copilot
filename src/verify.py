"""Verification Layer — independent second-pass entailment check.

For each (claim, cited passage) pair produced by the generator, this layer asks
— in a *fresh* LLM call that never sees the generator's reasoning, to avoid
self-confirmation bias — whether the passage actually supports the claim, and
labels it supported / partially_supported / unsupported.

An optional lightweight NLI model (roberta-large-mnli) can run first as a cheap
filter; the LLM judgement is the finer second layer. Both are optional: without
an API key / NLI model the layer degrades to a lexical-overlap heuristic so the
report still populates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field

import config
from src.llm import LLMClient
from src.generate import GenerationResult, Claim


LABELS = ("supported", "partially_supported", "unsupported")

VERIFY_SYSTEM = """你是一個嚴格的 entailment 判斷器。給你一個「論點」和一段「原文」，
你只判斷：這段原文是否直接支持這個論點？

只輸出 JSON（不要多餘文字）：
{"label": "supported" | "partially_supported" | "unsupported", "reason": "簡短理由"}

判斷準則：
- supported：原文明確、直接支持論點的全部內容。
- partially_supported：原文只支持論點的一部分，或證據強度弱於論點的宣稱。
- unsupported：原文與論點無關，或原文並未提供支持該論點的證據。
不要用原文以外的常識來補足。"""


@dataclass
class ClaimVerdict:
    statement: str
    citation_ids: list[str]
    label: str
    reason: str
    method: str            # "llm" | "nli" | "lexical"


@dataclass
class VerificationReport:
    query: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.verdicts)

    @property
    def n_supported(self) -> int:
        return sum(v.label == "supported" for v in self.verdicts)

    @property
    def n_unsupported(self) -> int:
        return sum(v.label == "unsupported" for v in self.verdicts)

    def citation_precision(self) -> float:
        if not self.verdicts:
            return 0.0
        ok = sum(v.label in ("supported", "partially_supported") for v in self.verdicts)
        return ok / len(self.verdicts)

    def hallucination_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return self.n_unsupported / len(self.verdicts)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _passage_text(result: GenerationResult, chunk_id: str) -> str:
    for p in result.passages:
        if p.chunk_id == chunk_id:
            return p.text
    return ""


def _lexical_label(statement: str, passage: str) -> tuple[str, str]:
    """Fallback judge: token-overlap heuristic (no model needed)."""
    st = set(re.findall(r"[a-z]{4,}", statement.lower()))
    pa = set(re.findall(r"[a-z]{4,}", passage.lower()))
    if not st:
        return "unsupported", "no content tokens in claim"
    overlap = len(st & pa) / len(st)
    if overlap >= 0.5:
        return "supported", f"lexical overlap {overlap:.2f}"
    if overlap >= 0.2:
        return "partially_supported", f"lexical overlap {overlap:.2f}"
    return "unsupported", f"lexical overlap {overlap:.2f}"


class Verifier:
    def __init__(self, llm: LLMClient | None = None, use_nli: bool = False):
        self.llm = llm or LLMClient()
        self.use_nli = use_nli
        self._nli = None

    def _nli_label(self, statement: str, passage: str) -> tuple[str, str] | None:
        if not self.use_nli:
            return None
        try:
            if self._nli is None:
                from transformers import pipeline

                self._nli = pipeline("text-classification", model="roberta-large-mnli")
            # premise=passage, hypothesis=claim
            out = self._nli(f"{passage} </s></s> {statement}", top_k=None)
            scores = {d["label"].lower(): d["score"] for d in out}
            if scores.get("entailment", 0) >= 0.6:
                return "supported", f"NLI entailment {scores['entailment']:.2f}"
            if scores.get("contradiction", 0) >= 0.6:
                return "unsupported", f"NLI contradiction {scores['contradiction']:.2f}"
            return "partially_supported", "NLI neutral"
        except Exception:
            return None

    def _judge(self, statement: str, passage: str) -> ClaimVerdict:
        if not passage:
            return ClaimVerdict(statement, [], "unsupported", "cited chunk not in context", "lexical")
        # Layer 1 (optional): NLI cheap filter
        nli = self._nli_label(statement, passage)
        # Layer 2: LLM judge (authoritative when available)
        if self.llm.available:
            raw = self.llm.complete(
                VERIFY_SYSTEM,
                f"原文：\n{passage}\n\n論點：\n{statement}",
                max_tokens=300,
                json_mode=True,
            )
            m = _JSON_RE.search(raw)
            if m:
                try:
                    data = json.loads(m.group(0))
                    label = data.get("label", "unsupported")
                    if label not in LABELS:
                        label = "unsupported"
                    return ClaimVerdict(statement, [], label, data.get("reason", ""), "llm")
                except json.JSONDecodeError:
                    pass
        if nli:
            return ClaimVerdict(statement, [], nli[0], nli[1], "nli")
        label, reason = _lexical_label(statement, passage)
        return ClaimVerdict(statement, [], label, reason, "lexical")

    def verify(self, result: GenerationResult) -> VerificationReport:
        report = VerificationReport(query=result.query)
        for claim in result.claims:
            # A claim is supported if *any* of its cited passages support it.
            best: ClaimVerdict | None = None
            rank = {"supported": 2, "partially_supported": 1, "unsupported": 0}
            for cid in claim.citation_ids or [""]:
                passage = _passage_text(result, cid)
                v = self._judge(claim.statement, passage)
                v.citation_ids = claim.citation_ids
                if best is None or rank[v.label] > rank[best.label]:
                    best = v
            report.verdicts.append(best)
        return report


def report_to_dict(report: VerificationReport) -> dict:
    return {
        "query": report.query,
        "n_claims": report.n_claims,
        "citation_precision": round(report.citation_precision(), 4),
        "hallucination_rate": round(report.hallucination_rate(), 4),
        "verdicts": [asdict(v) for v in report.verdicts],
    }
