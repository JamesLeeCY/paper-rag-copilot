"""Verification Layer — independent second-pass entailment check.

For each (claim, cited passage) pair produced by the generator, this layer asks
— in a *fresh* LLM call that never sees the generator's reasoning, to avoid
self-confirmation bias — whether the passage actually supports the claim, and
labels it supported / partially_supported / unsupported.

Checks run as a cascade, cheapest first:

  0. Quote grounding (no model): the generator must attach a verbatim <quote>
     from the cited passage. A quote that cannot be found in the passage is a
     fabricated citation and the claim is rejected without any LLM call.
  1. Judge panel: one or more LLM judges (``config.JUDGES``), ideally from
     different model families so their errors are less correlated, each vote
     independently; votes are combined by ``config.PANEL_RULE``. Disagreement
     yields ``disputed`` rather than silently picking a side.
  2. Fallbacks when no judge is reachable: optional NLI, then a lexical-overlap
     heuristic so the report still populates.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher

import config
from src.llm import LLMClient
from src.generate import GenerationResult, Claim


# Labels a single judge may emit.
LABELS = ("supported", "partially_supported", "unsupported")
# Final verdict ranking; "disputed" (judges disagree) sits just above unsupported.
_RANK = {"supported": 3, "partially_supported": 2, "disputed": 1, "unsupported": 0}

VERIFY_SYSTEM_V1 = """你是一個嚴格的 entailment 判斷器。給你一個「論點」和一段「原文」，
你只判斷：這段原文是否直接支持這個論點？

只輸出 JSON（不要多餘文字）：
{"label": "supported" | "partially_supported" | "unsupported", "reason": "簡短理由"}

判斷準則：
- supported：原文明確、直接支持論點的全部內容。
- partially_supported：原文只支持論點的一部分，或證據強度弱於論點的宣稱。
- unsupported：原文與論點無關，或原文並未提供支持該論點的證據。
不要用原文以外的常識來補足。"""

# v2 adds an explicit evidence-strength check. The judge validation set showed
# overclaims ("may" -> "will always", "suggests" -> "proves") were the blind
# spot of every judge under v1.
VERIFY_SYSTEM_V2 = """你是一個嚴格的 entailment 判斷器。給你一個「論點」和一段「原文」，
你只判斷：這段原文是否直接支持這個論點？

請依序檢查：
1. 內容：論點的每一部分是否都出現在原文中？數字、效果方向（增加／減少）、對象是否完全一致？
2. 證據強度：論點的確定程度是否高於原文？
   - 原文的保留用語：may, might, could, suggest, indicate, associated with, correlated with, linked to, potentially, possibly, likely, preliminary
   - 論點的過度宣稱：will, always, prove, demonstrate, cause, directly, certainly, definitely, all
   - 原文說「相關」而論點說「造成」，或原文說「可能」而論點說「必定／證明」，都屬於論點較強。
3. 範圍：原文限定的樣本、條件或情境，論點是否擴大成普遍結論？

只輸出 JSON（不要多餘文字）：
{"strength": "same" | "claim_stronger" | "claim_weaker", "label": "supported" | "partially_supported" | "unsupported", "reason": "簡短理由"}

判斷準則：
- supported：內容完全一致，且論點的確定程度與範圍不超過原文。
- partially_supported：原文只支持論點的一部分；或論點的確定程度、範圍超過原文（strength 為 claim_stronger 時，label 不可為 supported）。
- unsupported：數字或效果方向與原文矛盾、原文與論點無關，或原文並未提供支持該論點的證據。
不要用原文以外的常識來補足。"""

VERIFY_PROMPTS = {"v1": VERIFY_SYSTEM_V1, "v2": VERIFY_SYSTEM_V2}
VERIFY_SYSTEM = VERIFY_PROMPTS[config.VERIFY_PROMPT]


@dataclass
class ClaimVerdict:
    statement: str
    citation_ids: list[str]
    label: str             # one of _RANK
    reason: str
    method: str            # "quote" | "llm" | "panel" | "nli" | "lexical"
    votes: dict = field(default_factory=dict)   # judge name -> label
    quote_check: str = ""  # "verbatim" | "near" | "not_found" | "missing" | ""
    # Original ids when an unresolvable citation was re-attributed by its quote.
    citation_repaired_from: list = field(default_factory=list)


@dataclass
class VerificationReport:
    query: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.verdicts)

    def _count(self, label: str) -> int:
        return sum(v.label == label for v in self.verdicts)

    @property
    def n_supported(self) -> int:
        return self._count("supported")

    @property
    def n_partial(self) -> int:
        return self._count("partially_supported")

    @property
    def n_disputed(self) -> int:
        return self._count("disputed")

    @property
    def n_unsupported(self) -> int:
        return self._count("unsupported")

    def n_quote(self, status: str) -> int:
        return sum(v.quote_check == status for v in self.verdicts)

    @property
    def n_citation_repaired(self) -> int:
        return sum(bool(v.citation_repaired_from) for v in self.verdicts)

    @property
    def n_panel_judged(self) -> int:
        return sum(len(v.votes) >= 2 for v in self.verdicts)

    @property
    def n_panel_agreed(self) -> int:
        return sum(len(v.votes) >= 2 and len(set(v.votes.values())) == 1 for v in self.verdicts)

    def citation_precision(self, strict: bool = True) -> float:
        """Share of claims whose citation supports them.

        strict=True counts only ``supported`` (what the spec's ≥95% target
        means); strict=False also accepts ``partially_supported``. ``disputed``
        never counts as support.
        """
        if not self.verdicts:
            return 0.0
        ok = self.n_supported if strict else self.n_supported + self.n_partial
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


# --------------------------------------------------------------------------
# Stage 0: quote grounding
# --------------------------------------------------------------------------
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def _norm(text: str) -> str:
    text = text.translate(_QUOTE_CHARS).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .…\"'")


def check_quote(quote: str, passages: list[str]) -> str:
    """Is ``quote`` (near-)verbatim in any of ``passages``?

    Returns "verbatim", "near" (≥ QUOTE_MATCH_THRESHOLD of its characters align
    in order with the passage, ignoring alignment fragments shorter than 4
    characters so scattered letters cannot add up), "not_found", or "missing".
    """
    q = _norm(quote)
    if not q:
        return "missing"
    best = 0.0
    for p in passages:
        pn = _norm(p)
        if q in pn:
            return "verbatim"
        blocks = SequenceMatcher(None, q, pn, autojunk=False).get_matching_blocks()
        best = max(best, sum(b.size for b in blocks if b.size >= 4) / len(q))
    return "near" if best >= config.QUOTE_MATCH_THRESHOLD else "not_found"


# --------------------------------------------------------------------------
# Stage 1: judge panel
# --------------------------------------------------------------------------
def _parse_judge_spec(spec: str) -> LLMClient:
    backend, _, model = spec.partition(":")
    return LLMClient(backend=backend, model=model or None)


def aggregate_votes(labels: list[str], rule: str | None = None) -> str:
    """Combine judge labels: agreement wins; otherwise apply the panel rule.

    "disputed" is reserved for disagreement on the accept/reject boundary
    (some judges say supported, others do not) — the only case worth a human
    look. When every judge rejects and they differ only on severity
    (partially_supported vs unsupported), the majority label wins, ties going
    to the stricter "unsupported".
    """
    rule = rule or config.PANEL_RULE
    counts = Counter(labels)
    top, n_top = counts.most_common(1)[0]
    if n_top == len(labels):
        return top
    if rule == "majority" and n_top > len(labels) / 2:
        return top
    if "supported" not in counts:
        if counts["unsupported"] >= counts["partially_supported"]:
            return "unsupported"
        return "partially_supported"
    return "disputed"


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
    def __init__(
        self,
        llm: LLMClient | None = None,
        use_nli: bool = False,
        judges: list[LLMClient] | None = None,
    ):
        self.llm = llm or LLMClient()
        self.use_nli = use_nli
        self._nli = None
        if judges is None:
            judges = [_parse_judge_spec(s) for s in config.JUDGES] or [self.llm]
        # Unreachable judges are dropped. If none is left, judge with the
        # generator's own model before resorting to NLI / lexical fallbacks.
        self.judges = [j for j in judges if j.available]
        if not self.judges and self.llm.available:
            print("[verify] no configured judge available; using the generator's model")
            self.judges = [self.llm]
        elif len(self.judges) < len(judges):
            print(f"[verify] {len(judges) - len(self.judges)} judge(s) unavailable; "
                  f"panel runs with {len(self.judges)}")

    def describe(self) -> str:
        names = ", ".join(j.describe() for j in self.judges) or "none (fallback)"
        return f"judges=[{names}] rule={config.PANEL_RULE}"

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

    @staticmethod
    def _llm_vote(
        llm: LLMClient, statement: str, passage: str, prompt: str | None = None
    ) -> tuple[str, str] | None:
        """One judge's independent verdict; it sees only the claim and passage."""
        raw = llm.complete(
            prompt or VERIFY_SYSTEM,
            f"原文：\n{passage}\n\n論點：\n{statement}",
            max_tokens=300,
            json_mode=True,
        )
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        label = data.get("label", "unsupported")
        if label not in LABELS:
            label = "unsupported"
        # Guard against a self-contradicting judge: it says the claim overstates
        # the source yet still labels it supported.
        if data.get("strength") == "claim_stronger" and label == "supported":
            label = "partially_supported"
        return label, data.get("reason", "")

    def _judge(self, statement: str, passage: str) -> ClaimVerdict:
        if not passage:
            return ClaimVerdict(statement, [], "unsupported", "cited chunk not in context", "lexical")

        votes: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for judge in self.judges:
            vote = self._llm_vote(judge, statement, passage)
            if vote:
                votes[judge.describe()], reasons[judge.describe()] = vote
        if votes:
            label = aggregate_votes(list(votes.values()))
            if len(votes) == 1:
                return ClaimVerdict(statement, [], label, next(iter(reasons.values())), "llm", votes)
            if label == "disputed":
                reason = " | ".join(f"{name}: {votes[name]} — {reasons[name]}" for name in votes)
            else:
                reason = next(r for name, r in reasons.items() if votes[name] == label)
            return ClaimVerdict(statement, [], label, reason, "panel", votes)

        nli = self._nli_label(statement, passage)
        if nli:
            return ClaimVerdict(statement, [], nli[0], nli[1], "nli")
        label, reason = _lexical_label(statement, passage)
        return ClaimVerdict(statement, [], label, reason, "lexical")

    def verify_statement(self, statement: str, passages: list) -> ClaimVerdict:
        """Judge a user-written statement directly against retrieved passages.

        Used by the reverse check: the statement itself is the claim, so no
        generation step sits in between. The best label over ``passages`` wins;
        stops early on the first fully supporting passage.
        """
        best: ClaimVerdict | None = None
        for p in passages:
            v = self._judge(statement, p.text)
            v.citation_ids = [p.chunk_id]
            if best is None or _RANK[v.label] > _RANK[best.label]:
                best = v
            if best.label == "supported":
                break
        if best is None:
            best = ClaimVerdict(statement, [], "unsupported", "no passages retrieved", "lexical")
        return best

    def _verify_claim(self, claim: Claim, result: GenerationResult) -> ClaimVerdict:
        # Stage 0: the quote must actually appear in one of the cited passages.
        citation_ids = claim.citation_ids
        repaired = False
        cited = [t for t in (_passage_text(result, c) for c in citation_ids) if t]
        # Skipped when no cited id resolves to a passage: that is a bad citation
        # id (the judges mark it "cited chunk not in context"), not a bad quote.
        quote_status = check_quote(claim.quote, cited) if cited else ""
        if claim.quote and (not cited or quote_status == "not_found"):
            # The cited id is mangled ("chunk_id_0003") or points at the wrong
            # passage. If the verbatim quote sits in exactly one retrieved
            # passage, that passage is the deterministic source: re-attribute
            # and record the repair. A quote found nowhere stays a fabrication.
            hits = [p for p in result.passages
                    if check_quote(claim.quote, [p.text]) in ("verbatim", "near")]
            if len(hits) == 1:
                citation_ids, cited, repaired = [hits[0].chunk_id], [hits[0].text], True
                quote_status = check_quote(claim.quote, cited)
        if quote_status == "not_found":
            return ClaimVerdict(
                claim.statement, claim.citation_ids, "unsupported",
                "quote not found in cited passage(s) — fabricated citation",
                "quote", quote_check=quote_status,
            )
        if quote_status == "missing" and config.QUOTE_REQUIRED:
            return ClaimVerdict(
                claim.statement, claim.citation_ids, "unsupported",
                "no supporting quote given (QUOTE_REQUIRED)",
                "quote", quote_check=quote_status,
            )

        # Stage 1: judges. A claim is supported if *any* cited passage supports it.
        best: ClaimVerdict | None = None
        for cid in citation_ids or [""]:
            v = self._judge(claim.statement, _passage_text(result, cid))
            if best is None or _RANK[v.label] > _RANK[best.label]:
                best = v
        best.citation_ids = citation_ids
        best.quote_check = quote_status
        if repaired:
            best.citation_repaired_from = claim.citation_ids
            best.reason = (f"citation id {claim.citation_ids} not in context; "
                           f"re-attributed to {citation_ids[0]} by verbatim quote. {best.reason}")
        return best

    def verify(self, result: GenerationResult) -> VerificationReport:
        report = VerificationReport(query=result.query)
        for claim in result.claims:
            report.verdicts.append(self._verify_claim(claim, result))
        return report


def report_to_dict(report: VerificationReport) -> dict:
    return {
        "query": report.query,
        "n_claims": report.n_claims,
        "citation_precision_strict": round(report.citation_precision(strict=True), 4),
        "citation_precision_lenient": round(report.citation_precision(strict=False), 4),
        "hallucination_rate": round(report.hallucination_rate(), 4),
        "verdicts": [asdict(v) for v in report.verdicts],
    }
