"""End-to-end query pipeline: retrieve -> generate (grounded) -> verify.

Two entry points:
  * ``ask``   — question answering with citation-grounded, verified claims.
  * ``check`` — reverse hallucination check: paste a paragraph you wrote, get a
    per-sentence "is this supported by the library?" report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from src.retrieve import Retriever, get_retriever
from src.generate import generate, Claim, GenerationResult
from src.verify import Verifier, VerificationReport
from src.llm import LLMClient


@dataclass
class AnswerBundle:
    query: str
    generation: GenerationResult
    verification: VerificationReport
    passages: list = field(default_factory=list)


def ask(
    query: str,
    strategy: str | None = None,
    top_k: int | None = None,
    llm: LLMClient | None = None,
    verifier: Verifier | None = None,
    grounded: bool = True,
) -> AnswerBundle:
    strategy = strategy or config.CHUNK_STRATEGY
    retriever = get_retriever(strategy)
    passages = retriever.search(query, top_k=top_k)
    llm = llm or LLMClient()
    gen = generate(query, passages, llm=llm, grounded=grounded)
    verifier = verifier or Verifier(llm=llm)
    ver = verifier.verify(gen)
    return AnswerBundle(query=query, generation=gen, verification=ver, passages=passages)


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def check(
    paragraph: str,
    strategy: str | None = None,
    llm: LLMClient | None = None,
    verifier: Verifier | None = None,
) -> list[AnswerBundle]:
    """Split a written paragraph into sentences and verify each against the library.

    Each sentence is judged *directly* against its retrieved passages — it is
    not turned into a question for the generator, since verifying the
    generator's answer would say nothing about whether the user's own sentence
    is supported.
    """
    strategy = strategy or config.CHUNK_STRATEGY
    sentences = [s.strip() for s in _SENT_RE.split(paragraph.strip()) if len(s.strip()) > 15]
    llm = llm or LLMClient()
    verifier = verifier or Verifier(llm=llm)
    retriever = get_retriever(strategy)
    bundles = []
    for sent in sentences:
        passages = retriever.search(sent)
        verdict = verifier.verify_statement(sent, passages[: config.CHECK_TOP_N])
        gen = GenerationResult(
            query=sent, raw="", mode="check", passages=passages,
            claims=[Claim(statement=sent, citation_ids=verdict.citation_ids)],
        )
        ver = VerificationReport(query=sent, verdicts=[verdict])
        bundles.append(AnswerBundle(query=sent, generation=gen, verification=ver, passages=passages))
    return bundles


def format_answer(bundle: AnswerBundle) -> str:
    """Human-readable rendering for the CLI."""
    g, v = bundle.generation, bundle.verification
    lines = [f"Q: {bundle.query}", ""]
    if g.refused and not g.claims:
        lines.append(f"⚠  {config.REFUSAL_MARKER}")
    elif g.parse_failed:
        lines.append("⚠  無法解析模型輸出（既無 <claim> 也無拒答標記）")
    for claim, verdict in zip(g.claims, v.verdicts):
        badge = {
            "supported": "✔", "partially_supported": "◐",
            "disputed": "⚖", "unsupported": "✘",
        }.get(verdict.label, "?")
        lines.append(f"{badge} [{verdict.label}] {claim.statement}")
        # The verdict's ids: they differ from the claim's when a mangled
        # citation was re-attributed by its verbatim quote.
        for cid in verdict.citation_ids or claim.citation_ids:
            loc = next((p.locator() for p in bundle.passages if p.chunk_id == cid), cid)
            lines.append(f"      ↳ {loc}")
        if verdict.citation_repaired_from:
            lines.append(f"      · citation repaired (model cited {verdict.citation_repaired_from})")
        if claim.quote:
            lines.append(f"      “{claim.quote}” [quote: {verdict.quote_check}]")
        if len(verdict.votes) > 1:
            lines.append("      · votes: " + ", ".join(f"{j}={l}" for j, l in verdict.votes.items()))
        if verdict.reason:
            lines.append(f"      · verify: {verdict.reason}")
    if g.unsupported_note and config.REFUSAL_MARKER in g.unsupported_note:
        lines.append(f"\nNote: {g.unsupported_note}")
    if v.n_claims:
        lines.append(
            f"\nCitation precision: {v.citation_precision(strict=True):.0%} strict / "
            f"{v.citation_precision(strict=False):.0%} lenient | "
            f"Hallucination rate: {v.hallucination_rate():.0%}"
            + (f" | Disputed: {v.n_disputed}" if v.n_disputed else "")
        )
    return "\n".join(lines)
