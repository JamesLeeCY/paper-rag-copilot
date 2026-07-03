"""Generation Layer — citation-grounded prompting.

The generator is *forced* to structure every claim with the chunk_id(s) that
support it, and to refuse (emit config.REFUSAL_MARKER) when the retrieved
context does not support a claim. A deliberately weak "baseline" prompt with no
grounding constraints is also provided so the evaluation harness can quantify
how much the grounding discipline reduces hallucination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from src.llm import LLMClient
from src.retrieve import Passage


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
SYSTEM_GROUNDED = f"""你是一個嚴格的學術文獻查證助理。你只能根據下方提供的「檢索段落」回答問題。

規則：
1. 每一個論點（claim）都必須用一個 <claim> 標籤，並在 citation_ids 屬性列出支持它的段落 chunk_id（可多個，逗號分隔）。
2. 不可以使用任何檢索段落以外的知識、常識或推論來補足論點。
3. 如果檢索到的段落無法直接支持某個論點，禁止生成該論點；改為在 <unsupported_note> 中誠實說明：「{config.REFUSAL_MARKER}」。
4. 若問題完全沒有任何段落可支持，整個 <answer> 內不得有任何 <claim>，只在 <unsupported_note> 寫「{config.REFUSAL_MARKER}」。
5. 論點內容要貼近原文證據強度，不得誇大（例如把「相關」寫成「造成」）。

只輸出以下 XML，不要有其他文字：
<answer>
  <claim citation_ids="chunk_id,chunk_id">論點內容</claim>
  ...
  <unsupported_note>（若有查無依據的部分，在此列出；若無則留空）</unsupported_note>
</answer>"""

SYSTEM_BASELINE = """You are a helpful research assistant. Answer the user's
question about the dissertation as completely and fluently as you can. Write a
few sentences of prose."""


@dataclass
class Claim:
    statement: str
    citation_ids: list[str]


@dataclass
class GenerationResult:
    query: str
    raw: str
    claims: list[Claim] = field(default_factory=list)
    unsupported_note: str = ""
    refused: bool = False
    mode: str = "grounded"       # "grounded" | "baseline"
    passages: list[Passage] = field(default_factory=list)


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------
def build_context(passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        sec = f"§{p.section_number} {p.section}".strip()
        blocks.append(
            f"[chunk_id: {p.chunk_id}] (來源: {p.source}, {sec})\n{p.text}"
        )
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------
_CLAIM_RE = re.compile(
    r"<claim[^>]*citation_ids=\"([^\"]*)\"[^>]*>(.*?)</claim>", re.DOTALL
)
_NOTE_RE = re.compile(r"<unsupported_note>(.*?)</unsupported_note>", re.DOTALL)


def parse_grounded(raw: str, query: str) -> GenerationResult:
    claims = []
    for ids, body in _CLAIM_RE.findall(raw):
        cid_list = [c.strip() for c in ids.split(",") if c.strip()]
        claims.append(Claim(statement=body.strip(), citation_ids=cid_list))
    note_m = _NOTE_RE.search(raw)
    note = note_m.group(1).strip() if note_m else ""
    refused = (len(claims) == 0) or (config.REFUSAL_MARKER in note and len(claims) == 0)
    return GenerationResult(
        query=query, raw=raw, claims=claims, unsupported_note=note,
        refused=refused, mode="grounded",
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate(
    query: str,
    passages: list[Passage],
    llm: LLMClient | None = None,
    grounded: bool = True,
) -> GenerationResult:
    llm = llm or LLMClient()
    context = build_context(passages)

    if grounded:
        user = f"檢索段落：\n\n{context}\n\n---\n\n問題：{query}"
        if not llm.available:
            return _mock_grounded(query, passages)
        raw = llm.complete(SYSTEM_GROUNDED, user)
        res = parse_grounded(raw, query)
    else:
        user = f"Context passages:\n\n{context}\n\n---\n\nQuestion: {query}"
        if not llm.available:
            res = GenerationResult(query=query, raw="[mock baseline]", mode="baseline")
            res.claims = [Claim(statement="[mock baseline answer]", citation_ids=[])]
            res.passages = passages
            return res
        raw = llm.complete(SYSTEM_BASELINE, user, temperature=0.3)
        res = GenerationResult(
            query=query, raw=raw, mode="baseline",
            claims=[Claim(statement=raw.strip(), citation_ids=[])],
        )

    res.passages = passages
    return res


def _mock_grounded(query: str, passages: list[Passage]) -> GenerationResult:
    """Deterministic stand-in when no API key is present: turns the top passage
    into a single cited claim so the pipeline shape is exercised in tests."""
    if not passages:
        return GenerationResult(
            query=query, raw="[mock]", refused=True, mode="grounded",
            unsupported_note=config.REFUSAL_MARKER,
        )
    top = passages[0]
    snippet = top.text[:200].replace("\n", " ")
    res = GenerationResult(
        query=query,
        raw="[mock grounded — set ANTHROPIC_API_KEY for real generation]",
        mode="grounded",
        claims=[Claim(statement=f"(mock) {snippet}", citation_ids=[top.chunk_id])],
    )
    res.passages = passages
    return res
