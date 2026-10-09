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
1. 每一個論點（claim）都必須用一個 <claim> 標籤，並在 citation_ids 屬性列出支持它的段落編號（可多個，逗號分隔）。段落編號就是每段檢索段落開頭方括號內的字串（例如 c_section_0003），必須原樣照抄，不可改寫或自行編造。
2. 不可以使用任何檢索段落以外的知識、常識或推論來補足論點。
3. 如果檢索到的段落無法直接支持某個論點，禁止生成該論點；改為在 <unsupported_note> 中誠實說明：「{config.REFUSAL_MARKER}」。
4. 若問題完全沒有任何段落可支持，整個 <answer> 內不得有任何 <claim>，只在 <unsupported_note> 寫「{config.REFUSAL_MARKER}」。
5. 論點內容要貼近原文證據強度，不得誇大（例如把「相關」寫成「造成」）。
6. 每個 <claim> 內必須附一個 <quote>，逐字複製被引段落中最直接支持該論點的一句原文（保持原文語言，不可改寫、翻譯或摘要）。

只輸出以下 XML，不要有其他文字：
<answer>
  <claim citation_ids="段落編號,段落編號">論點內容<quote>逐字原文句子</quote></claim>
  ...
  <unsupported_note>（若有查無依據的部分，在此列出；若無則留空）</unsupported_note>
</answer>"""

SYSTEM_GROUNDED_V2 = f"""你是一個嚴格的學術文獻查證助理。你只能根據下方提供的「檢索段落」回答問題。

規則：
1. 每一個論點（claim）都必須用一個 <claim> 標籤，並在 citation_ids 屬性列出支持它的段落編號（可多個，逗號分隔）。段落編號就是每段檢索段落開頭方括號內的字串（例如 c_section_0003），必須原樣照抄，不可改寫或自行編造。
2. 不可以使用任何檢索段落以外的知識、常識或推論來補足論點。
3. 如果檢索到的段落無法直接支持某個論點，禁止生成該論點；改為在 <unsupported_note> 中誠實說明：「{config.REFUSAL_MARKER}」。
4. 若問題完全沒有任何段落可支持，整個 <answer> 內不得有任何 <claim>，只在 <unsupported_note> 寫「{config.REFUSAL_MARKER}」。這個拒答標記只能出現在 <unsupported_note> 內，絕不可放進 <claim>。
5. 論點內容要貼近原文證據強度，不得誇大（例如把「相關」寫成「造成」）。
6. 每個 <claim> 內必須附一個 <quote>，逐字複製被引段落中最直接支持該論點的一句原文。即使論點用中文寫，<quote> 也必須保持段落的原文語言（英文段落就照抄英文），不可翻譯、改寫或摘要，也不可把不相鄰的句子拼在一起。
7. 每段檢索段落開頭標有「段落類型」。類型為「研究假設」或「研究方法」的段落，描述的是研究的預測、計畫或程序，不是研究結果：引用它們時只能寫成「研究假設……」「研究預計……」「研究採用……」，絕不可寫成「研究發現／結果顯示／證實了……」。要陳述研究結果，只能引用類型為「研究結果」或「討論」的段落。
8. 如果段落中的發現屬於被引用的其他研究（例如「Smith et al. (2020) found ...」），論點必須寫明是該研究的發現（例如「Smith 等人（2020）的研究發現……」），不可寫成本研究的結果。
9. 如果段落明確寫出否定的事實（例如某項分析沒有顯著差異、某件事沒有做、某結果未被觀察到），這也是有依據的答案：請照實寫成論點並附上引文，不要拒答。

只輸出以下 XML，不要有其他文字：
<answer>
  <claim citation_ids="段落編號,段落編號">論點內容<quote>逐字原文句子</quote></claim>
  ...
  <unsupported_note>（若有查無依據的部分，在此列出；若無則留空）</unsupported_note>
</answer>"""

GROUNDED_PROMPTS = {"v1": SYSTEM_GROUNDED, "v2": SYSTEM_GROUNDED_V2}

SYSTEM_BASELINE = """You are a helpful research assistant. Answer the user's
question about the dissertation as completely and fluently as you can. Write a
few sentences of prose."""


@dataclass
class Claim:
    statement: str
    citation_ids: list[str]
    quote: str = ""              # verbatim supporting sentence ("" if none given)


@dataclass
class GenerationResult:
    query: str
    raw: str
    claims: list[Claim] = field(default_factory=list)
    unsupported_note: str = ""
    refused: bool = False
    # True when the model produced neither a <claim> nor the refusal marker —
    # malformed output, which must not be scored as a correct refusal.
    parse_failed: bool = False
    # True when the model wrapped the refusal marker in a <claim> instead of
    # <unsupported_note>: a format error, counted separately, not a claim.
    misplaced_refusal: bool = False
    mode: str = "grounded"       # "grounded" | "baseline" | "check"
    passages: list[Passage] = field(default_factory=list)


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------
def build_context(passages: list[Passage], version: str = "v1") -> str:
    blocks = []
    for p in passages:
        if version == "v1":
            sec = f"§{p.section_number} {p.section}".strip()
            blocks.append(f"[{p.chunk_id}] (來源: {p.source}, {sec})\n{p.text}")
        else:
            heading = " > ".join(p.section_path) or f"§{p.section_number} {p.section}".strip()
            blocks.append(f"[{p.chunk_id}] (來源: {p.source}；章節: {heading}；"
                          f"段落類型: {passage_type(p)})\n{p.text}")
    return "\n\n---\n\n".join(blocks)


# Passage type from the heading path, innermost heading first (a "Hypotheses"
# subsection inside a methods chapter is a hypothesis passage). Checked in
# order; the first match wins.
_PASSAGE_TYPES = (
    ("研究假設", re.compile(r"hypothes[ie]s|研究假設|假設", re.I)),
    ("研究結果", re.compile(r"\bresults?\b|findings|研究結果|結果", re.I)),
    ("討論", re.compile(r"discussion|conclusions?|limitations?|implications?|討論|結論", re.I)),
    ("研究方法", re.compile(r"method|materials|participants|procedure|design|measure|"
                        r"protocol|acquisition|analysis|stimul|instrument|研究方法|方法", re.I)),
    ("背景", re.compile(r"introduction|background|literature|review|theory|theoretical|"
                      r"research questions?|aims?|objectives?|緒論|背景|文獻", re.I)),
)


def _match(heading: str, skip_hypotheses: bool = True) -> str | None:
    for label, rx in _PASSAGE_TYPES:
        if skip_hypotheses and label == "研究假設":
            continue
        if rx.search(heading):
            return label
    return None


def passage_type(p: Passage) -> str:
    """研究假設 / 研究結果 / 討論 / 研究方法 / 背景 / 其他, from the headings.

    The chapter (outermost heading) decides, because subsection titles carry
    misleading words ("1.3 What Determines the Efficacy of Design ..." sits in
    the Introduction). One exception: an innermost "Hypotheses" heading marks a
    hypothesis passage wherever it sits — and only the innermost, since a
    chapter titled "... and Hypotheses" also holds background. If the chapter
    title is unlabelled, inner headings decide, innermost first."""
    path = [h for h in (list(p.section_path) or [p.section]) if h]
    if not path:
        return "其他"
    if _PASSAGE_TYPES[0][1].search(path[-1]):
        return "研究假設"
    for heading in [path[0]] + path[:0:-1]:
        label = _match(heading)
        if label:
            return label
    return "其他"


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------
# A claim ends at </claim> — or, when a model drops the closing tag, at the
# next <claim>, at <unsupported_note>, at </answer>, or at the end of output.
_CLAIM_RE = re.compile(
    r"<claim[^>]*citation_ids=\"([^\"]*)\"[^>]*>(.*?)"
    r"(?:</claim>|(?=<claim\b|<unsupported_note>|</answer>)|\Z)",
    re.DOTALL,
)
_NOTE_RE = re.compile(r"<unsupported_note>(.*?)</unsupported_note>", re.DOTALL)
_QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.DOTALL)
_CID_PREFIX_RE = re.compile(r"^\s*\[?\s*chunk_id\s*:\s*", re.IGNORECASE)


def parse_grounded(raw: str, query: str) -> GenerationResult:
    claims = []
    for ids, body in _CLAIM_RE.findall(raw):
        # Models sometimes copy the context label too ("chunk_id: c_section_0003").
        cid_list = [_CID_PREFIX_RE.sub("", c).strip(" []") for c in ids.split(",") if c.strip()]
        quote_m = _QUOTE_RE.search(body)
        quote = quote_m.group(1).strip() if quote_m else ""
        statement = _QUOTE_RE.sub("", body).strip()
        claims.append(Claim(statement=statement, citation_ids=cid_list, quote=quote))
    # A <claim> that only restates the refusal marker asserts nothing: drop it
    # and flag the format error (it was the model's way of refusing).
    marker_only = [c for c in claims if _is_marker_only(c.statement)]
    claims = [c for c in claims if not _is_marker_only(c.statement)]
    note_m = _NOTE_RE.search(raw)
    note = note_m.group(1).strip() if note_m else ""
    # A refusal must be explicit: no claims AND the refusal marker emitted. The
    # marker is matched anywhere in the output, not only inside
    # <unsupported_note>, since small local models often drift from the XML.
    has_marker = config.REFUSAL_MARKER in raw
    refused = not claims and has_marker
    parse_failed = not claims and not has_marker
    return GenerationResult(
        query=query, raw=raw, claims=claims, unsupported_note=note,
        refused=refused, parse_failed=parse_failed,
        misplaced_refusal=bool(marker_only), mode="grounded",
    )


def _is_marker_only(statement: str) -> bool:
    """True when a claim's text is just the refusal marker (± punctuation)."""
    core = statement.strip().strip("。.,，、:：;；!！ \"'「」『』（）()")
    return core == config.REFUSAL_MARKER


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
    version = config.GENERATOR_PROMPT if grounded else "v1"
    context = build_context(passages, version)

    if grounded:
        user = f"檢索段落：\n\n{context}\n\n---\n\n問題：{query}"
        if not llm.available:
            return _mock_grounded(query, passages)
        raw = llm.complete(GROUNDED_PROMPTS[version], user)
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
