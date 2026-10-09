"""Judge validation set — known-answer (claim, passage) pairs for testing judges.

Every grounding metric rests on the verifier's judges, so the judges themselves
need a benchmark. This module builds one with *known* gold labels, without any
LLM, by taking real sentences from the corpus and perturbing them with rules:

  original      verbatim sentence vs its own passage      -> supported
  negation      effect direction flipped / "not" inserted -> unsupported
  number        a numeric value changed                   -> unsupported
  overclaim     hedge / association upgraded to certainty -> partially_supported
  conjunction   sentence + an unrelated claim appended    -> partially_supported
  swap_passage  sentence vs an unrelated passage          -> unsupported

Items are split dev/test *by source sentence*, so an original and its
perturbation never land on different sides (no leakage when tuning a judge's
threshold on dev and reporting on test).

Synthetic positives are verbatim, so they are easier than real paraphrased
claims. Human-labelled items in ``judge_set_human.jsonl`` (same schema, see
``judge_set_human.example.jsonl``) cover that gap and are loaded alongside.

Run:  python -m eval.judge_set [--n-sentences 60] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

import config
from src.ingest import load_chunks

SYNTHETIC_PATH = config.GOLDEN_DIR / "judge_set_synthetic.jsonl"
HUMAN_PATH = config.GOLDEN_DIR / "judge_set_human.jsonl"
REGRESSION_PATH = config.GOLDEN_DIR / "judge_set_regression.jsonl"
# Hand-written items shaped like real generator errors (paraphrased Chinese
# claims over English passages) with matched controls; local, like the others.
HARD_PATH = config.GOLDEN_DIR / "judge_set_hard.jsonl"

GOLD = {
    "original": "supported",
    "negation": "unsupported",
    "number": "unsupported",
    "overclaim": "partially_supported",
    "conjunction": "partially_supported",
    "swap_passage": "unsupported",
    "plan_to_result": "unsupported",
    "attribution_swap": "unsupported",
}
# Top-up only: drawn from sentences outside the main sample, so adding them
# never changes the items (or cached votes) of the original rotation.
TOPUP_ONLY = {"plan_to_result", "attribution_swap"}
PERTURBATIONS = [p for p in GOLD if p != "original" and p not in TOPUP_ONLY]

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WORD_RE = re.compile(r"[a-z]{4,}")

# --------------------------------------------------------------------------
# Perturbation rules (each returns the perturbed claim, or None if N/A)
# --------------------------------------------------------------------------
_FLIP_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"), ("increases", "decreases"),
    ("increasing", "decreasing"), ("higher", "lower"), ("greater", "smaller"),
    ("improved", "impaired"), ("improve", "impair"), ("improves", "impairs"),
    ("enhanced", "diminished"), ("positive", "negative"), ("positively", "negatively"),
    ("more", "less"), ("larger", "smaller"), ("stronger", "weaker"),
    ("better", "worse"), ("reduced", "increased"), ("reduces", "increases"),
    ("reduce", "increase"),
]
_FLIP = {}
for _a, _b in _FLIP_PAIRS:
    _FLIP.setdefault(_a, _b)
    _FLIP.setdefault(_b, _a)
_FLIP_RE = re.compile(r"\b(" + "|".join(sorted(_FLIP, key=len, reverse=True)) + r")\b", re.I)
_NOT_RE = re.compile(r"\b(is|are|was|were|has|have|does|did|can)\b")


def _match_case(src: str, word: str) -> str:
    return word.capitalize() if src[0].isupper() else word


def negate(s: str) -> str | None:
    m = _FLIP_RE.search(s)
    if m:
        w = m.group(1)
        return s[: m.start()] + _match_case(w, _FLIP[w.lower()]) + s[m.end():]
    m = _NOT_RE.search(s)
    if m:
        neg = "cannot" if m.group(1) == "can" else f"{m.group(1)} not"
        return s[: m.start()] + neg + s[m.end():]
    return None


_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\w)")


# Numbers that label things rather than state findings ("Figure 17", "Study 2").
_LABEL_BEFORE_RE = re.compile(
    r"\b(figure|fig\.|table|study|experiment|section|chapter|appendix|hypothesis|"
    r"step|phase|wave|session|run|component|h)\s*$", re.I)


def change_number(s: str) -> str | None:
    for m in _NUM_RE.finditer(s):
        tok = m.group(1)
        if re.fullmatch(r"(19|20)\d{2}", tok):      # years are citations, skip
            continue
        if _LABEL_BEFORE_RE.search(s[: m.start()]):
            continue
        if "." in tok:
            v, dec = float(tok), len(tok.split(".")[1])
            new = f"{(v + 0.37 if v < 0.6 else v - 0.37):.{dec}f}"
        else:
            v = int(tok)
            new = str(v * 2 + 3)
        return s[: m.start()] + new + s[m.end():]
    return None


_OVERCLAIM_RULES = [
    (re.compile(r"\b(is|are|was|were)( positively| negatively| significantly)? "
                r"(associated|correlated|linked) with\b", re.I),
     lambda m: f"{m.group(1)} shown to directly cause"),
    (re.compile(r"\b(may|might|could) ", re.I), lambda m: "will always "),
    (re.compile(r"\bsuggest(s|ed)? that\b", re.I),
     lambda m: {"s": "proves", "ed": "proved"}.get(m.group(1) or "", "prove") + " that"),
    (re.compile(r"\b(likely|possibly|potentially) ", re.I), lambda m: "certainly "),
    (re.compile(r"\bindicat(es|ed|e) that\b", re.I),
     lambda m: {"es": "proves", "ed": "proved"}.get(m.group(1).lower(), "prove") + " that"),
    (re.compile(r"\b(appears|appear|seems|seem) to\b", re.I),
     lambda m: ("is" if m.group(1).lower().endswith("s") else "are") + " proven to"),
    (re.compile(r"\bcan ", re.I), lambda m: "will always "),
    (re.compile(r"\b(partially|partly) ", re.I), lambda m: "fully "),
]


def overclaim(s: str) -> str | None:
    for pat, repl in _OVERCLAIM_RULES:
        m = pat.search(s)
        if m:
            return s[: m.start()] + repl(m) + s[m.end():]
    return None


_IRREGULAR_PAST = {
    "be": "were", "have": "had", "show": "showed", "lead": "led", "find": "found",
    "become": "became", "make": "made", "take": "took", "give": "gave", "see": "saw",
    "go": "went", "run": "ran", "begin": "began", "hold": "held", "bring": "brought",
}


def _past(verb: str) -> str:
    v = verb.lower()
    if v in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[v]
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


_HYP_LABEL_RE = re.compile(r"^(?:H\d+[a-z]?|Hypothesis \d+[a-z]?)\s*[:.]\s*", re.I)
_WILL_RE = re.compile(r"\bwill (not )?(be )?([a-z]+)\b", re.I)
# Only predictions about outcomes make a meaningful "reported finding" when
# restated in the past tense; planned procedures ("will be recruited") and
# instrument descriptions do not.
_HYPOTHESIS_CUE_RE = re.compile(
    r"^(?:H\d+[a-z]?|Hypothesis \d+[a-z]?)\s*[:.]|\bhypothesi[sz]e|\bexpect|\bpredict", re.I)
_OUTCOME_VERB_RE = re.compile(
    r"\bwill (?:not )?(?:reduce|increase|improve|enhance|elicit|show|exhibit|moderate|predict|"
    r"lead|result|decrease|demonstrate|produce|facilitate|impair|engage|activate|differ|outperform)\b", re.I)


def _lower_first(text: str) -> str:
    """Lowercase the first letter unless it starts an acronym ("VR", "ISC")."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def plan_to_result(s: str) -> str | None:
    """A hypothesis or planned assessment restated as a reported finding.

    "H1a: X will reduce Y" -> "The results showed that X reduced Y". The
    source only predicts or plans, so the claim is unsupported — the error a
    generator makes when it reads a methods/hypothesis section as results.
    """
    if not _WILL_RE.search(s):
        return None
    if not (_HYPOTHESIS_CUE_RE.search(s.strip()) or _OUTCOME_VERB_RE.search(s)):
        return None
    body = _HYP_LABEL_RE.sub("", s.strip())

    def repl(m: re.Match) -> str:
        neg, be, verb = m.group(1), m.group(2), m.group(3)
        if be:                                   # "will be assessed" -> "were assessed"
            return ("were not " if neg else "were ") + verb
        return ("did not " + verb) if neg else _past(verb)

    body = _WILL_RE.sub(repl, body, count=1)
    return "The results showed that " + _lower_first(body)


_CITE_SUBJECT_RE = re.compile(
    r"\b[A-Z][A-Za-z'\-]+(?: et al\.| and [A-Z][A-Za-z'\-]+)? \((?:19|20)\d{2}[a-z]?\)")
_CITE_PAREN_RE = re.compile(r"\s*\((?:[^()]*?(?:19|20)\d{2}[a-z]?[^()]*?)\)")
_FINDING_VERB_RE = re.compile(
    r"(?<!self-)\b(found|showed|shown|reported|demonstrated|observed|revealed|indicated)\b", re.I)
# The authors describing their own study ("we", "our") is not a cited finding.
_OWN_VOICE_RE = re.compile(r"\b(we|our)\b", re.I)
_OTHER_WORK_RE = re.compile(r"\b(studies|research|review|meta-analys\w*|trials?|authors)\b", re.I)
_SPECULATIVE_RE = re.compile(r"\b(should|would|could|may|might)\b", re.I)
_CONNECTOR_RE = re.compile(
    r"^(?:For example|For instance|Notably|Similarly|Likewise|Moreover|In addition|"
    r"Consistent with this|In line with this|Importantly),\s*", re.I)
_AUTHOR_LEFT_RE = re.compile(r"\b[A-Z][a-zÀ-ɏ'\-]+ (?:et al\.|and colleagues)")


def attribution_swap(s: str) -> str | None:
    """A cited study's finding restated as this study's own finding.

    "Berman et al. (2008) demonstrated that X" -> "This study demonstrated
    that X". The passage attributes X to another study, so the claim is
    unsupported — the error of reading a literature review as results.
    """
    if not _FINDING_VERB_RE.search(s) or _SPECULATIVE_RE.search(s) or _OWN_VOICE_RE.search(s):
        return None
    s = _CONNECTOR_RE.sub("", s.strip())
    m = _CITE_SUBJECT_RE.search(s)
    if m:
        out = s[: m.start()] + ("This study" if m.start() == 0 else "this study") + s[m.end():]
        out = _CITE_PAREN_RE.sub("", out)
    elif _CITE_PAREN_RE.search(s):
        body = _CITE_PAREN_RE.sub("", s).strip()
        if _OTHER_WORK_RE.search(body):    # "studies found…" keeps it attributed elsewhere
            return None
        out = "In this study, " + _lower_first(body)
    else:
        return None
    # Another author still named in the sentence keeps the attribution right.
    if _AUTHOR_LEFT_RE.search(out):
        return None
    return out[:1].upper() + out[1:]


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------
def _overlap(a: str, b: str) -> float:
    wa, wb = set(_WORD_RE.findall(a.lower())), set(_WORD_RE.findall(b.lower()))
    return len(wa & wb) / len(wa) if wa else 1.0


def _split_of(group: str) -> str:
    return "dev" if int(hashlib.md5(group.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _candidate_sentences(chunks) -> list[tuple[str, object]]:
    out = []
    for c in chunks:
        for s in _SENT_RE.split(c.text.replace("\n", " ")):
            s = s.strip()
            if 60 <= len(s) <= 350 and len(s.split()) >= 8 and not s.startswith(("Table", "Figure")):
                out.append((s, c))
    return out


def _distant(chunks, c, sentence, rng):
    """A chunk from another top-level section that shares few content words."""
    top = (c.section_number or "?").split(".")[0]
    pool = [o for o in chunks
            if (o.section_number or "?").split(".")[0] != top and _overlap(sentence, o.text) < 0.2]
    return rng.choice(pool) if pool else None


def build(n_sentences: int = 60, seed: int = 42, overclaim_extra: int = 30,
          stage_extra: int = 25) -> list[dict]:
    rng = random.Random(seed)
    chunks = load_chunks("section")
    cands = _candidate_sentences(chunks)
    rng.shuffle(cands)

    items, used = [], {p: 0 for p in PERTURBATIONS}
    for gi, (sent, c) in enumerate(cands[:n_sentences]):
        group = f"S{gi:03d}"
        base = {"group": group, "split": _split_of(group), "source": "synthetic"}
        items.append({**base, "id": f"{group}-original", "perturbation": "original",
                      "gold_label": GOLD["original"], "claim": sent,
                      "passage": c.text, "chunk_id": c.chunk_id})

        # One perturbation per sentence, rotating to the least-used applicable
        # type so the set stays balanced across error kinds.
        options = {}
        if (x := negate(sent)) and x != sent:
            options["negation"] = (x, c)
        if (x := change_number(sent)) and x != sent:
            options["number"] = (x, c)
        if (x := overclaim(sent)) and x != sent:
            options["overclaim"] = (x, c)
        far = _distant(chunks, c, sent, rng)
        if far:
            options["swap_passage"] = (sent, far)
            other = [s for s, oc in cands if oc.chunk_id == far.chunk_id]
            if other:
                tail = rng.choice(other)
                options["conjunction"] = (
                    f"{sent.rstrip('.')}; moreover, {tail[0].lower()}{tail[1:]}", c)
        if not options:
            continue
        kind = min(options, key=lambda k: (used[k], k))
        used[kind] += 1
        claim, passage_chunk = options[kind]
        items.append({**base, "id": f"{group}-{kind}", "perturbation": kind,
                      "gold_label": GOLD[kind], "claim": claim,
                      "passage": passage_chunk.text, "chunk_id": passage_chunk.chunk_id})

    # Overclaims apply to few sentences, so the rotation above leaves too few
    # (and a lopsided dev/test split). Top up from the unused sentences.
    n_extra = 0
    taken: set[int] = set()
    for xi, (sent, c) in enumerate(cands[n_sentences:]):
        if n_extra >= overclaim_extra:
            break
        x = overclaim(sent)
        if not x or x == sent:
            continue
        group = f"X{xi:03d}"
        items.append({"group": group, "split": _split_of(group), "source": "synthetic",
                      "id": f"{group}-overclaim", "perturbation": "overclaim",
                      "gold_label": GOLD["overclaim"], "claim": x,
                      "passage": c.text, "chunk_id": c.chunk_id})
        taken.add(xi)
        n_extra += 1

    # Study-stage and attribution errors (the generator's failures on real
    # corpora) are rare per sentence: top up each from unused sentences.
    for kind, fn, prefix, cap in (("plan_to_result", plan_to_result, "P", stage_extra),
                                  ("attribution_swap", attribution_swap, "A", stage_extra)):
        n = 0
        for xi, (sent, c) in enumerate(cands[n_sentences:]):
            if n >= cap:
                break
            if xi in taken:
                continue
            x = fn(sent)
            if not x or x == sent:
                continue
            group = f"{prefix}{xi:03d}"
            items.append({"group": group, "split": _split_of(group), "source": "synthetic",
                          "id": f"{group}-{kind}", "perturbation": kind,
                          "gold_label": GOLD[kind], "claim": x,
                          "passage": c.text, "chunk_id": c.chunk_id})
            taken.add(xi)
            n += 1
    return items


def load_items(include_human: bool = True) -> list[dict]:
    """Synthetic items, plus human-labelled and regression items when present.

    Regression items are real generator outputs that a judge once got wrong,
    pinned with their verified label so every later prompt or judge change is
    checked against them.
    """
    items = [json.loads(l) for l in SYNTHETIC_PATH.open(encoding="utf-8")]
    if include_human:
        for path, source in ((HUMAN_PATH, "human"), (REGRESSION_PATH, "regression"),
                             (HARD_PATH, "hard")):
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    it = json.loads(line)
                    it.setdefault("source", source)
                    it.setdefault("perturbation", source)
                    it.setdefault("split", _split_of(it.get("group", it["id"])))
                    items.append(it)
    return items


def main():
    ap = argparse.ArgumentParser(description="Build the synthetic judge validation set")
    ap.add_argument("--n-sentences", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = build(args.n_sentences, args.seed)
    with SYNTHETIC_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {}
    for it in items:
        counts[it["perturbation"]] = counts.get(it["perturbation"], 0) + 1
    splits = {s: sum(it["split"] == s for it in items) for s in ("dev", "test")}
    print(f"[judge-set] {len(items)} items -> {SYNTHETIC_PATH}")
    print(f"            by type: {counts}")
    print(f"            by split: {splits}")


if __name__ == "__main__":
    main()
