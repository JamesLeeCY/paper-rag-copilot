# Measuring Whether a RAG Assistant Hallucinates — A Methodology

*A short technical write-up of the evaluation design behind the Dissertation RAG
Copilot. Written to be portfolio/interview material.*

## The problem

"Does this RAG assistant hallucinate?" is usually answered by vibes — you read a
few answers and they *feel* right. That is not evidence. For a tool meant to help
verify claims in a PhD dissertation, "feels right" is exactly the failure mode we
are trying to eliminate. So the project treats trustworthiness as something to be
**measured**, with a reproducible harness and explicit targets.

## Three questions, four metrics

We decompose "is it trustworthy?" into questions that can each be scored:

1. **Can it find the right evidence?** → Retrieval Hit Rate@k, MRR
2. **When it makes a claim, is the cited evidence real support?** → Citation
   Precision, Hallucination Rate
3. **When there is no evidence, does it admit it?** → Refusal Correctness

The third is the one people skip, and it is the most important: a system that
answers everything confidently is worse than useless for verification.

## Building a golden set without fooling yourself

The retrieval golden set is 26 natural-language questions, each anchored to the
**docx paragraph indices** that actually answer it. The key design decision:
ground truth is a *paragraph span*, not a chunk id. Chunk ids differ between the
fixed-size and section-aware strategies, so scoring on chunk ids would make the
two indexes incomparable. Scoring on "does the retrieved chunk's paragraph span
cover a target paragraph?" is strategy-agnostic — the same golden set fairly
benchmarks both. This is what makes the chunking ablation a clean A/B.

A tempting shortcut is to use a source sentence verbatim as the query. Don't:
the query then nearly equals the passage and Hit Rate inflates to ~100%
meaninglessly. The questions here are paraphrased information needs (the kind of
thing a real user types — "which method was used to align signals across
subjects?"), not copied sentences.

## Trap questions: the honesty test

Ten questions ask about topics genuinely **absent** from the dissertation —
psilocybin, EEG bands, salivary cortisol, blue-space, HRV, actigraphy, BDNF
genotyping. The corpus is fMRI-based nature-intervention research, so a faithful
system has *nothing* to cite for these. Refusal Correctness = the fraction it
correctly answers with the refusal marker rather than confabulating. These traps
are chosen to be *near* the domain (they sound plausible) so they actually test
discrimination, not keyword mismatch.

## Two prompts, one comparison

Grounding is enforced structurally, not requested politely. The generation prompt
forces every claim into `<claim citation_ids="…">`, and forbids any statement the
retrieved passages don't support — the model must emit a fixed refusal string
instead. A deliberately weak **baseline prompt** (no grounding constraints) is
kept alongside so the harness can quantify *how much* the discipline reduces
hallucination, rather than just asserting that it does.

## Verification as an independent second opinion

The verification layer re-checks each `(claim, cited passage)` pair in a **fresh
LLM call that never sees the generator's reasoning**. This matters: asking the
same context that produced a claim "is this claim right?" invites
self-confirmation bias. An optional NLI model (`roberta-large-mnli`) can run as a
cheap first-pass filter before the LLM adjudicates. Each verdict is
supported / partially_supported / unsupported, and a claim counts as grounded
only if at least one of its cited passages entails it.

## What the numbers say

Section-aware chunking reaches **100% Hit@5 / 0.952 MRR**, versus 88% / 0.794 for
the fixed-size baseline — a 12-point Hit@5 gain from respecting section and
paragraph boundaries so an argument isn't sliced mid-thought. That single result
is the whole thesis of the project in miniature: **a measurable pipeline lets you
turn a design intuition ("don't cut arguments in half") into a number, and then
defend it.**

## Closing the loop

Every knob — chunk size, embedding model, fusion weights, rerank on/off — lives
in `config.py`, and the harness re-runs against the same golden set on demand.
That "change → re-measure" loop, not any single number, is the deliverable.
