"""check-cost simulation: no LLM, no corpus. Run: python -m pytest tests"""
from types import SimpleNamespace as NS

from eval import check_cost as C


def verdicts(**labels):
    return {cid: {"label": lab, "seconds": 10.0} for cid, lab in labels.items()}


def test_simulate_stops_early_on_support():
    v = verdicts(a="partially_supported", b="supported", c="unsupported")
    assert C.simulate(["a", "b", "c"], v) == ("supported", 2, 20.0)


def test_simulate_keeps_the_best_label_without_support():
    v = verdicts(a="unsupported", b="partially_supported", c="disputed")
    assert C.simulate(["a", "b", "c"], v) == ("partially_supported", 3, 30.0)


def test_strategies_cover_retrieval_and_similarity_orders():
    ps = [NS(chunk_id=c) for c in "abcde"]
    by_sim = [NS(chunk_id=c) for c in "edcba"]
    s = C.strategies(ps, by_sim)
    assert s["retrieval_top3 (current)"] == ["a", "b", "c"]
    assert s["similarity_top2"] == ["e", "d"] and s["retrieval_top1"] == ["a"]
