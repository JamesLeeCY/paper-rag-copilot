"""Wilson confidence intervals used in the reports. Run: python -m pytest tests"""
import pytest

from eval import stats


@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 20, 0.0, 0.1611),      # 0 of 20 still allows ~16%
    (20, 20, 0.8389, 1.0),
    (3, 55, 0.0187, 0.1482),
    (10, 20, 0.2993, 0.7007),  # symmetric around 50%
])
def test_wilson_reference_values(k, n, lo, hi):
    got = stats.wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-3) and got[1] == pytest.approx(hi, abs=1e-3)


def test_wilson_stays_in_bounds_and_contains_the_estimate():
    for n in (1, 6, 31, 83):
        for k in range(n + 1):
            lo, hi = stats.wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_empty_sample():
    assert stats.wilson(0, 0) is None
    assert stats.rate(0, 0) == {"k": 0, "n": 0, "rate": None, "ci": None}
    assert stats.fmt(stats.rate(0, 0)) == "n/a"


def test_format():
    assert stats.fmt(stats.rate(3, 55)) == "5% [2–15] (3/55)"
    assert stats.fmt(stats.rate(3, 55), bold=True).startswith("**5%**")
