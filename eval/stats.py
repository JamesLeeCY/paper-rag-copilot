"""Confidence intervals for the small-sample rates in the reports.

Every headline rate here is k successes out of n with n often 6-80, so a point
estimate alone overstates what is known: 1 of 20 and 0 of 20 are both
consistent with a true rate of several percent. The Wilson score interval is
used because it stays inside [0, 1] and behaves at 0% and 100%, where the
normal approximation collapses to a zero-width interval.
"""
from __future__ import annotations

import math

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for k/n, or None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict:
    """{"k", "n", "rate", "ci": [lo, hi]} for storing alongside results."""
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}


def fmt(r: dict | None, bold: bool = False) -> str:
    """'84% [67–93] (26/31)' — rate, 95% CI in percentage points, counts."""
    if not r or not r.get("n"):
        return "n/a"
    lo, hi = r["ci"]
    head = f"{r['rate']:.0%}"
    if bold:
        head = f"**{head}**"
    return f"{head} [{lo * 100:.0f}–{hi * 100:.0f}] ({r['k']}/{r['n']})"
