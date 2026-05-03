from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation: sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)).
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(arr_a), len(arr_b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = float(np.var(arr_a, ddof=1))
    s2 = float(np.var(arr_b, ddof=1))
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-12
    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled_std)


def odds_ratio(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """Compute odds ratio from a 2x2 McNemar-style table [[a, b], [c, d]].

    OR = b / c. Applies Haldane correction (+0.5) when b or c is zero.
    """
    b = float(table[0][1])
    c = float(table[1][0])
    if b == 0 or c == 0:
        b += 0.5
        c += 0.5
    return b / c


_CONTINUOUS_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}
_BINARY_THRESHOLDS = {"small": 1.5, "medium": 2.0, "large": 3.0}


def classify_magnitude(value: float, metric_type: str) -> str:
    """Classify effect size magnitude using Cohen's conventions."""
    abs_val = abs(value)
    if metric_type == "binary":
        thresholds = _BINARY_THRESHOLDS
    else:
        thresholds = _CONTINUOUS_THRESHOLDS

    if abs_val >= thresholds["large"]:
        return "large"
    if abs_val >= thresholds["medium"]:
        return "medium"
    if abs_val >= thresholds["small"]:
        return "small"
    return "negligible"
