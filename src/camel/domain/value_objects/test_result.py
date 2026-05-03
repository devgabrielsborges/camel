from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestResult:
    metric_name: str
    category: str
    test_name: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    reject_null: bool
    effect_size: float
    effect_magnitude: str
    ci_lower: float
    ci_upper: float
    candidate_value: float
    threshold_value: float
    sample_size_candidate: int
    sample_size_reference: int
    is_paired: bool
