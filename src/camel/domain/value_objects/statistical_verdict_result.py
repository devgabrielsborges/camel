from __future__ import annotations

from dataclasses import dataclass, field

from camel.domain.services.verdict import Verdict
from camel.domain.value_objects.test_result import TestResult


@dataclass(frozen=True)
class StatisticalVerdictResult:
    verdict: Verdict
    test_results: list[TestResult] = field(default_factory=list)
    critical_failures: list[TestResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    threshold_profile_version: str = ""
    alpha: float = 0.05
    correction_method: str = "benjamini-hochberg"
