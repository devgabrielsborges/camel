from __future__ import annotations

from camel.domain.services.verdict import Verdict
from camel.domain.value_objects.statistical_verdict_result import StatisticalVerdictResult
from camel.domain.value_objects.test_result import TestResult

_ACTIONABLE_MAGNITUDES = {"medium", "large"}


def compute_statistical_verdict(
    test_results: list[TestResult],
    critical_metrics: set[tuple[str, str]],
    *,
    alpha: float = 0.05,
    correction_method: str = "benjamini-hochberg",
    profile_version: str = "1.0.0",
) -> StatisticalVerdictResult:
    """Compute a statistical verdict from BH-corrected test results.

    A metric triggers a critical failure when:
    - reject_null is True (statistically significant after BH correction)
    - effect_magnitude is "medium" or "large" (practically significant)
    - (metric_name, category) is in the critical_metrics set
    """
    critical_failures: list[TestResult] = []
    reasons: list[str] = []

    for r in test_results:
        key = (r.metric_name, r.category)
        if key not in critical_metrics:
            continue
        if not r.reject_null:
            continue
        if r.effect_magnitude in _ACTIONABLE_MAGNITUDES:
            critical_failures.append(r)
            reasons.append(
                f"{r.metric_name} ({r.category}): significant (p_adj={r.p_value_adjusted:.4f}), "
                f"effect={r.effect_magnitude} ({r.effect_size:.3f})"
            )
        elif r.effect_magnitude == "small":
            reasons.append(
                f"{r.metric_name} ({r.category}): significant but small effect "
                f"(p_adj={r.p_value_adjusted:.4f}, d={r.effect_size:.3f}) — inconclusive signal"
            )

    if not critical_failures:
        has_inconclusive_signal = any(
            r.reject_null and r.effect_magnitude == "small"
            and (r.metric_name, r.category) in critical_metrics
            for r in test_results
        )
        if has_inconclusive_signal:
            verdict = Verdict.INCONCLUSIVE
        else:
            verdict = Verdict.CAPABLE
            if not reasons:
                reasons.append("All critical metrics within reference thresholds")
    else:
        verdict = Verdict.NOT_CAPABLE

    return StatisticalVerdictResult(
        verdict=verdict,
        test_results=test_results,
        critical_failures=critical_failures,
        reasons=reasons,
        threshold_profile_version=profile_version,
        alpha=alpha,
        correction_method=correction_method,
    )
