from __future__ import annotations

from camel.domain.services.statistical_verdict import compute_statistical_verdict
from camel.domain.services.verdict import Verdict
from camel.domain.value_objects.statistical_verdict_result import StatisticalVerdictResult
from camel.domain.value_objects.test_result import TestResult


def _make_test_result(
    reject_null: bool = False,
    effect_magnitude: str = "negligible",
    metric_name: str = "token_overlap_f1",
    category: str = "positivo",
    p_value_adjusted: float = 0.5,
) -> TestResult:
    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="wilcoxon",
        statistic=0.0,
        p_value=0.03,
        p_value_adjusted=p_value_adjusted,
        reject_null=reject_null,
        effect_size=0.5,
        effect_magnitude=effect_magnitude,
        ci_lower=0.0,
        ci_upper=1.0,
        candidate_value=0.5,
        threshold_value=0.4,
        sample_size_candidate=50,
        sample_size_reference=50,
        is_paired=True,
    )


_CRITICAL = {
    ("token_overlap_f1", "positivo"),
    ("token_overlap_f1", "negativo"),
    ("refusal_detection", "negativo"),
    ("discrimination_delta", "global"),
}


class TestComputeStatisticalVerdict:
    def test_capable_when_no_critical_failures(self) -> None:
        results = [
            _make_test_result(
                reject_null=False, metric_name="token_overlap_f1", category="positivo"
            ),
            _make_test_result(
                reject_null=False, metric_name="refusal_detection", category="negativo"
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert isinstance(verdict, StatisticalVerdictResult)
        assert verdict.verdict == Verdict.CAPABLE
        assert len(verdict.critical_failures) == 0

    def test_not_capable_with_large_effect_critical_failure(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="large",
                metric_name="token_overlap_f1",
                category="positivo",
                p_value_adjusted=0.01,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.NOT_CAPABLE
        assert len(verdict.critical_failures) == 1

    def test_not_capable_with_medium_effect_critical_failure(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="medium",
                metric_name="token_overlap_f1",
                category="positivo",
                p_value_adjusted=0.01,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.NOT_CAPABLE

    def test_inconclusive_with_small_effect_critical_failure(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="small",
                metric_name="token_overlap_f1",
                category="positivo",
                p_value_adjusted=0.04,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.INCONCLUSIVE

    def test_capable_when_non_critical_metric_fails(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="large",
                metric_name="groundedness",
                category="positivo",
                p_value_adjusted=0.01,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.CAPABLE
        assert len(verdict.critical_failures) == 0

    def test_reasons_populated(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="large",
                metric_name="token_overlap_f1",
                category="positivo",
                p_value_adjusted=0.005,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert len(verdict.reasons) > 0

    def test_empty_results_is_capable(self) -> None:
        verdict = compute_statistical_verdict([], _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.CAPABLE

    def test_all_test_results_preserved(self) -> None:
        results = [
            _make_test_result(metric_name="token_overlap_f1", category="positivo"),
            _make_test_result(metric_name="refusal_detection", category="negativo"),
            _make_test_result(metric_name="groundedness", category="positivo"),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert len(verdict.test_results) == 3

    def test_multiple_critical_failures(self) -> None:
        results = [
            _make_test_result(
                reject_null=True,
                effect_magnitude="large",
                metric_name="token_overlap_f1",
                category="positivo",
                p_value_adjusted=0.01,
            ),
            _make_test_result(
                reject_null=True,
                effect_magnitude="medium",
                metric_name="refusal_detection",
                category="negativo",
                p_value_adjusted=0.02,
            ),
        ]
        verdict = compute_statistical_verdict(results, _CRITICAL, alpha=0.05)
        assert verdict.verdict == Verdict.NOT_CAPABLE
        assert len(verdict.critical_failures) == 2
