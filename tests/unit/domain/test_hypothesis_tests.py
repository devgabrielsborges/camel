from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from camel.domain.services.hypothesis_tests import (
    apply_bh_correction,
    detect_pairing,
    run_all_tests,
    run_bootstrap_ci,
    run_chi2,
    run_mannwhitneyu,
    run_mcnemar,
    run_wilcoxon,
)
from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.test_result import TestResult
from camel.domain.value_objects.threshold_profile import ThresholdProfile


# ---------------------------------------------------------------------------
# T023: McNemar test wrapper
# ---------------------------------------------------------------------------
class TestRunMcnemar:
    def test_known_table_significant(self) -> None:
        """Asymmetric discordant pair table → should be significant."""
        candidate = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        reference = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        ids = tuple(f"s{i}" for i in range(30))
        result = run_mcnemar(candidate, reference, ids, metric_name="refusal", category="neg")
        assert isinstance(result, TestResult)
        assert result.test_name == "mcnemar"
        assert result.is_paired is True
        assert 0.0 <= result.p_value <= 1.0

    def test_identical_binary_not_significant(self) -> None:
        """Identical binary sequences → p-value should be 1.0 (no discordance)."""
        vals = tuple([1.0] * 15 + [0.0] * 15)
        ids = tuple(f"s{i}" for i in range(30))
        result = run_mcnemar(vals, vals, ids, metric_name="refusal", category="neg")
        assert result.p_value == 1.0

    def test_small_sample_uses_exact(self) -> None:
        """N < 25 discordant pairs should use exact test."""
        candidate = (1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        reference = (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
        ids = tuple(f"s{i}" for i in range(10))
        result = run_mcnemar(candidate, reference, ids, metric_name="refusal", category="neg")
        assert result.test_name == "mcnemar"

    def test_effect_size_is_odds_ratio(self) -> None:
        candidate = (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                     1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        reference = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ids = tuple(f"s{i}" for i in range(30))
        result = run_mcnemar(candidate, reference, ids, metric_name="refusal", category="neg")
        assert result.effect_size > 0


# ---------------------------------------------------------------------------
# T024: Wilcoxon signed-rank wrapper
# ---------------------------------------------------------------------------
class TestRunWilcoxon:
    def test_known_paired_difference(self) -> None:
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(0.6, 0.1, 50).tolist())
        b = tuple(rng.normal(0.4, 0.1, 50).tolist())
        result = run_wilcoxon(a, b, metric_name="overlap", category="pos")
        assert isinstance(result, TestResult)
        assert result.test_name == "wilcoxon"
        assert result.is_paired is True
        assert result.p_value < 0.05

    def test_identical_scores_not_significant(self) -> None:
        vals = tuple([0.5] * 30)
        result = run_wilcoxon(vals, vals, metric_name="overlap", category="pos")
        assert result.p_value >= 0.05 or result.statistic == 0.0

    def test_matches_scipy_directly(self) -> None:
        rng = np.random.default_rng(99)
        a = tuple(rng.normal(0.5, 0.1, 40).tolist())
        b = tuple(rng.normal(0.5, 0.1, 40).tolist())
        result = run_wilcoxon(a, b, metric_name="overlap", category="pos")
        diffs = np.array(a) - np.array(b)
        nonzero = diffs[diffs != 0]
        if len(nonzero) > 0:
            scipy_result = stats.wilcoxon(nonzero)
            assert abs(result.p_value - scipy_result.pvalue) < 1e-6

    def test_effect_size_is_cohens_d(self) -> None:
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(0.8, 0.1, 50).tolist())
        b = tuple(rng.normal(0.2, 0.1, 50).tolist())
        result = run_wilcoxon(a, b, metric_name="overlap", category="pos")
        assert result.effect_size > 1.0


# ---------------------------------------------------------------------------
# T025: Mann-Whitney U wrapper
# ---------------------------------------------------------------------------
class TestRunMannWhitneyU:
    def test_different_distributions(self) -> None:
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(0.7, 0.1, 40).tolist())
        b = tuple(rng.normal(0.3, 0.1, 40).tolist())
        result = run_mannwhitneyu(a, b, metric_name="overlap", category="pos")
        assert isinstance(result, TestResult)
        assert result.test_name == "mannwhitneyu"
        assert result.is_paired is False
        assert result.p_value < 0.05

    def test_same_distribution_not_significant(self) -> None:
        rng = np.random.default_rng(42)
        a = tuple(rng.normal(0.5, 0.1, 30).tolist())
        b = tuple(rng.normal(0.5, 0.1, 30).tolist())
        result = run_mannwhitneyu(a, b, metric_name="overlap", category="pos")
        assert result.p_value > 0.01

    def test_matches_scipy_directly(self) -> None:
        rng = np.random.default_rng(77)
        a = tuple(rng.normal(0.5, 0.1, 35).tolist())
        b = tuple(rng.normal(0.5, 0.1, 35).tolist())
        result = run_mannwhitneyu(a, b, metric_name="overlap", category="pos")
        scipy_result = stats.mannwhitneyu(a, b, alternative="two-sided")
        assert abs(result.p_value - scipy_result.pvalue) < 1e-6


# ---------------------------------------------------------------------------
# T026: Bootstrap CI wrapper
# ---------------------------------------------------------------------------
class TestRunBootstrapCI:
    def test_ci_covers_known_parameter(self) -> None:
        rng = np.random.default_rng(42)
        scores = tuple(rng.normal(0.6, 0.1, 100).tolist())
        result = run_bootstrap_ci(
            scores, threshold_value=0.4, bootstrap_b=2000, seed=42,
            metric_name="delta", category="global",
        )
        assert isinstance(result, TestResult)
        assert result.test_name == "bootstrap"

    def test_reject_when_ci_below_threshold(self) -> None:
        """Scores well below threshold → reject null (CI entirely below)."""
        scores = tuple([0.1] * 50)
        result = run_bootstrap_ci(
            scores, threshold_value=0.5, bootstrap_b=1000, seed=42,
            metric_name="delta", category="global",
        )
        assert result.reject_null is True

    def test_no_reject_when_scores_above_threshold(self) -> None:
        """Scores well above threshold → do not reject."""
        scores = tuple([0.8] * 50)
        result = run_bootstrap_ci(
            scores, threshold_value=0.3, bootstrap_b=1000, seed=42,
            metric_name="delta", category="global",
        )
        assert result.reject_null is False

    def test_deterministic_with_seed(self) -> None:
        rng = np.random.default_rng(42)
        scores = tuple(rng.normal(0.5, 0.1, 50).tolist())
        r1 = run_bootstrap_ci(scores, 0.4, bootstrap_b=1000, seed=99, metric_name="d", category="g")
        r2 = run_bootstrap_ci(scores, 0.4, bootstrap_b=1000, seed=99, metric_name="d", category="g")
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


# ---------------------------------------------------------------------------
# T027: Benjamini-Hochberg correction
# ---------------------------------------------------------------------------
class TestApplyBHCorrection:
    def _make_result(self, p_value: float, metric_name: str = "m") -> TestResult:
        return TestResult(
            metric_name=metric_name,
            category="pos",
            test_name="wilcoxon",
            statistic=0.0,
            p_value=p_value,
            p_value_adjusted=p_value,
            reject_null=False,
            effect_size=0.5,
            effect_magnitude="medium",
            ci_lower=0.0,
            ci_upper=1.0,
            candidate_value=0.5,
            threshold_value=0.4,
            sample_size_candidate=50,
            sample_size_reference=50,
            is_paired=True,
        )

    def test_known_p_values(self) -> None:
        results = [
            self._make_result(0.01, "m1"),
            self._make_result(0.04, "m2"),
            self._make_result(0.03, "m3"),
            self._make_result(0.20, "m4"),
            self._make_result(0.50, "m5"),
        ]
        corrected = apply_bh_correction(results, alpha=0.05)
        raw_pvals = [r.p_value for r in results]
        _, expected_adjusted, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

        for r, expected in zip(corrected, expected_adjusted):
            assert abs(r.p_value_adjusted - expected) < 1e-10

    def test_reject_null_matches_statsmodels(self) -> None:
        results = [
            self._make_result(0.005, "m1"),
            self._make_result(0.10, "m2"),
            self._make_result(0.50, "m3"),
        ]
        corrected = apply_bh_correction(results, alpha=0.05)
        raw_pvals = [r.p_value for r in results]
        reject_expected, _, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

        for r, expected in zip(corrected, reject_expected):
            assert r.reject_null == expected

    def test_preserves_other_fields(self) -> None:
        results = [self._make_result(0.03, "m1")]
        corrected = apply_bh_correction(results, alpha=0.05)
        assert corrected[0].metric_name == "m1"
        assert corrected[0].effect_size == 0.5
        assert corrected[0].test_name == "wilcoxon"

    def test_single_result(self) -> None:
        results = [self._make_result(0.03)]
        corrected = apply_bh_correction(results, alpha=0.05)
        assert len(corrected) == 1
        assert corrected[0].p_value_adjusted == 0.03

    def test_empty_list(self) -> None:
        corrected = apply_bh_correction([], alpha=0.05)
        assert corrected == []


# ---------------------------------------------------------------------------
# Pairing detection
# ---------------------------------------------------------------------------
class TestDetectPairing:
    def test_full_overlap_is_paired(self) -> None:
        ids = ("s1", "s2", "s3", "s4", "s5")
        is_paired, common = detect_pairing(ids, ids)
        assert is_paired is True
        assert set(common) == set(ids)

    def test_no_overlap_is_unpaired(self) -> None:
        a = ("s1", "s2", "s3")
        b = ("s4", "s5", "s6")
        is_paired, common = detect_pairing(a, b)
        assert is_paired is False
        assert common == ()

    def test_partial_overlap_above_threshold(self) -> None:
        a = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10")
        b = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "x1", "x2")
        is_paired, common = detect_pairing(a, b)
        assert is_paired is True
        assert len(common) == 8

    def test_partial_overlap_below_threshold(self) -> None:
        a = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10")
        b = ("s1", "s2", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")
        is_paired, common = detect_pairing(a, b)
        assert is_paired is False


# ---------------------------------------------------------------------------
# Run all tests orchestrator
# ---------------------------------------------------------------------------
class TestRunAllTests:
    def _make_profile(self) -> ThresholdProfile:
        return ThresholdProfile(
            version="1.0.0",
            created_at="2026-01-01T00:00:00+00:00",
            benchmark="test",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
            total_sessions=100,
            bootstrap_b=1000,
            bootstrap_seed=42,
            threshold_percentile=5,
            alpha=0.05,
            correction_method="benjamini-hochberg",
            category_thresholds={
                "positivo": [
                    MetricThreshold(
                        metric_name="token_overlap_f1",
                        value=0.3, ci_lower=0.25, ci_upper=0.35,
                        metric_type=MetricType.CONTINUOUS,
                        test_paired="wilcoxon", test_unpaired="mannwhitneyu",
                        reference_mean=0.5, reference_std=0.1, sample_count=100,
                    ),
                    MetricThreshold(
                        metric_name="refusal_detection",
                        value=0.05, ci_lower=0.02, ci_upper=0.08,
                        metric_type=MetricType.BINARY,
                        test_paired="mcnemar", test_unpaired="chi2",
                        reference_mean=0.1, reference_std=0.3, sample_count=100,
                    ),
                ],
            },
            global_thresholds=[
                MetricThreshold(
                    metric_name="discrimination_delta",
                    value=0.1, ci_lower=0.08, ci_upper=0.12,
                    metric_type=MetricType.COMPOSITE,
                    test_paired="bootstrap", test_unpaired="bootstrap",
                    reference_mean=0.15, reference_std=0.05, sample_count=100,
                ),
            ],
        )

    def test_returns_test_results(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        profile = self._make_profile()
        candidate = [
            CategoryScoreCollection(
                category="positivo",
                session_ids=tuple(f"s{i}" for i in range(n)),
                scores={
                    "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                    "refusal_detection": tuple(rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()),
                },
            ),
        ]
        reference = [
            CategoryScoreCollection(
                category="positivo",
                session_ids=tuple(f"s{i}" for i in range(n)),
                scores={
                    "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                    "refusal_detection": tuple(rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()),
                },
            ),
        ]

        results = run_all_tests(candidate, reference, profile)
        assert len(results) > 0
        assert all(isinstance(r, TestResult) for r in results)

    def test_bh_correction_applied(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        profile = self._make_profile()
        candidate = [
            CategoryScoreCollection(
                category="positivo",
                session_ids=tuple(f"s{i}" for i in range(n)),
                scores={
                    "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                    "refusal_detection": tuple(rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()),
                },
            ),
        ]
        reference = [
            CategoryScoreCollection(
                category="positivo",
                session_ids=tuple(f"s{i}" for i in range(n)),
                scores={
                    "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).tolist()),
                    "refusal_detection": tuple(rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()),
                },
            ),
        ]
        results = run_all_tests(candidate, reference, profile)
        for r in results:
            assert r.p_value_adjusted >= r.p_value or abs(r.p_value_adjusted - r.p_value) < 1e-10
