from __future__ import annotations

import numpy as np

from camel.domain.services.threshold_derivation import derive_thresholds
from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.threshold_profile import ThresholdProfile


def _make_collection(
    category: str,
    n: int = 200,
    seed: int = 42,
) -> CategoryScoreCollection:
    rng = np.random.default_rng(seed)
    session_ids = tuple(f"s_{i}" for i in range(n))
    return CategoryScoreCollection(
        category=category,
        session_ids=session_ids,
        scores={
            "token_overlap_f1": tuple(rng.normal(0.5, 0.1, n).clip(0, 1).tolist()),
            "refusal_detection": tuple(rng.choice([0.0, 1.0], n, p=[0.9, 0.1]).tolist()),
        },
    )


class TestDeriveThresholds:
    def test_returns_threshold_profile(self) -> None:
        collections = [_make_collection("positivo"), _make_collection("negativo", seed=99)]
        profile = derive_thresholds(
            collections,
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        assert isinstance(profile, ThresholdProfile)

    def test_produces_category_thresholds(self) -> None:
        collections = [_make_collection("positivo"), _make_collection("negativo", seed=99)]
        profile = derive_thresholds(
            collections,
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        assert "positivo" in profile.category_thresholds
        assert "negativo" in profile.category_thresholds

    def test_threshold_within_expected_range(self) -> None:
        """5th percentile of N(0.5, 0.1) ≈ 0.335. Bootstrap threshold should be near that."""
        collections = [_make_collection("positivo", n=500, seed=42)]
        profile = derive_thresholds(
            collections,
            bootstrap_b=5000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        overlap_thresholds = [
            t
            for t in profile.category_thresholds["positivo"]
            if t.metric_name == "token_overlap_f1"
        ]
        assert len(overlap_thresholds) == 1
        th = overlap_thresholds[0]
        assert 0.20 < th.value < 0.55, f"Threshold {th.value} outside expected range"

    def test_deterministic_with_fixed_seed(self) -> None:
        collections = [_make_collection("positivo")]
        kwargs = dict(
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        p1 = derive_thresholds(collections, **kwargs)
        p2 = derive_thresholds(collections, **kwargs)
        for t1, t2 in zip(
            p1.category_thresholds["positivo"],
            p2.category_thresholds["positivo"],
        ):
            assert t1.value == t2.value
            assert t1.ci_lower == t2.ci_lower
            assert t1.ci_upper == t2.ci_upper

    def test_ci_bounds_contain_threshold(self) -> None:
        collections = [_make_collection("positivo")]
        profile = derive_thresholds(
            collections,
            bootstrap_b=2000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        for t in profile.category_thresholds["positivo"]:
            assert (
                t.ci_lower <= t.value <= t.ci_upper
            ), f"{t.metric_name}: ci_lower={t.ci_lower}, value={t.value}, ci_upper={t.ci_upper}"

    def test_binary_metric_threshold_in_zero_one(self) -> None:
        collections = [_make_collection("positivo")]
        profile = derive_thresholds(
            collections,
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        refusal_thresholds = [
            t
            for t in profile.category_thresholds["positivo"]
            if t.metric_name == "refusal_detection"
        ]
        assert len(refusal_thresholds) == 1
        th = refusal_thresholds[0]
        assert 0.0 <= th.value <= 1.0

    def test_metadata_preserved(self) -> None:
        collections = [_make_collection("positivo")]
        profile = derive_thresholds(
            collections,
            bootstrap_b=2000,
            seed=99,
            percentile=10,
            alpha=0.01,
            benchmark="my-benchmark",
            reference_models=("m1", "m2"),
            reference_run_ids=("r1", "r2"),
        )
        assert profile.bootstrap_b == 2000
        assert profile.bootstrap_seed == 99
        assert profile.threshold_percentile == 10
        assert profile.alpha == 0.01
        assert profile.reference_models == ("m1", "m2")
        assert profile.reference_run_ids == ("r1", "r2")
        assert profile.benchmark == "my-benchmark"

    def test_global_discrimination_delta(self) -> None:
        """When both positivo and negativo are present, should produce a global threshold."""
        collections = [_make_collection("positivo"), _make_collection("negativo", seed=99)]
        profile = derive_thresholds(
            collections,
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        delta_thresholds = [
            t for t in profile.global_thresholds if t.metric_name == "discrimination_delta"
        ]
        assert len(delta_thresholds) == 1
        assert delta_thresholds[0].metric_type == MetricType.COMPOSITE

    def test_sample_count_matches_input(self) -> None:
        n = 150
        collections = [_make_collection("positivo", n=n)]
        profile = derive_thresholds(
            collections,
            bootstrap_b=1000,
            seed=42,
            percentile=5,
            alpha=0.05,
            benchmark="test-bench",
            reference_models=("model-a",),
            reference_run_ids=("run-a",),
        )
        for t in profile.category_thresholds["positivo"]:
            assert t.sample_count == n
