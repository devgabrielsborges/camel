from __future__ import annotations

import json

from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.threshold_profile import ThresholdProfile


def _make_metric_threshold(
    metric_name: str = "token_overlap_f1",
    metric_type: MetricType = MetricType.CONTINUOUS,
) -> MetricThreshold:
    return MetricThreshold(
        metric_name=metric_name,
        value=0.42,
        ci_lower=0.39,
        ci_upper=0.45,
        metric_type=metric_type,
        test_paired="wilcoxon",
        test_unpaired="mannwhitneyu",
        reference_mean=0.50,
        reference_std=0.12,
        sample_count=100,
    )


def _make_profile() -> ThresholdProfile:
    return ThresholdProfile(
        version="1.0.0",
        created_at="2026-05-03T00:00:00+00:00",
        benchmark="WeniEval-Benchmark-2.0.0",
        reference_models=("gpt-5.4-mini", "gpt-5.4"),
        reference_run_ids=("run_abc", "run_def"),
        total_sessions=200,
        bootstrap_b=10000,
        bootstrap_seed=42,
        threshold_percentile=5,
        alpha=0.05,
        correction_method="benjamini-hochberg",
        category_thresholds={
            "positivo": [
                _make_metric_threshold("token_overlap_f1", MetricType.CONTINUOUS),
                _make_metric_threshold("refusal_detection", MetricType.BINARY),
            ],
            "negativo": [
                _make_metric_threshold("token_overlap_f1", MetricType.CONTINUOUS),
            ],
        },
        global_thresholds=[
            _make_metric_threshold("discrimination_delta", MetricType.COMPOSITE),
        ],
    )


class TestThresholdProfileRoundTrip:
    def test_to_dict_returns_dict(self) -> None:
        profile = _make_profile()
        result = profile.to_dict()
        assert isinstance(result, dict)

    def test_from_dict_to_dict_round_trip(self) -> None:
        profile = _make_profile()
        restored = ThresholdProfile.from_dict(profile.to_dict())
        assert restored == profile

    def test_json_serialization_round_trip(self) -> None:
        profile = _make_profile()
        json_str = json.dumps(profile.to_dict())
        parsed = json.loads(json_str)
        restored = ThresholdProfile.from_dict(parsed)
        assert restored == profile

    def test_to_dict_contains_all_top_level_keys(self) -> None:
        profile = _make_profile()
        result = profile.to_dict()
        expected_keys = {
            "version",
            "created_at",
            "benchmark",
            "reference_models",
            "reference_run_ids",
            "total_sessions",
            "bootstrap_b",
            "bootstrap_seed",
            "threshold_percentile",
            "alpha",
            "correction_method",
            "category_thresholds",
            "global_thresholds",
        }
        assert set(result.keys()) == expected_keys

    def test_reference_models_serialized_as_list(self) -> None:
        profile = _make_profile()
        result = profile.to_dict()
        assert isinstance(result["reference_models"], list)
        assert result["reference_models"] == ["gpt-5.4-mini", "gpt-5.4"]

    def test_reference_models_restored_as_tuple(self) -> None:
        profile = _make_profile()
        restored = ThresholdProfile.from_dict(profile.to_dict())
        assert isinstance(restored.reference_models, tuple)

    def test_category_thresholds_preserved(self) -> None:
        profile = _make_profile()
        restored = ThresholdProfile.from_dict(profile.to_dict())
        assert set(restored.category_thresholds.keys()) == {"positivo", "negativo"}
        assert len(restored.category_thresholds["positivo"]) == 2
        assert len(restored.category_thresholds["negativo"]) == 1

    def test_global_thresholds_preserved(self) -> None:
        profile = _make_profile()
        restored = ThresholdProfile.from_dict(profile.to_dict())
        assert len(restored.global_thresholds) == 1
        assert restored.global_thresholds[0].metric_name == "discrimination_delta"

    def test_empty_thresholds_round_trip(self) -> None:
        profile = ThresholdProfile(
            version="1.0.0",
            created_at="2026-05-03T00:00:00+00:00",
            benchmark="WeniEval-Benchmark-2.0.0",
            reference_models=("gpt-5.4-mini",),
            reference_run_ids=("run_abc",),
            total_sessions=100,
            bootstrap_b=10000,
            bootstrap_seed=42,
            threshold_percentile=5,
            alpha=0.05,
            correction_method="benjamini-hochberg",
        )
        restored = ThresholdProfile.from_dict(profile.to_dict())
        assert restored.category_thresholds == {}
        assert restored.global_thresholds == []

    def test_make_timestamp_returns_iso_format(self) -> None:
        ts = ThresholdProfile.make_timestamp()
        assert "T" in ts
        assert "+" in ts or "Z" in ts
