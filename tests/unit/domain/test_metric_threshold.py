from __future__ import annotations

from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType


def _make_threshold(**overrides: object) -> MetricThreshold:
    defaults = {
        "metric_name": "token_overlap_f1",
        "value": 0.42,
        "ci_lower": 0.39,
        "ci_upper": 0.45,
        "metric_type": MetricType.CONTINUOUS,
        "test_paired": "wilcoxon",
        "test_unpaired": "mannwhitneyu",
        "reference_mean": 0.50,
        "reference_std": 0.12,
        "sample_count": 100,
    }
    defaults.update(overrides)
    return MetricThreshold(**defaults)  # type: ignore[arg-type]


class TestMetricThresholdConstruction:
    def test_ci_bounds_valid(self) -> None:
        t = _make_threshold(ci_lower=0.39, value=0.42, ci_upper=0.45)
        assert t.ci_lower <= t.value <= t.ci_upper

    def test_ci_lower_equals_value(self) -> None:
        t = _make_threshold(ci_lower=0.42, value=0.42, ci_upper=0.45)
        assert t.ci_lower <= t.value <= t.ci_upper

    def test_ci_upper_equals_value(self) -> None:
        t = _make_threshold(ci_lower=0.39, value=0.45, ci_upper=0.45)
        assert t.ci_lower <= t.value <= t.ci_upper

    def test_sample_count_positive(self) -> None:
        t = _make_threshold(sample_count=1)
        assert t.sample_count > 0

    def test_metric_type_continuous(self) -> None:
        t = _make_threshold(metric_type=MetricType.CONTINUOUS)
        assert t.metric_type == MetricType.CONTINUOUS

    def test_metric_type_binary(self) -> None:
        t = _make_threshold(metric_type=MetricType.BINARY)
        assert t.metric_type == MetricType.BINARY

    def test_metric_type_composite(self) -> None:
        t = _make_threshold(metric_type=MetricType.COMPOSITE)
        assert t.metric_type == MetricType.COMPOSITE


class TestMetricThresholdSerialization:
    def test_to_dict_from_dict_round_trip(self) -> None:
        original = _make_threshold()
        restored = MetricThreshold.from_dict(original.to_dict())
        assert restored == original

    def test_to_dict_metric_type_is_string(self) -> None:
        t = _make_threshold()
        d = t.to_dict()
        assert isinstance(d["metric_type"], str)
        assert d["metric_type"] == "continuous"

    def test_from_dict_restores_metric_type_enum(self) -> None:
        t = _make_threshold()
        restored = MetricThreshold.from_dict(t.to_dict())
        assert isinstance(restored.metric_type, MetricType)
        assert restored.metric_type == MetricType.CONTINUOUS


class TestMetricThresholdImmutability:
    def test_frozen_dataclass(self) -> None:
        t = _make_threshold()
        try:
            t.value = 0.99  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
