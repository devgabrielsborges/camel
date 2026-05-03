from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from camel.domain.value_objects.metric_type import MetricType


@dataclass(frozen=True)
class MetricThreshold:
    metric_name: str
    value: float
    ci_lower: float
    ci_upper: float
    metric_type: MetricType
    test_paired: str
    test_unpaired: str
    reference_mean: float
    reference_std: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "metric_type": str(self.metric_type),
            "test_paired": self.test_paired,
            "test_unpaired": self.test_unpaired,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricThreshold:
        return cls(
            metric_name=str(data["metric_name"]),
            value=float(data["value"]),
            ci_lower=float(data["ci_lower"]),
            ci_upper=float(data["ci_upper"]),
            metric_type=MetricType(str(data["metric_type"])),
            test_paired=str(data["test_paired"]),
            test_unpaired=str(data["test_unpaired"]),
            reference_mean=float(data["reference_mean"]),
            reference_std=float(data["reference_std"]),
            sample_count=int(data["sample_count"]),
        )
