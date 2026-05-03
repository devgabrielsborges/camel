from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from camel.domain.value_objects.metric_threshold import MetricThreshold


@dataclass(frozen=True)
class ThresholdProfile:
    version: str
    created_at: str
    benchmark: str
    reference_models: tuple[str, ...]
    reference_run_ids: tuple[str, ...]
    total_sessions: int
    bootstrap_b: int
    bootstrap_seed: int
    threshold_percentile: int
    alpha: float
    correction_method: str
    category_thresholds: dict[str, list[MetricThreshold]] = field(default_factory=dict)
    global_thresholds: list[MetricThreshold] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        cat_thresholds: dict[str, list[dict[str, object]]] = {}
        for category, thresholds in self.category_thresholds.items():
            cat_thresholds[category] = [t.to_dict() for t in thresholds]

        return {
            "version": self.version,
            "created_at": self.created_at,
            "benchmark": self.benchmark,
            "reference_models": list(self.reference_models),
            "reference_run_ids": list(self.reference_run_ids),
            "total_sessions": self.total_sessions,
            "bootstrap_b": self.bootstrap_b,
            "bootstrap_seed": self.bootstrap_seed,
            "threshold_percentile": self.threshold_percentile,
            "alpha": self.alpha,
            "correction_method": self.correction_method,
            "category_thresholds": cat_thresholds,
            "global_thresholds": [t.to_dict() for t in self.global_thresholds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdProfile:
        cat_thresholds: dict[str, list[MetricThreshold]] = {}
        for category, threshold_list in data.get("category_thresholds", {}).items():
            cat_thresholds[category] = [
                MetricThreshold.from_dict(t) for t in threshold_list
            ]

        global_thresholds = [
            MetricThreshold.from_dict(t) for t in data.get("global_thresholds", [])
        ]

        return cls(
            version=str(data["version"]),
            created_at=str(data["created_at"]),
            benchmark=str(data["benchmark"]),
            reference_models=tuple(data["reference_models"]),
            reference_run_ids=tuple(data["reference_run_ids"]),
            total_sessions=int(data["total_sessions"]),
            bootstrap_b=int(data["bootstrap_b"]),
            bootstrap_seed=int(data["bootstrap_seed"]),
            threshold_percentile=int(data["threshold_percentile"]),
            alpha=float(data["alpha"]),
            correction_method=str(data["correction_method"]),
            category_thresholds=cat_thresholds,
            global_thresholds=global_thresholds,
        )

    @staticmethod
    def make_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
