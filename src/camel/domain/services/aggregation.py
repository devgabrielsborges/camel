from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from camel.domain.value_objects.score import Score


@dataclass(frozen=True)
class AggregatedMetric:
    scorer_name: str
    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class CategoryBreakdown:
    category: str
    metrics: list[AggregatedMetric] = field(default_factory=list)


def _to_float(value: float | bool) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def aggregate_scores(scores: list[Score]) -> list[AggregatedMetric]:
    by_scorer: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_scorer[s.scorer_name].append(_to_float(s.value))

    return [
        AggregatedMetric(
            scorer_name=name,
            mean=round(_mean(vals), 4),
            std=round(_std(vals), 4),
            count=len(vals),
        )
        for name, vals in sorted(by_scorer.items())
    ]


def aggregate_by_category(
    scores_by_category: dict[str, list[Score]],
) -> list[CategoryBreakdown]:
    return [
        CategoryBreakdown(
            category=cat,
            metrics=aggregate_scores(cat_scores),
        )
        for cat, cat_scores in sorted(scores_by_category.items())
    ]
