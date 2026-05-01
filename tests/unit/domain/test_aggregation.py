from __future__ import annotations

from camel.domain.services.aggregation import (
    aggregate_by_category,
    aggregate_scores,
)
from camel.domain.value_objects.score import Score


class TestAggregateScores:
    def test_single_scorer(self) -> None:
        scores = [
            Score(scorer_name="f1", value=0.8),
            Score(scorer_name="f1", value=0.6),
        ]
        result = aggregate_scores(scores)
        assert len(result) == 1
        assert result[0].scorer_name == "f1"
        assert result[0].mean == 0.7
        assert result[0].count == 2

    def test_multiple_scorers(self) -> None:
        scores = [
            Score(scorer_name="f1", value=0.8),
            Score(scorer_name="f1", value=0.6),
            Score(scorer_name="exact", value=True),
            Score(scorer_name="exact", value=False),
        ]
        result = aggregate_scores(scores)
        assert len(result) == 2
        names = {m.scorer_name for m in result}
        assert names == {"exact", "f1"}

    def test_bool_values_converted(self) -> None:
        scores = [
            Score(scorer_name="match", value=True),
            Score(scorer_name="match", value=False),
            Score(scorer_name="match", value=True),
        ]
        result = aggregate_scores(scores)
        assert len(result) == 1
        metric = result[0]
        assert abs(metric.mean - 0.6667) < 0.01

    def test_empty_scores(self) -> None:
        result = aggregate_scores([])
        assert result == []

    def test_std_zero_for_identical(self) -> None:
        scores = [
            Score(scorer_name="f1", value=0.5),
            Score(scorer_name="f1", value=0.5),
        ]
        result = aggregate_scores(scores)
        assert result[0].std == 0.0

    def test_std_positive_for_variance(self) -> None:
        scores = [
            Score(scorer_name="f1", value=0.0),
            Score(scorer_name="f1", value=1.0),
        ]
        result = aggregate_scores(scores)
        assert result[0].std > 0.0


class TestAggregateByCategory:
    def test_single_category(self) -> None:
        data = {
            "positivo": [
                Score(scorer_name="f1", value=0.9),
                Score(scorer_name="f1", value=0.7),
            ]
        }
        result = aggregate_by_category(data)
        assert len(result) == 1
        assert result[0].category == "positivo"
        assert len(result[0].metrics) == 1
        assert result[0].metrics[0].mean == 0.8

    def test_multiple_categories(self) -> None:
        data = {
            "positivo": [Score(scorer_name="f1", value=0.9)],
            "negativo": [Score(scorer_name="f1", value=0.3)],
        }
        result = aggregate_by_category(data)
        assert len(result) == 2
        categories = {r.category for r in result}
        assert categories == {"negativo", "positivo"}

    def test_empty_input(self) -> None:
        result = aggregate_by_category({})
        assert result == []
