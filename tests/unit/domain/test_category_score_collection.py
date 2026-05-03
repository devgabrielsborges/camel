from __future__ import annotations

import pytest

from camel.domain.value_objects.category_score_collection import CategoryScoreCollection


class TestCategoryScoreCollectionConstruction:
    def test_basic_construction(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=("s1", "s2", "s3"),
            scores={"token_overlap_f1": (0.5, 0.6, 0.7)},
        )
        assert collection.category == "positivo"
        assert len(collection.session_ids) == 3
        assert len(collection.scores["token_overlap_f1"]) == 3

    def test_score_tuple_length_matches_session_ids(self) -> None:
        collection = CategoryScoreCollection(
            category="negativo",
            session_ids=("s1", "s2"),
            scores={
                "token_overlap_f1": (0.3, 0.4),
                "refusal_detection": (1.0, 0.0),
            },
        )
        n = len(collection.session_ids)
        for metric_name, values in collection.scores.items():
            assert len(values) == n, (
                f"Score tuple for '{metric_name}' has length {len(values)}, "
                f"expected {n} to match session_ids"
            )

    def test_multiple_metrics_same_length(self) -> None:
        ids = ("s1", "s2", "s3", "s4")
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=ids,
            scores={
                "token_overlap_f1": (0.5, 0.6, 0.7, 0.8),
                "refusal_detection": (0.0, 0.0, 1.0, 0.0),
                "groundedness": (0.9, 0.8, 0.7, 0.6),
            },
        )
        for values in collection.scores.values():
            assert len(values) == len(ids)

    def test_empty_scores(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=(),
            scores={},
        )
        assert len(collection.session_ids) == 0
        assert len(collection.scores) == 0

    def test_single_session(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=("s1",),
            scores={"token_overlap_f1": (0.5,)},
        )
        assert len(collection.session_ids) == 1
        assert collection.scores["token_overlap_f1"] == (0.5,)


class TestCategoryScoreCollectionImmutability:
    def test_frozen_dataclass(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=("s1",),
            scores={"token_overlap_f1": (0.5,)},
        )
        with pytest.raises(AttributeError):
            collection.category = "negativo"  # type: ignore[misc]

    def test_session_ids_is_tuple(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=("s1", "s2"),
            scores={"token_overlap_f1": (0.5, 0.6)},
        )
        assert isinstance(collection.session_ids, tuple)

    def test_scores_values_are_tuples(self) -> None:
        collection = CategoryScoreCollection(
            category="positivo",
            session_ids=("s1", "s2"),
            scores={"token_overlap_f1": (0.5, 0.6)},
        )
        for values in collection.scores.values():
            assert isinstance(values, tuple)
