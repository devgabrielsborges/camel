from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from camel.application.use_cases.export_results import ExportResults
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.session import Session
from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.domain.value_objects.score import Score


def _make_scored_trace(session_id: str) -> Trace:
    trace = Trace(
        trace_id=f"trace-{session_id}",
        session_id=session_id,
        input_text="What is the return policy?",
        output_text="Returns are accepted within 30 days of purchase.",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        model="gpt-4o-mini",
        latency_ms=200,
    )
    trace.add_score(Score(scorer_name="token_overlap_f1", value=0.85))
    trace.add_score(Score(scorer_name="class_exact_match", value=True))
    trace.add_score(Score(scorer_name="refusal_detection", value=False))
    trace.add_score(
        Score(scorer_name="correctness", value=0.9, rationale="Good answer")
    )
    trace.add_score(
        Score(scorer_name="guidelines", value=0.95, rationale="Follows guidelines")
    )
    return trace


def _make_evaluation_with_sessions(
    record: DatasetRecord,
) -> Evaluation:
    evaluation = Evaluation(
        evaluation_id="eval-001",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.COMPLETE,
    )
    session = Session(
        session_id=record.id,
        evaluation_id=evaluation.evaluation_id,
        dataset_record=record,
    )
    session.add_trace(_make_scored_trace(record.id))
    evaluation.add_session(session)
    return evaluation


def test_export_writes_csv_file(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 1
        assert Path(output_path).exists()


def test_export_csv_contains_expected_columns(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        expected_columns = {
            "id",
            "question",
            "prediction",
            "data_category_QA",
            "language",
            "model",
            "token_overlap_f1",
            "class_exact_match",
            "refusal_detection",
            "correctness_score",
            "guidelines_score",
        }
        assert expected_columns.issubset(set(rows[0].keys()))


def test_export_csv_values_are_correct(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["id"] == sample_dataset_record.id
        assert row["question"] == sample_dataset_record.question
        assert row["prediction"] == "Returns are accepted within 30 days of purchase."
        assert row["data_category_QA"] == "positivo"
        assert row["model"] == "gpt-4o-mini"
        assert float(row["token_overlap_f1"]) == pytest.approx(0.85)


def test_export_multiple_sessions(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = Evaluation(
        evaluation_id="eval-002",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.COMPLETE,
    )

    for i in range(3):
        record = sample_dataset_record.model_copy(update={"id": f"session-{i}"})
        session = Session(
            session_id=record.id,
            evaluation_id=evaluation.evaluation_id,
            dataset_record=record,
        )
        session.add_trace(_make_scored_trace(record.id))
        evaluation.add_session(session)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 3

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        ids = {r["id"] for r in rows}
        assert ids == {"session-0", "session-1", "session-2"}


def test_export_creates_parent_directories(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "nested" / "dir" / "predictions.csv")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        assert Path(output_path).exists()


def test_export_returns_zero_for_no_sessions() -> None:
    evaluation = Evaluation(
        evaluation_id="eval-empty",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.COMPLETE,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 0
        assert Path(output_path).exists()
