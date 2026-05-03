from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
    trace.add_score(Score(scorer_name="correctness", value=0.9, rationale="Good answer"))
    trace.add_score(Score(scorer_name="guidelines", value=0.95, rationale="Follows guidelines"))
    trace.add_score(Score(scorer_name="hedging_detection", value=False))
    trace.add_score(Score(scorer_name="question_response_overlap", value=0.42))
    trace.add_score(Score(scorer_name="response_length_ratio", value=1.5))
    trace.add_score(Score(scorer_name="rouge_l", value=0.33))
    trace.add_score(Score(scorer_name="chunk_attribution", value=0.78))
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


def _read_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_export_writes_jsonl_file(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 1
        assert Path(output_path).exists()


def test_export_jsonl_contains_expected_fields(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        rows = _read_jsonl(output_path)

    assert len(rows) == 1
    expected_fields = {
        "id",
        "run_id",
        "timestamp",
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
        "hedging_detection",
        "question_response_overlap",
        "response_length_ratio",
        "rouge_l",
        "chunk_attribution",
        "self_consistency",
        "self_consistency_variance",
    }
    assert expected_fields.issubset(set(rows[0].keys()))


def test_export_jsonl_values_are_correct(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        rows = _read_jsonl(output_path)
        row = rows[0]

    assert row["id"] == sample_dataset_record.id
    assert row["question"] == sample_dataset_record.question
    assert row["prediction"] == "Returns are accepted within 30 days of purchase."
    assert row["data_category_QA"] == "positivo"
    assert row["model"] == "gpt-4o-mini"
    assert row["token_overlap_f1"] == pytest.approx(0.85)


def test_export_appends_to_existing_file(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path, run_id="run-1")
        use_case.execute(evaluation=evaluation, output_path=output_path, run_id="run-2")

        rows = _read_jsonl(output_path)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-2"


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
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 3

        rows = _read_jsonl(output_path)

    assert len(rows) == 3
    ids = {r["id"] for r in rows}
    assert ids == {"session-0", "session-1", "session-2"}


def test_export_creates_parent_directories(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "nested" / "dir" / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        assert Path(output_path).exists()


def test_export_jsonl_contains_xai_metric_values(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        rows = _read_jsonl(output_path)
        row = rows[0]

    assert row["hedging_detection"] is False
    assert row["question_response_overlap"] == pytest.approx(0.42)
    assert row["response_length_ratio"] == pytest.approx(1.5)
    assert row["rouge_l"] == pytest.approx(0.33)
    assert row["chunk_attribution"] == pytest.approx(0.78)
    assert row["self_consistency"] is None
    assert row["self_consistency_variance"] is None


def test_export_jsonl_self_consistency_session_level(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = Evaluation(
        evaluation_id="eval-sc",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.COMPLETE,
    )
    session = Session(
        session_id=sample_dataset_record.id,
        evaluation_id=evaluation.evaluation_id,
        dataset_record=sample_dataset_record,
    )
    trace = _make_scored_trace(sample_dataset_record.id)
    trace.add_score(
        Score(
            scorer_name="self_consistency",
            value=0.92,
            metadata={"variance": 0.004},
        )
    )
    session.add_trace(trace)
    evaluation.add_session(session)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        use_case.execute(evaluation=evaluation, output_path=output_path)

        rows = _read_jsonl(output_path)
        row = rows[0]

    assert row["self_consistency"] == pytest.approx(0.92)
    assert row["self_consistency_variance"] == pytest.approx(0.004)


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
        output_path = str(Path(tmpdir) / "predictions.jsonl")

        use_case = ExportResults()
        row_count = use_case.execute(evaluation=evaluation, output_path=output_path)

        assert row_count == 0
