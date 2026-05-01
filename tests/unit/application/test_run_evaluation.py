from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from camel.application.use_cases.run_evaluation import RunEvaluation
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.session import Session
from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.infrastructure.adapters.mlflow_scorer import DeterministicScorer


def _make_trace(session_id: str) -> Trace:
    return Trace(
        trace_id=f"trace-{session_id}",
        session_id=session_id,
        input_text="What is the return policy?",
        output_text="Returns are accepted within 30 days of purchase.",
        token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        model="gpt-4o-mini",
        latency_ms=200,
    )


def _make_evaluation_with_sessions(
    record: DatasetRecord,
) -> Evaluation:
    evaluation = Evaluation(
        evaluation_id="eval-001",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.EVALUATING,
    )
    session = Session(
        session_id=record.id,
        evaluation_id=evaluation.evaluation_id,
        dataset_record=record,
    )
    session.add_trace(_make_trace(record.id))
    evaluation.add_session(session)
    return evaluation


def test_execute_produces_scores(sample_dataset_record: DatasetRecord) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)
    mock_tracker = MagicMock()

    use_case = RunEvaluation(
        deterministic_scorer=DeterministicScorer(),
        llm_scorer=None,
        tracker_adapter=mock_tracker,
    )

    overall, by_category = use_case.execute(evaluation, run_id="run-123")

    assert len(overall) > 0
    assert evaluation.status == EvaluationStatus.COMPLETE

    scorer_names = {m.scorer_name for m in overall}
    assert "token_overlap_f1" in scorer_names
    assert "refusal_detection" in scorer_names


def test_execute_logs_metrics(sample_dataset_record: DatasetRecord) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)
    mock_tracker = MagicMock()

    use_case = RunEvaluation(
        deterministic_scorer=DeterministicScorer(),
        llm_scorer=None,
        tracker_adapter=mock_tracker,
    )

    use_case.execute(evaluation, run_id="run-123")

    mock_tracker.log_metrics.assert_called_once()
    call_args = mock_tracker.log_metrics.call_args
    metrics = call_args[0][1]
    assert "total_scored" in metrics
    assert metrics["total_scored"] == 1.0


def test_execute_category_breakdown(sample_dataset_record: DatasetRecord) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)
    mock_tracker = MagicMock()

    use_case = RunEvaluation(
        deterministic_scorer=DeterministicScorer(),
        llm_scorer=None,
        tracker_adapter=mock_tracker,
    )

    _, by_category = use_case.execute(evaluation, run_id="run-123")

    assert len(by_category) == 1
    assert by_category[0].category == "positivo"


def test_execute_transitions_to_failed_on_error(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)
    mock_tracker = MagicMock()
    mock_tracker.log_metrics.side_effect = RuntimeError("MLflow down")

    use_case = RunEvaluation(
        deterministic_scorer=DeterministicScorer(),
        llm_scorer=None,
        tracker_adapter=mock_tracker,
    )

    with pytest.raises(RuntimeError, match="MLflow down"):
        use_case.execute(evaluation, run_id="run-123")

    assert evaluation.status == EvaluationStatus.FAILED


def test_execute_adds_scores_to_traces(
    sample_dataset_record: DatasetRecord,
) -> None:
    evaluation = _make_evaluation_with_sessions(sample_dataset_record)
    mock_tracker = MagicMock()

    use_case = RunEvaluation(
        deterministic_scorer=DeterministicScorer(),
        llm_scorer=None,
        tracker_adapter=mock_tracker,
    )

    use_case.execute(evaluation, run_id="run-123")

    trace_obj = evaluation.sessions[0].traces[0]
    assert len(trace_obj.scores) > 0
