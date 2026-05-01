from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from camel.application.use_cases.run_pipeline import RunPipeline
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.session import Session
from camel.domain.entities.trace import Trace
from camel.domain.services.aggregation import AggregatedMetric
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.domain.value_objects.score import Score


def _make_trace(session_id: str) -> Trace:
    trace = Trace(
        trace_id=f"trace-{session_id}",
        session_id=session_id,
        input_text="What is the return policy?",
        output_text="Returns are accepted within 30 days.",
        token_usage=TokenUsage(input_tokens=10, output_tokens=15, total_tokens=25),
        model="gpt-4o-mini",
        latency_ms=150,
    )
    trace.add_score(Score(scorer_name="token_overlap_f1", value=0.8))
    return trace


def _make_evaluation_after_inference(
    record: DatasetRecord,
) -> Evaluation:
    evaluation = Evaluation(
        evaluation_id="eval-pipeline",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
        status=EvaluationStatus.INFERRING,
    )
    session = Session(
        session_id=record.id,
        evaluation_id=evaluation.evaluation_id,
        dataset_record=record,
    )
    session.add_trace(_make_trace(record.id))
    evaluation.add_session(session)
    return evaluation


@pytest.fixture()
def mock_tracker() -> MagicMock:
    mock = MagicMock()
    mock.start_run.return_value = "run-123"
    mock.register_prompt.return_value = "prompts:/test/1"
    return mock


@pytest.fixture()
def mock_register_dataset() -> MagicMock:
    mock = MagicMock()
    mock.execute.return_value = 10
    return mock


@pytest.fixture()
def mock_run_inference() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_run_evaluation() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_export_results() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_pipeline_executes_all_phases(
    mock_tracker: MagicMock,
    mock_register_dataset: MagicMock,
    mock_run_inference: AsyncMock,
    mock_run_evaluation: MagicMock,
    mock_export_results: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    eval_after_infer = _make_evaluation_after_inference(sample_dataset_record)
    mock_run_inference.execute.return_value = eval_after_infer

    aggregated = [AggregatedMetric(scorer_name="token_overlap_f1", mean=0.8, std=0.0, count=1)]
    mock_run_evaluation.execute.return_value = (aggregated, [])
    mock_export_results.execute.return_value = 1

    pipeline = RunPipeline(
        tracker_adapter=mock_tracker,
        register_dataset=mock_register_dataset,
        run_inference=mock_run_inference,
        run_evaluation=mock_run_evaluation,
        export_results=mock_export_results,
    )

    prompt_tpl = PromptTemplate(
        template_path="prompts/system_prompt.j2",
        version_uri="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        result = await pipeline.execute(
            evaluation=Evaluation(
                evaluation_id="eval-pipeline",
                experiment_name="test_experiment",
                eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
                prompt_version="",
                dataset_name="test_dataset",
            ),
            categories=["positivo"],
            output_path=output_path,
            prompt_template=prompt_tpl,
        )

    mock_tracker.register_prompt.assert_called_once()
    mock_register_dataset.execute.assert_called_once()
    mock_tracker.start_run.assert_called_once()
    mock_tracker.enable_autolog.assert_called_once()
    mock_run_inference.execute.assert_awaited_once()
    mock_run_evaluation.execute.assert_called_once()
    mock_export_results.execute.assert_called_once()
    mock_tracker.end_run.assert_called_once()
    mock_tracker.disable_autolog.assert_called_once()
    assert result.run_id == "run-123"
    assert result.evaluation.status == EvaluationStatus.EVALUATING


@pytest.mark.asyncio
async def test_pipeline_halts_on_inference_failure(
    mock_tracker: MagicMock,
    mock_register_dataset: MagicMock,
    mock_run_inference: AsyncMock,
    mock_run_evaluation: MagicMock,
    mock_export_results: MagicMock,
) -> None:
    mock_run_inference.execute.side_effect = RuntimeError("Inference failed")

    pipeline = RunPipeline(
        tracker_adapter=mock_tracker,
        register_dataset=mock_register_dataset,
        run_inference=mock_run_inference,
        run_evaluation=mock_run_evaluation,
        export_results=mock_export_results,
    )

    with pytest.raises(RuntimeError, match="Inference failed"):
        await pipeline.execute(
            evaluation=Evaluation(
                evaluation_id="eval-pipeline",
                experiment_name="test_experiment",
                eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
                prompt_version="",
                dataset_name="test_dataset",
            ),
            categories=["positivo"],
            output_path="/tmp/test.csv",
        )

    mock_run_evaluation.execute.assert_not_called()
    mock_export_results.execute.assert_not_called()
    mock_tracker.end_run.assert_called_once()
    mock_tracker.disable_autolog.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_halts_on_evaluation_failure(
    mock_tracker: MagicMock,
    mock_register_dataset: MagicMock,
    mock_run_inference: AsyncMock,
    mock_run_evaluation: MagicMock,
    mock_export_results: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    eval_after_infer = _make_evaluation_after_inference(sample_dataset_record)
    mock_run_inference.execute.return_value = eval_after_infer
    mock_run_evaluation.execute.side_effect = RuntimeError("Evaluation failed")

    pipeline = RunPipeline(
        tracker_adapter=mock_tracker,
        register_dataset=mock_register_dataset,
        run_inference=mock_run_inference,
        run_evaluation=mock_run_evaluation,
        export_results=mock_export_results,
    )

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        await pipeline.execute(
            evaluation=Evaluation(
                evaluation_id="eval-pipeline",
                experiment_name="test_experiment",
                eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
                prompt_version="",
                dataset_name="test_dataset",
            ),
            categories=["positivo"],
            output_path="/tmp/test.csv",
        )

    mock_export_results.execute.assert_not_called()
    mock_tracker.end_run.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_halts_on_export_failure(
    mock_tracker: MagicMock,
    mock_register_dataset: MagicMock,
    mock_run_inference: AsyncMock,
    mock_run_evaluation: MagicMock,
    mock_export_results: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    eval_after_infer = _make_evaluation_after_inference(sample_dataset_record)
    mock_run_inference.execute.return_value = eval_after_infer

    aggregated = [AggregatedMetric(scorer_name="token_overlap_f1", mean=0.8, std=0.0, count=1)]
    mock_run_evaluation.execute.return_value = (aggregated, [])
    mock_export_results.execute.side_effect = OSError("Disk full")

    pipeline = RunPipeline(
        tracker_adapter=mock_tracker,
        register_dataset=mock_register_dataset,
        run_inference=mock_run_inference,
        run_evaluation=mock_run_evaluation,
        export_results=mock_export_results,
    )

    with pytest.raises(OSError, match="Disk full"):
        await pipeline.execute(
            evaluation=Evaluation(
                evaluation_id="eval-pipeline",
                experiment_name="test_experiment",
                eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
                prompt_version="",
                dataset_name="test_dataset",
            ),
            categories=["positivo"],
            output_path="/tmp/test.csv",
        )


@pytest.mark.asyncio
async def test_pipeline_passes_limit_to_inference(
    mock_tracker: MagicMock,
    mock_register_dataset: MagicMock,
    mock_run_inference: AsyncMock,
    mock_run_evaluation: MagicMock,
    mock_export_results: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    eval_after_infer = _make_evaluation_after_inference(sample_dataset_record)
    mock_run_inference.execute.return_value = eval_after_infer

    aggregated = [AggregatedMetric(scorer_name="token_overlap_f1", mean=0.8, std=0.0, count=1)]
    mock_run_evaluation.execute.return_value = (aggregated, [])
    mock_export_results.execute.return_value = 1

    pipeline = RunPipeline(
        tracker_adapter=mock_tracker,
        register_dataset=mock_register_dataset,
        run_inference=mock_run_inference,
        run_evaluation=mock_run_evaluation,
        export_results=mock_export_results,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "predictions.csv")

        await pipeline.execute(
            evaluation=Evaluation(
                evaluation_id="eval-pipeline",
                experiment_name="test_experiment",
                eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
                prompt_version="",
                dataset_name="test_dataset",
            ),
            categories=["positivo"],
            output_path=output_path,
            limit=5,
        )

    call_kwargs = mock_run_inference.execute.call_args
    assert call_kwargs.kwargs.get("limit") == 5 or call_kwargs[1].get("limit") == 5
