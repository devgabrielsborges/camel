from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from camel.application.use_cases.run_inference import RunInference
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.domain.value_objects.prompt_template import PromptTemplate


def _make_trace(session_id: str) -> Trace:
    return Trace(
        trace_id=f"trace-{session_id}",
        session_id=session_id,
        input_text="test question",
        output_text="test answer",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        model="gpt-4o-mini",
        latency_ms=100,
    )


def _make_evaluation() -> Evaluation:
    return Evaluation(
        evaluation_id="eval-001",
        experiment_name="test_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
    )


@pytest.fixture()
def mock_dataset() -> MagicMock:
    mock = MagicMock()
    return mock


@pytest.fixture()
def mock_agent() -> AsyncMock:
    mock = AsyncMock()
    mock.flush = AsyncMock()
    return mock


@pytest.fixture()
def mock_tracker() -> MagicMock:
    mock = MagicMock()
    mock.start_run.return_value = "run-123"
    return mock


@pytest.fixture()
def mock_renderer() -> MagicMock:
    mock = MagicMock()
    mock.render.return_value = PromptTemplate(
        template_path="prompts/system_prompt.j2",
        version_uri="prompts:/test/1",
        rendered_content="You are a helpful assistant.",
        token_count=5,
    )
    return mock


@pytest.mark.asyncio
async def test_execute_processes_records(
    mock_dataset: MagicMock,
    mock_agent: AsyncMock,
    mock_tracker: MagicMock,
    mock_renderer: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    mock_dataset.load_filtered.return_value = iter([sample_dataset_record])
    mock_agent.invoke.return_value = _make_trace(sample_dataset_record.id)

    use_case = RunInference(
        dataset_adapter=mock_dataset,
        agent_adapter=mock_agent,
        tracker_adapter=mock_tracker,
        prompt_renderer=mock_renderer,
        batch_size=10,
        concurrency=5,
    )

    evaluation = _make_evaluation()
    result = await use_case.execute(
        evaluation=evaluation,
        categories=["positivo"],
    )

    assert result.status == EvaluationStatus.EVALUATING
    assert len(result.sessions) == 1
    assert result.sessions[0].session_id == sample_dataset_record.id


@pytest.mark.asyncio
async def test_execute_respects_limit(
    mock_dataset: MagicMock,
    mock_agent: AsyncMock,
    mock_tracker: MagicMock,
    mock_renderer: MagicMock,
    sample_dataset_record: DatasetRecord,
) -> None:
    records = [sample_dataset_record.model_copy(update={"id": f"rec-{i}"}) for i in range(5)]
    mock_dataset.load_filtered.return_value = iter(records)

    async def side_effect(rec: DatasetRecord, prompt: str) -> Trace:
        return _make_trace(rec.id)

    mock_agent.invoke.side_effect = side_effect

    use_case = RunInference(
        dataset_adapter=mock_dataset,
        agent_adapter=mock_agent,
        tracker_adapter=mock_tracker,
        prompt_renderer=mock_renderer,
        batch_size=10,
        concurrency=5,
    )

    evaluation = _make_evaluation()
    result = await use_case.execute(
        evaluation=evaluation,
        categories=["positivo"],
        limit=2,
    )

    assert len(result.sessions) == 2


@pytest.mark.asyncio
async def test_execute_transitions_to_failed_on_exception(
    mock_dataset: MagicMock,
    mock_agent: AsyncMock,
    mock_tracker: MagicMock,
    mock_renderer: MagicMock,
) -> None:
    mock_dataset.load_filtered.side_effect = RuntimeError("DB connection failed")

    use_case = RunInference(
        dataset_adapter=mock_dataset,
        agent_adapter=mock_agent,
        tracker_adapter=mock_tracker,
        prompt_renderer=mock_renderer,
        batch_size=10,
        concurrency=5,
    )

    evaluation = _make_evaluation()
    with pytest.raises(RuntimeError, match="DB connection failed"):
        await use_case.execute(evaluation=evaluation, categories=["positivo"])

    assert evaluation.status == EvaluationStatus.FAILED
