"""End-to-end pipeline smoke test on a 2-sample subset.

Validates the full infer → evaluate → export pipeline using mocked
external services (OpenAI agent, MLflow tracker). Asserts peak memory
stays under the 8 GB budget.
"""

from __future__ import annotations

import csv
import tempfile
import tracemalloc
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from camel.application.use_cases.export_results import ExportResults
from camel.application.use_cases.run_evaluation import RunEvaluation
from camel.application.use_cases.run_inference import RunInference
from camel.application.use_cases.run_pipeline import RunPipeline
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.trace import Trace
from camel.domain.services.prompt_renderer import PromptRenderer
from camel.domain.value_objects import Chunk, ClassDef, TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.model_config import ModelConfig
from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.infrastructure.adapters.mlflow_scorer import get_deterministic_scorers

EIGHT_GB_BYTES = 8 * 1024 * 1024 * 1024


def _make_record(idx: int) -> DatasetRecord:
    category = "positivo" if idx % 2 == 0 else "negativo"
    return DatasetRecord(
        id=f"e2e-session-{idx}",
        question=f"What is item {idx}?",
        content=f"Item {idx} is a test item used for validation.",
        context_metadata="{'kind': 'test'}",
        name="TestBot",
        occupation="QA Tester",
        adjective="Precise",
        chatbot_goal="Answer questions about test items.",
        instructions=["Be accurate", "Only answer based on the content"],
        chunks_big=[Chunk(content=f"Item {idx} is a test item.", score=1.0)],
        classes=[ClassDef(class_name="Items", context="test items", class_id="P1")],
        chosen_class_id="P1",
        language=1,
        data_category_qa=category,
        content_base_uuids=f"uuid-{idx}",
    )


def _make_trace(record: DatasetRecord) -> Trace:
    return Trace(
        trace_id=f"trace-{record.id}",
        session_id=record.id,
        input_text=record.question,
        output_text=f"Item {record.id} is a test item used for validation.",
        token_usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
        model="gpt-4o-mini",
        latency_ms=150,
    )


@pytest.fixture()
def sample_records() -> list[DatasetRecord]:
    return [_make_record(i) for i in range(2)]


@pytest.fixture()
def mock_dataset(sample_records: list[DatasetRecord]) -> MagicMock:
    mock = MagicMock()
    mock.load_filtered.return_value = iter(sample_records)
    return mock


@pytest.fixture()
def mock_agent(sample_records: list[DatasetRecord]) -> AsyncMock:
    mock = AsyncMock()

    async def _invoke(rec: DatasetRecord, prompt: str) -> Trace:
        return _make_trace(rec)

    mock.invoke.side_effect = _invoke
    mock.flush = AsyncMock()
    return mock


@pytest.fixture()
def mock_tracker() -> MagicMock:
    mock = MagicMock()
    mock.start_run.return_value = "e2e-run-id"
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
async def test_e2e_pipeline_smoke(
    mock_dataset: MagicMock,
    mock_agent: AsyncMock,
    mock_tracker: MagicMock,
    mock_renderer: MagicMock,
) -> None:
    """Full pipeline on 2 samples: infer -> evaluate -> export."""
    tracemalloc.start()

    try:
        run_inference = RunInference(
            dataset_adapter=mock_dataset,
            agent_adapter=mock_agent,
            tracker_adapter=mock_tracker,
            prompt_renderer=mock_renderer,
            batch_size=10,
            concurrency=5,
        )

        deterministic_scorers = get_deterministic_scorers()
        run_evaluation = RunEvaluation(
            scorers=deterministic_scorers,
            tracker_adapter=mock_tracker,
        )

        export_results = ExportResults()

        pipeline = RunPipeline(
            run_inference=run_inference,
            run_evaluation=run_evaluation,
            export_results=export_results,
        )

        evaluation = Evaluation(
            evaluation_id="e2e-eval",
            experiment_name="E2ETest",
            eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
            prompt_version="prompts:/test/1",
            dataset_name="e2e_dataset",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "predictions.csv")

            result = await pipeline.execute(
                evaluation=evaluation,
                categories=["positivo", "negativo"],
                run_id="e2e-run-id",
                output_path=output_path,
            )

            assert result.evaluation.status == EvaluationStatus.COMPLETE
            assert len(result.evaluation.sessions) == 2
            assert result.exported_rows == 2
            assert len(result.overall_metrics) > 0

            assert Path(output_path).exists()
            with open(output_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            for row in rows:
                assert row["id"]
                assert row["question"]
                assert row["prediction"]
                assert row["model"] == "gpt-4o-mini"
                assert row["token_overlap_f1"]
                assert row["class_exact_match"]
                assert row["refusal_detection"]

            categories_in_csv = {r["data_category_QA"] for r in rows}
            assert categories_in_csv == {"positivo", "negativo"}

            for session in result.evaluation.sessions:
                for trace in session.traces:
                    assert len(trace.scores) == len(deterministic_scorers)

            for metric in result.overall_metrics:
                assert metric.count == 2

    finally:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert peak_bytes < EIGHT_GB_BYTES, (
        f"Peak memory {peak_bytes / (1024**3):.2f} GB exceeds 8 GB budget"
    )


@pytest.mark.asyncio
async def test_e2e_pipeline_halt_propagation(
    mock_dataset: MagicMock,
    mock_agent: AsyncMock,
    mock_tracker: MagicMock,
    mock_renderer: MagicMock,
) -> None:
    """Pipeline halts and propagates error on inference failure."""
    mock_dataset.load_filtered.side_effect = RuntimeError("Simulated DB failure")

    run_inference = RunInference(
        dataset_adapter=mock_dataset,
        agent_adapter=mock_agent,
        tracker_adapter=mock_tracker,
        prompt_renderer=mock_renderer,
        batch_size=10,
        concurrency=5,
    )

    run_evaluation = RunEvaluation(
        scorers=get_deterministic_scorers(),
        tracker_adapter=mock_tracker,
    )

    export_results = ExportResults()

    pipeline = RunPipeline(
        run_inference=run_inference,
        run_evaluation=run_evaluation,
        export_results=export_results,
    )

    evaluation = Evaluation(
        evaluation_id="e2e-fail",
        experiment_name="E2ETest",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="",
        dataset_name="e2e_dataset",
    )

    with pytest.raises(RuntimeError, match="Simulated DB failure"):
        await pipeline.execute(
            evaluation=evaluation,
            categories=["positivo"],
            run_id="e2e-run-id",
            output_path="/tmp/should-not-exist.csv",
        )

    assert evaluation.status == EvaluationStatus.FAILED
