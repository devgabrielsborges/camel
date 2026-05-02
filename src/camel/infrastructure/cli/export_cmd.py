from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["positivo", "negativo"]


def _create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


@app.command()
def export(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="MLflow run ID to export results from",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL path (overrides RESULTS_DIR env var)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="MLflow experiment name (default: WeniEval)",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated data_category_QA filter values",
    ),
) -> None:
    """Export inference results to JSONL from a completed run."""
    from camel.application.use_cases.export_results import ExportResults
    from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
    from camel.domain.entities.session import Session
    from camel.domain.entities.trace import Trace
    from camel.domain.value_objects import TokenUsage
    from camel.domain.value_objects.dataset_record import DatasetRecord
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings

    settings = Settings()

    exp_name = experiment or settings.experiment_name
    output_path = output or f"{settings.results_dir}/predictions.jsonl"
    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES

    tracker = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)

    with _create_progress() as progress:
        steps_task = progress.add_task("Pipeline steps", total=3)

        progress.update(steps_task, description="[1/3] Retrieving traces")
        mlflow_traces = tracker.search_traces(
            experiment_name=exp_name,
            run_id=run_id,
        )

        if not mlflow_traces:
            progress.stop()
            typer.echo("No traces found for this run.", err=True)
            raise typer.Exit(code=1)
        progress.advance(steps_task)

        progress.update(steps_task, description="[2/3] Matching traces to dataset")
        dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
        records_by_id: dict[str, DatasetRecord] = {}
        for record in dataset_adapter.load_filtered(cat_list):
            records_by_id[record.id] = record

        evaluation = Evaluation(
            evaluation_id=f"export-{run_id}",
            experiment_name=exp_name,
            eval_model=ModelConfig(model_name=settings.openai_model, temperature=0.0),
            prompt_version="",
            dataset_name=settings.dataset_name,
            status=EvaluationStatus.COMPLETE,
        )

        for mlflow_trace in mlflow_traces:
            trace_id = str(mlflow_trace.get("trace_id", ""))
            request = mlflow_trace.get("request", "")
            response = mlflow_trace.get("response", "")
            tags = mlflow_trace.get("tags", {}) or {}
            group_id = str(tags.get("group_id", trace_id))

            matched_record = records_by_id.get(group_id)
            if matched_record is None:
                continue

            trace_obj = Trace(
                trace_id=trace_id,
                session_id=group_id,
                input_text=str(request),
                output_text=str(response),
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                model=settings.openai_model,
                latency_ms=0,
            )

            session = Session(
                session_id=group_id,
                evaluation_id=evaluation.evaluation_id,
                dataset_record=matched_record,
            )
            session.add_trace(trace_obj)
            evaluation.add_session(session)
        progress.advance(steps_task)

        progress.update(steps_task, description="[3/3] Exporting to JSONL")
        export_task = progress.add_task("Exporting", total=len(evaluation.sessions))

        def _on_export_progress(current: int, _total: int) -> None:
            progress.update(export_task, completed=current)

        use_case = ExportResults()
        row_count = use_case.execute(
            evaluation=evaluation, output_path=output_path, on_progress=_on_export_progress
        )
        progress.update(export_task, completed=len(evaluation.sessions))
        progress.advance(steps_task)

    typer.echo(f"\nExported {row_count} rows to {output_path}")
