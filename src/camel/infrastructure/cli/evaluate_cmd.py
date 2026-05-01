from __future__ import annotations

import logging
from typing import Optional

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["positivo", "negativo"]


@app.command()
def evaluate(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="MLflow run ID from a previous inference run",
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
    no_llm_judge: bool = typer.Option(
        False,
        "--no-llm-judge",
        help="Skip LLM-as-judge scorers (deterministic only)",
    ),
) -> None:
    """Score cached traces from a previous inference run."""
    from camel.application.use_cases.run_evaluation import RunEvaluation
    from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
    from camel.domain.entities.session import Session
    from camel.domain.entities.trace import Trace
    from camel.domain.value_objects import TokenUsage
    from camel.domain.value_objects.dataset_record import DatasetRecord
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings
    from camel.infrastructure.factories.scorer_factory import create_scorers

    settings = Settings()  # type: ignore[call-arg]

    exp_name = experiment or settings.experiment_name
    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES

    tracker = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    scorers = create_scorers(settings, no_llm_judge=no_llm_judge)

    typer.echo(f"Retrieving traces from run {run_id}...")
    mlflow_traces = tracker.search_traces(
        experiment_name=exp_name,
        run_id=run_id,
    )

    if not mlflow_traces:
        typer.echo("No traces found for this run. Ensure inference was completed.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(mlflow_traces)} traces")

    dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
    records_by_id: dict[str, DatasetRecord] = {}
    for record in dataset_adapter.load_filtered(cat_list):
        records_by_id[record.id] = record

    evaluation = Evaluation(
        evaluation_id=f"eval-{run_id}",
        experiment_name=exp_name,
        eval_model=ModelConfig(model_name=settings.openai_model, temperature=0.0),
        prompt_version="",
        dataset_name=settings.dataset_name,
        status=EvaluationStatus.EVALUATING,
    )

    for mlflow_trace in mlflow_traces:
        trace_id = str(mlflow_trace.get("trace_id", ""))
        request = mlflow_trace.get("request", "")
        response = mlflow_trace.get("response", "")
        tags = mlflow_trace.get("tags", {}) or {}
        group_id = str(tags.get("group_id", trace_id))

        matched_record = records_by_id.get(group_id)
        if matched_record is None:
            logger.warning("No dataset record found for group_id=%s, skipping", group_id)
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

    if not evaluation.sessions:
        typer.echo(
            "Could not match any traces to dataset records. "
            "Ensure the dataset is loaded and traces have group_id tags.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Matched {len(evaluation.sessions)} sessions for evaluation")

    use_case = RunEvaluation(
        scorers=scorers,
        tracker_adapter=tracker,
    )

    overall, by_category = use_case.execute(evaluation, run_id=run_id)

    typer.echo(f"Evaluation complete: {len(evaluation.sessions)} sessions scored")
    typer.echo(f"Status: {evaluation.status.value}")

    if overall:
        typer.echo("\nScorer Summary:")
        for m in overall:
            typer.echo(f"  {m.scorer_name}: mean={m.mean:.4f}, std={m.std:.4f}, n={m.count}")

    if by_category:
        typer.echo("\nCategory Breakdown:")
        for cb in by_category:
            typer.echo(f"  {cb.category}:")
            for m in cb.metrics:
                typer.echo(f"    {m.scorer_name}: mean={m.mean:.4f}, n={m.count}")
