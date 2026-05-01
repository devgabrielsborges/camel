from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["positivo", "negativo"]


@app.command(name="run")
def run_pipeline(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Max number of records to process (all if omitted)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Number of records per batch (overrides BATCH_SIZE env var)",
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        "-c",
        help="Max concurrent agent calls (overrides CONCURRENCY env var)",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated data_category_QA filter values",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="MLflow experiment name (default: WeniEval)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (overrides RESULTS_DIR env var)",
    ),
    no_llm_judge: bool = typer.Option(
        False,
        "--no-llm-judge",
        help="Skip LLM-as-judge scorers (deterministic only)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (openai: model name, litellm: provider/model)",
    ),
) -> None:
    """Execute the full pipeline: register prompt -> register dataset -> infer -> evaluate -> export."""
    from camel.application.use_cases.export_results import ExportResults
    from camel.application.use_cases.register_dataset import RegisterDataset
    from camel.application.use_cases.run_evaluation import RunEvaluation
    from camel.application.use_cases.run_inference import RunInference
    from camel.application.use_cases.run_pipeline import RunPipeline
    from camel.domain.entities.evaluation import Evaluation
    from camel.domain.services.prompt_renderer import PromptRenderer
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.domain.value_objects.prompt_template import PromptTemplate
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings
    from camel.infrastructure.factories.agent_factory import create_agent_adapter
    from camel.infrastructure.factories.scorer_factory import create_scorers

    settings = Settings()

    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES
    exp_name = experiment or settings.experiment_name
    bs = batch_size or settings.batch_size
    conc = concurrency or settings.concurrency
    output_path = output or f"{settings.results_dir}/predictions.csv"
    effective_model = model or settings.openai_model

    dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
    agent_adapter = create_agent_adapter(settings, model_override=model)
    tracker_adapter = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    renderer = PromptRenderer(template_path=settings.prompt_template_path)
    scorers = create_scorers(settings, no_llm_judge=no_llm_judge)

    evaluation = Evaluation(
        evaluation_id=str(uuid.uuid4()),
        experiment_name=exp_name,
        eval_model=ModelConfig(model_name=effective_model, temperature=0.0),
        prompt_version="",
        dataset_name=settings.dataset_name,
    )

    prompt_template = PromptTemplate(
        template_path=settings.prompt_template_path,
        version_uri="",
    )

    register_dataset = RegisterDataset(
        dataset_adapter=dataset_adapter,
        tracker_adapter=tracker_adapter,
    )

    run_inference = RunInference(
        dataset_adapter=dataset_adapter,
        agent_adapter=agent_adapter,
        tracker_adapter=tracker_adapter,
        prompt_renderer=renderer,
        batch_size=bs,
        concurrency=conc,
    )

    run_evaluation = RunEvaluation(
        scorers=scorers,
        tracker_adapter=tracker_adapter,
    )

    export_results = ExportResults()

    pipeline = RunPipeline(
        tracker_adapter=tracker_adapter,
        register_dataset=register_dataset,
        run_inference=run_inference,
        run_evaluation=run_evaluation,
        export_results=export_results,
    )

    try:
        result = asyncio.run(
            pipeline.execute(
                evaluation=evaluation,
                categories=cat_list,
                output_path=output_path,
                prompt_template=prompt_template,
                limit=limit,
            )
        )
    except RuntimeError as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Pipeline complete: {result.exported_rows} rows exported to {output_path}")
    typer.echo(f"MLflow run ID: {result.run_id}")
    typer.echo(f"Sessions: {len(result.evaluation.sessions)}")
    typer.echo(f"Status: {result.evaluation.status.value}")

    if result.overall_metrics:
        typer.echo("\nScorer Summary:")
        for m in result.overall_metrics:
            typer.echo(f"  {m.scorer_name}: mean={m.mean:.4f}, std={m.std:.4f}, n={m.count}")
