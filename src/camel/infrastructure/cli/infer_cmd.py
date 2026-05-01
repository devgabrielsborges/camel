from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["positivo", "negativo"]


@app.command()
def infer(
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
    register: bool = typer.Option(
        False,
        "--register-dataset",
        help="Also register the filtered dataset in MLflow",
    ),
) -> None:
    """Run batch inference on the filtered dataset."""
    from camel.application.use_cases.register_dataset import RegisterDataset
    from camel.application.use_cases.run_inference import RunInference
    from camel.domain.entities.evaluation import Evaluation
    from camel.domain.services.prompt_renderer import PromptRenderer
    from camel.domain.value_objects.model_config import ModelConfig
    from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.adapters.openai_agent import OpenAIAgentAdapter
    from camel.infrastructure.config.settings import Settings

    settings = Settings()  # type: ignore[call-arg]

    cat_list = categories.split(",") if categories else DEFAULT_CATEGORIES
    exp_name = experiment or "WeniEval"
    bs = batch_size or settings.batch_size
    conc = concurrency or settings.concurrency

    dataset_adapter = DuckDBDatasetAdapter(db_path=settings.duckdb_path)
    agent_adapter = OpenAIAgentAdapter(model=settings.openai_model)
    tracker_adapter = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    renderer = PromptRenderer(template_path=settings.prompt_template_path)

    evaluation = Evaluation(
        evaluation_id=str(uuid.uuid4()),
        experiment_name=exp_name,
        eval_model=ModelConfig(model_name=settings.openai_model, temperature=0.0),
        prompt_version="",
        dataset_name="weni_eval_dataset",
    )

    prompt_version_uri = ""
    try:
        from camel.domain.value_objects.prompt_template import PromptTemplate

        tpl = PromptTemplate(
            template_path=settings.prompt_template_path,
            version_uri="",
        )
        prompt_version_uri = tracker_adapter.register_prompt(tpl)
        logger.info("Registered prompt: %s", prompt_version_uri)
    except Exception:
        logger.warning("Could not register prompt with MLflow, continuing without")

    if register:
        reg_uc = RegisterDataset(
            dataset_adapter=dataset_adapter,
            tracker_adapter=tracker_adapter,
        )
        count = reg_uc.execute(
            dataset_name=evaluation.dataset_name,
            categories=cat_list,
            limit=limit,
        )
        typer.echo(f"Registered {count} records as MLflow dataset")

    use_case = RunInference(
        dataset_adapter=dataset_adapter,
        agent_adapter=agent_adapter,
        tracker_adapter=tracker_adapter,
        prompt_renderer=renderer,
        batch_size=bs,
        concurrency=conc,
    )

    result = asyncio.run(
        use_case.execute(
            evaluation=evaluation,
            categories=cat_list,
            limit=limit,
            prompt_version_uri=prompt_version_uri,
        )
    )

    typer.echo(
        f"Inference complete: {len(result.sessions)} sessions, " f"status={result.status.value}"
    )
