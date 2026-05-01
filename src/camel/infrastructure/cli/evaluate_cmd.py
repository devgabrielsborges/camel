from __future__ import annotations

import logging
from typing import Optional

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)


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
    no_llm_judge: bool = typer.Option(
        False,
        "--no-llm-judge",
        help="Skip LLM-as-judge scorers (deterministic only)",
    ),
) -> None:
    """Score cached traces from a previous inference run."""
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings
    from camel.infrastructure.factories.scorer_factory import create_scorers

    settings = Settings()  # type: ignore[call-arg]

    tracker = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)
    scorers = create_scorers(settings, no_llm_judge=no_llm_judge)

    scorer_names = [getattr(s, "name", "unknown") for s in scorers]

    typer.echo(
        f"This command scores traces from run {run_id}. "
        "In the current implementation, evaluation requires an Evaluation "
        "object with sessions populated from inference. "
        "Use 'camel run' for the full pipeline."
    )
    typer.echo(f"Scorers: {', '.join(scorer_names)}")
    typer.echo(f"MLflow tracking: {settings.mlflow_tracking_uri}")
