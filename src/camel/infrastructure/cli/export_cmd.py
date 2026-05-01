from __future__ import annotations

import logging
from typing import Optional

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)


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
        help="Output CSV path (overrides RESULTS_DIR env var)",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="MLflow experiment name (default: WeniEval)",
    ),
) -> None:
    """Export evaluation results to CSV."""
    from camel.infrastructure.config.settings import Settings

    settings = Settings()  # type: ignore[call-arg]
    output_path = output or f"{settings.results_dir}/predictions.csv"

    typer.echo(
        f"Export requires an Evaluation object with sessions populated "
        f"from inference + evaluation. "
        f"Run ID: {run_id}, Output: {output_path}. "
        f"Use 'camel run' for the full pipeline."
    )
    typer.echo(f"MLflow tracking: {settings.mlflow_tracking_uri}")
