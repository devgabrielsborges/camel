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
    from camel.infrastructure.adapters.mlflow_scorer import (
        DeterministicScorer,
        LLMJudgeScorer,
    )
    from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter
    from camel.infrastructure.config.settings import Settings

    settings = Settings()  # type: ignore[call-arg]

    tracker = MLflowTrackerAdapter(tracking_uri=settings.mlflow_tracking_uri)

    DeterministicScorer()
    llm_scorer: LLMJudgeScorer | None = None
    if not no_llm_judge:
        llm_scorer = LLMJudgeScorer(judge_model=settings.judge_model)

    typer.echo(
        f"This command scores traces from run {run_id}. "
        "In the current implementation, evaluation requires an Evaluation "
        "object with sessions populated from inference. "
        "Use 'camel run' for the full pipeline."
    )
    typer.echo(f"Deterministic scorers: token_overlap_f1, class_exact_match, refusal_detection")
    if llm_scorer:
        typer.echo(f"LLM judge: {settings.judge_model} (Correctness + Guidelines)")
    else:
        typer.echo("LLM judge: disabled")

    typer.echo(f"MLflow tracking: {settings.mlflow_tracking_uri}")
