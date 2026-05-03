from __future__ import annotations

import logging
from typing import Optional

import duckdb
import typer
from rich.console import Console
from rich.table import Table

from camel.domain.services.threshold_derivation import derive_thresholds
from camel.domain.value_objects.threshold_profile import ThresholdProfile
from camel.infrastructure.adapters.duckdb_reference_scores import load_reference_scores
from camel.infrastructure.adapters.threshold_repository import ThresholdProfileRepository
from camel.infrastructure.cli.app import app
from camel.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)
console = Console()


@app.command("derive-thresholds")
def derive_thresholds_cmd(
    models: list[str] = typer.Option(
        ...,
        "--models",
        "-m",
        help="Reference model names to derive thresholds from",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for ThresholdProfile JSON (overrides THRESHOLD_PROFILE_PATH)",
    ),
    bootstrap_b: int = typer.Option(
        10000,
        "--bootstrap-b",
        help="Number of bootstrap resamples",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible bootstrap",
    ),
    percentile: int = typer.Option(
        5,
        "--percentile",
        help="Percentile for threshold derivation (e.g. 5 = 5th percentile)",
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        help="Significance level for confidence intervals",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to DuckDB database (overrides DUCKDB_PATH)",
    ),
) -> None:
    """Derive empirical thresholds from known-good reference model runs."""
    settings = Settings()
    resolved_db = db_path or settings.duckdb_path
    resolved_output = output or settings.threshold_profile_path

    console.print(f"\n[bold]Deriving thresholds from models:[/bold] {', '.join(models)}")
    console.print(f"  Database: {resolved_db}")
    console.print(f"  Bootstrap B={bootstrap_b}, seed={seed}, percentile={percentile}")
    console.print(f"  Alpha={alpha}")
    console.print()

    collections = load_reference_scores(resolved_db, models)
    if not collections:
        console.print("[red]No reference scores found for the given models.[/red]")
        raise typer.Exit(code=1)

    total_sessions = sum(len(c.session_ids) for c in collections)
    console.print(f"Loaded {total_sessions} sessions across {len(collections)} categories")

    run_ids: list[str] = []
    conn = duckdb.connect(resolved_db, read_only=True)
    try:
        placeholders = ", ".join(["?" for _ in models])
        rows = conn.execute(
            f"SELECT DISTINCT run_id FROM fct_inference_results WHERE model IN ({placeholders})",
            models,
        ).fetchall()
        run_ids = [str(r[0]) for r in rows]
    finally:
        conn.close()

    profile = derive_thresholds(
        collections,
        bootstrap_b=bootstrap_b,
        seed=seed,
        percentile=percentile,
        alpha=alpha,
        benchmark=settings.experiment_name,
        reference_models=tuple(models),
        reference_run_ids=tuple(run_ids),
    )

    repo = ThresholdProfileRepository()
    repo.save(profile, resolved_output)

    console.print(f"\n[green]ThresholdProfile saved to {resolved_output}[/green]")
    _print_summary(profile)


def _print_summary(profile: ThresholdProfile) -> None:
    table = Table(title="Category Thresholds", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Metric")
    table.add_column("Threshold", justify="right")
    table.add_column("CI Lower", justify="right")
    table.add_column("CI Upper", justify="right")
    table.add_column("Ref Mean", justify="right")
    table.add_column("Ref Std", justify="right")
    table.add_column("N", justify="right")

    for category, thresholds in profile.category_thresholds.items():
        for t in thresholds:
            table.add_row(
                category,
                t.metric_name,
                f"{t.value:.4f}",
                f"{t.ci_lower:.4f}",
                f"{t.ci_upper:.4f}",
                f"{t.reference_mean:.4f}",
                f"{t.reference_std:.4f}",
                str(t.sample_count),
            )

    console.print(table)

    if profile.global_thresholds:
        g_table = Table(title="Global Thresholds", show_lines=True)
        g_table.add_column("Metric")
        g_table.add_column("Threshold", justify="right")
        g_table.add_column("CI Lower", justify="right")
        g_table.add_column("CI Upper", justify="right")
        g_table.add_column("Ref Mean", justify="right")
        g_table.add_column("N", justify="right")

        for t in profile.global_thresholds:
            g_table.add_row(
                t.metric_name,
                f"{t.value:.4f}",
                f"{t.ci_lower:.4f}",
                f"{t.ci_upper:.4f}",
                f"{t.reference_mean:.4f}",
                str(t.sample_count),
            )

        console.print(g_table)
