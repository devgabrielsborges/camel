from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

HF_DATASET_DEFAULT = "Weni/WeniEval-Benchmark-2.0.0"
HF_SPLIT_DEFAULT = "train"


@app.command()
def prepare(
    gold: bool = typer.Option(
        False,
        "--gold",
        help="Also build gold dbt models (requires a completed pipeline run)",
    ),
    skip_download: bool = typer.Option(
        False,
        "--skip-download",
        help="Skip dataset download (use existing parquet file)",
    ),
) -> None:
    """Download the evaluation dataset and run dbt transformations."""
    from camel.infrastructure.config.settings import Settings

    settings = Settings()  # type: ignore[call-arg]

    raw_path = Path(settings.raw_parquet_path)

    if not skip_download:
        _download_dataset(raw_path)
    elif not raw_path.exists():
        typer.echo(f"Parquet file not found at {raw_path}", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"Skipping download, using existing {raw_path}")

    _run_dbt(gold=gold)

    typer.echo("Data preparation complete")


def _download_dataset(raw_path: Path) -> None:
    dataset_repo = os.environ.get("HF_DATASET_REPO", HF_DATASET_DEFAULT)
    dataset_split = os.environ.get("HF_DATASET_SPLIT", HF_SPLIT_DEFAULT)

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading {dataset_repo} (split={dataset_split})...")

    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_repo, split=dataset_split)
        ds.to_parquet(str(raw_path))
    except ImportError:
        typer.echo(
            "The 'datasets' package is required for download. " "Install it with: uv add datasets",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Dataset saved to {raw_path}")


def _run_dbt(gold: bool) -> None:
    dbt_dir = Path("dbt")
    if not dbt_dir.exists():
        typer.echo("dbt/ directory not found", err=True)
        raise typer.Exit(code=1)

    cmd: list[str] = [sys.executable, "-m", "dbt", "run"]
    if gold:
        cmd += ["--vars", '{"enable_gold": true}']

    typer.echo(f"Running dbt ({'bronze + silver + gold' if gold else 'bronze + silver'})...")

    result = subprocess.run(cmd, cwd=str(dbt_dir), capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo("dbt run failed:", err=True)
        typer.echo(result.stderr or result.stdout, err=True)
        raise typer.Exit(code=1)

    typer.echo("dbt transformations complete")
