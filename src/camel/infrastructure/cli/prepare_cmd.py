from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import typer

from camel.infrastructure.cli.app import app

logger = logging.getLogger(__name__)

HF_DATASET_DEFAULT = "Weni/WeniEval-Benchmark-2.0.0"
HF_SPLIT_DEFAULT = "train"
VALID_CATEGORIES = ["positivo", "negativo"]


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
    skip_sample: bool = typer.Option(
        False,
        "--skip-sample",
        help="Skip stratified sampling (use existing silver parquet)",
    ),
) -> None:
    """Download the evaluation dataset, apply stratified sampling, and run dbt transformations."""
    from camel.infrastructure.config.settings import Settings

    settings = Settings()

    raw_path = Path(settings.raw_parquet_path)
    silver_path = Path(settings.silver_parquet_path)

    if not skip_download:
        _download_dataset(raw_path)
    elif not raw_path.exists():
        typer.echo(f"Parquet file not found at {raw_path}", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"Skipping download, using existing {raw_path}")

    if not skip_sample:
        _stratified_sample(
            raw_path,
            silver_path,
            fraction=settings.sample_fraction,
            seed=settings.sample_seed,
        )
    elif not silver_path.exists():
        typer.echo(f"Silver parquet not found at {silver_path}", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"Skipping sampling, using existing {silver_path}")

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
            "The 'datasets' package is required for download. Install it with: uv add datasets",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Dataset saved to {raw_path}")


def _stratified_sample(
    raw_path: Path,
    silver_path: Path,
    *,
    fraction: float,
    seed: int,
) -> None:
    """Filter to valid categories and apply stratified sampling preserving class proportions."""
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Applying stratified sampling (fraction={fraction}, seed={seed})...")

    df = pd.read_parquet(raw_path)
    df = df[df.data_category_QA.isin(VALID_CATEGORIES)]

    total_before = len(df)
    weights = df.data_category_QA.map(1 / df.data_category_QA.value_counts())
    sample = df.sample(int(len(df) * fraction), random_state=seed, weights=weights)

    sample.to_parquet(str(silver_path), index=False)

    distribution = sample.data_category_QA.value_counts()
    typer.echo(
        f"Silver layer: {len(sample)} records sampled from {total_before} "
        f"(positivo={distribution.get('positivo', 0)}, negativo={distribution.get('negativo', 0)})"
    )


def _run_dbt(gold: bool) -> None:
    dbt_dir = Path("dbt")
    if not dbt_dir.exists():
        typer.echo("dbt/ directory not found", err=True)
        raise typer.Exit(code=1)

    dbt_bin = shutil.which("dbt")
    if dbt_bin is None:
        typer.echo("dbt executable not found. Install with: uv add dbt-core dbt-duckdb", err=True)
        raise typer.Exit(code=1)

    cmd: list[str] = [dbt_bin, "run"]
    if gold:
        cmd += ["--vars", '{"enable_gold": true}']

    typer.echo(f"Running dbt ({'bronze + silver + gold' if gold else 'bronze + silver'})...")

    result = subprocess.run(cmd, cwd=str(dbt_dir), capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo("dbt run failed:", err=True)
        typer.echo(result.stderr or result.stdout, err=True)
        raise typer.Exit(code=1)

    typer.echo("dbt transformations complete")
