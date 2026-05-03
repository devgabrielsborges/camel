from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer

from camel.infrastructure.cli.app import app
from camel.infrastructure.config.settings import Settings


@app.command("dashboard")
def dashboard(
    db_path: str = typer.Option(
        "",
        "--db-path",
        help="Path to DuckDB database file (default from DUCKDB_PATH env var)",
    ),
    port: int = typer.Option(
        8501,
        "--port",
        help="Port for the Streamlit server",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not auto-open the browser",
    ),
) -> None:
    """Launch the Streamlit evaluation dashboard."""
    resolved_db_path = db_path or Settings().duckdb_path

    if not Path(resolved_db_path).exists():
        typer.echo(
            f"Error: DuckDB file not found at '{resolved_db_path}'. "
            "Run `camel prepare && camel run` first, then `cd dbt && dbt run`.",
            err=True,
        )
        raise typer.Exit(code=1)

    streamlit_bin = shutil.which("streamlit")
    if streamlit_bin is None:
        typer.echo(
            "Error: streamlit is not installed. Run `uv add streamlit`.",
            err=True,
        )
        raise typer.Exit(code=2)

    app_path = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
    ]

    if no_browser:
        cmd.extend(["--server.headless", "true"])

    cmd.extend(["--", "--db-path", resolved_db_path])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        typer.echo(f"Streamlit exited with code {exc.returncode}", err=True)
        raise typer.Exit(code=2) from exc
