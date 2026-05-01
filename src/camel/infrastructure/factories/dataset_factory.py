from __future__ import annotations

from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
from camel.infrastructure.config.settings import Settings


def create_dataset_adapter(settings: Settings) -> DuckDBDatasetAdapter:
    return DuckDBDatasetAdapter(db_path=settings.duckdb_path)
