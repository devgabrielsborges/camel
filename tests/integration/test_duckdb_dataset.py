from __future__ import annotations

import pytest

from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
from camel.infrastructure.config.settings import Settings

DB_PATH = Settings(openai_api_key="test").duckdb_path


@pytest.fixture()
def adapter() -> DuckDBDatasetAdapter:
    return DuckDBDatasetAdapter(db_path=DB_PATH)


def test_count_returns_positive(adapter: DuckDBDatasetAdapter) -> None:
    total = adapter.count(["positivo", "negativo"])
    assert total > 0


def test_load_filtered_yields_records(adapter: DuckDBDatasetAdapter) -> None:
    records = list(adapter.load_filtered(["positivo"]))
    assert len(records) > 0


def test_record_has_expected_fields(adapter: DuckDBDatasetAdapter) -> None:
    records = list(adapter.load_filtered(["positivo"]))
    record = records[0]

    assert record.id
    assert record.question
    assert record.data_category_qa == "positivo"
    assert isinstance(record.language, int)


def test_record_chunks_parsed(adapter: DuckDBDatasetAdapter) -> None:
    records = list(adapter.load_filtered(["positivo"]))
    records_with_chunks = [r for r in records if r.chunks_big]
    assert len(records_with_chunks) > 0

    sample = records_with_chunks[0]
    for chunk in sample.chunks_big:
        assert chunk.content
        assert isinstance(chunk.score, float)


def test_record_classes_parsed(adapter: DuckDBDatasetAdapter) -> None:
    records = list(adapter.load_filtered(["positivo"]))
    records_with_classes = [r for r in records if r.classes]
    assert len(records_with_classes) > 0

    sample = records_with_classes[0]
    for cls in sample.classes:
        assert cls.class_name
        assert cls.class_id


def test_filter_negativo_only(adapter: DuckDBDatasetAdapter) -> None:
    records = list(adapter.load_filtered(["negativo"]))
    assert all(r.data_category_qa == "negativo" for r in records)
