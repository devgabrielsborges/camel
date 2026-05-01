from __future__ import annotations

from collections.abc import Iterator

import duckdb

from camel.domain.value_objects import Chunk, ClassDef
from camel.domain.value_objects.dataset_record import DatasetRecord


def _parse_chunks(raw: object) -> list[Chunk]:
    if not isinstance(raw, list):
        return []
    return [
        Chunk(content=c["content"], score=float(c["score"]) if c.get("score") is not None else 0.0)
        for c in raw
    ]


def _parse_classes(raw: object) -> list[ClassDef]:
    if not isinstance(raw, list):
        return []
    return [ClassDef(class_name=c["class"], context=c["context"], class_id=c["id"]) for c in raw]


def _to_str_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


class DuckDBDatasetAdapter:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def load_filtered(self, categories: list[str]) -> Iterator[DatasetRecord]:
        conn = duckdb.connect(self._db_path, read_only=True)
        try:
            placeholders = ", ".join(["?" for _ in categories])
            query = (
                f"SELECT * FROM main.int_filtered_dataset "
                f"WHERE data_category_QA IN ({placeholders})"
            )
            result = conn.execute(query, categories)
            columns = [desc[0] for desc in result.description]

            while True:
                row = result.fetchone()
                if row is None:
                    break
                row_dict = dict(zip(columns, row))
                yield DatasetRecord(
                    id=str(row_dict["id"]),
                    question=str(row_dict["question"]),
                    content=str(row_dict.get("content", "")),
                    context_metadata=str(row_dict.get("context_metadata", "")),
                    name=str(row_dict["name"]),
                    occupation=str(row_dict["occupation"]),
                    adjective=str(row_dict["adjective"]),
                    chatbot_goal=str(row_dict["chatbot_goal"]),
                    instructions=_to_str_list(row_dict.get("instructions")),
                    chunks_big=_parse_chunks(row_dict.get("chunks_big")),
                    classes=_parse_classes(row_dict.get("classes")),
                    chosen_class_id=str(row_dict.get("chosen_class_id", "")),
                    language=int(row_dict.get("language", 0)),
                    data_category_qa=str(row_dict.get("data_category_QA", "")),
                    content_base_uuids=str(row_dict.get("content_base_uuids", "")),
                )
        finally:
            conn.close()

    def count(self, categories: list[str]) -> int:
        conn = duckdb.connect(self._db_path, read_only=True)
        try:
            placeholders = ", ".join(["?" for _ in categories])
            query = (
                f"SELECT count(*) FROM main.int_filtered_dataset "
                f"WHERE data_category_QA IN ({placeholders})"
            )
            result = conn.execute(query, categories).fetchone()
            return int(result[0]) if result else 0
        finally:
            conn.close()
