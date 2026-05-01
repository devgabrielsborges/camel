from __future__ import annotations

import logging
from collections.abc import Iterator

from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)


def _record_to_mlflow_dict(record: DatasetRecord) -> dict[str, object]:
    return {
        "inputs": {
            "question": record.question,
            "language": record.language,
        },
        "expectations": {
            "expected_response": record.content,
            "guidelines": "; ".join(record.instructions),
            "chosen_class_id": record.chosen_class_id,
        },
        "tags": {
            "data_category_QA": record.data_category_qa,
            "session_id": record.id,
        },
    }


class RegisterDataset:
    def __init__(
        self,
        dataset_adapter: DuckDBDatasetAdapter,
        tracker_adapter: MLflowTrackerAdapter,
    ) -> None:
        self._dataset = dataset_adapter
        self._tracker = tracker_adapter

    def execute(
        self,
        dataset_name: str,
        categories: list[str],
        limit: int | None = None,
    ) -> int:
        rows: Iterator[DatasetRecord] = self._dataset.load_filtered(categories)
        mlflow_records: list[dict[str, object]] = []
        count = 0

        for record in rows:
            if limit is not None and count >= limit:
                break
            mlflow_records.append(_record_to_mlflow_dict(record))
            count += 1

        if mlflow_records:
            self._tracker.register_dataset(dataset_name, mlflow_records)

        logger.info("Registered %d records as dataset '%s'", count, dataset_name)
        return count
