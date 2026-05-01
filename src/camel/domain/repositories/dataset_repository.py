from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from camel.domain.value_objects.dataset_record import DatasetRecord


class DatasetRepository(Protocol):
    def load_filtered(self, categories: list[str]) -> Iterator[DatasetRecord]: ...

    def count(self, categories: list[str]) -> int: ...
