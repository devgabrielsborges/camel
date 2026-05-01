from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects.dataset_record import DatasetRecord


class InferenceRequest(BaseModel, frozen=True):
    dataset_record: DatasetRecord
    rendered_prompt: str
