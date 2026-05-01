from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects import Chunk, ClassDef


class DatasetRecord(BaseModel, frozen=True):
    id: str
    question: str
    content: dict[str, str]
    context_metadata: dict[str, str]
    name: str
    occupation: str
    adjective: str
    chatbot_goal: str
    instructions: list[str]
    chunks_big: dict[str, list[Chunk]]
    classes: dict[str, list[ClassDef]]
    chosen_class_id: str
    language: int
    data_category_qa: str
    content_base_uuids: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetRecord):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
