from __future__ import annotations

from pydantic import BaseModel

from camel.domain.value_objects import Chunk, ClassDef


class DatasetRecord(BaseModel, frozen=True):
    id: str
    question: str
    content: str
    context_metadata: str
    name: str
    occupation: str
    adjective: str
    chatbot_goal: str
    instructions: list[str]
    chunks_big: list[Chunk]
    classes: list[ClassDef]
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
