from __future__ import annotations

from pydantic import BaseModel


class PromptTemplate(BaseModel, frozen=True):
    template_path: str
    version_uri: str
    rendered_content: str | None = None
    token_count: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PromptTemplate):
            return NotImplemented
        return self.version_uri == other.version_uri

    def __hash__(self) -> int:
        return hash(self.version_uri)
