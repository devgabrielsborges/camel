from __future__ import annotations

from pathlib import Path

import jinja2
import tiktoken

from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.domain.value_objects.prompt_template import PromptTemplate

_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class PromptRenderer:
    def __init__(self, template_path: str) -> None:
        raw = Path(template_path).read_text(encoding="utf-8")
        self._env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
        )
        self._template = self._env.from_string(raw)
        self._template_path = template_path

    def render(self, record: DatasetRecord, version_uri: str) -> PromptTemplate:
        rendered = self._template.render(
            name=record.name,
            occupation=record.occupation,
            adjective=record.adjective,
            chatbot_goal=record.chatbot_goal,
            instructions=record.instructions,
            chunks_big={
                k: [c.model_dump() for c in v] for k, v in record.chunks_big.items()
            },
            classes={
                k: [c.model_dump() for c in v] for k, v in record.classes.items()
            },
            language=record.language,
        )
        return PromptTemplate(
            template_path=self._template_path,
            version_uri=version_uri,
            rendered_content=rendered,
            token_count=count_tokens(rendered),
        )
