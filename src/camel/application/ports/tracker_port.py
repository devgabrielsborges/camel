from __future__ import annotations

from typing import Any, Protocol

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.prompt_template import PromptTemplate


class TrackerPort(Protocol):
    def enable_autolog(self) -> None: ...

    def disable_autolog(self) -> None: ...

    def start_run(self, evaluation: Evaluation) -> str: ...

    def end_run(self, run_id: str) -> None: ...

    def set_run_tags(self, run_id: str, tags: dict[str, str]) -> None: ...

    def register_prompt(self, template: PromptTemplate) -> str: ...

    def register_dataset(
        self,
        name: str,
        records: list[dict[str, object]],
    ) -> None: ...

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...

    def search_traces(
        self,
        experiment_name: str,
        run_id: str,
    ) -> list[dict[str, Any]]: ...
