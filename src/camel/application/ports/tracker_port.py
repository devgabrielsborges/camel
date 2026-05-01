from __future__ import annotations

from typing import Protocol

from camel.domain.entities.evaluation import Evaluation
from camel.domain.entities.trace import Trace
from camel.domain.value_objects.prompt_template import PromptTemplate


class TrackerPort(Protocol):
    def start_run(self, evaluation: Evaluation) -> str: ...

    def end_run(self, run_id: str) -> None: ...

    def log_trace(self, run_id: str, trace: Trace) -> None: ...

    def register_prompt(self, template: PromptTemplate) -> str: ...

    def register_dataset(
        self,
        name: str,
        records: list[dict[str, object]],
    ) -> None: ...

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...
