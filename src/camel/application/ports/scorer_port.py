from __future__ import annotations

from typing import Protocol

from camel.domain.value_objects.score import Score


class ScorerPort(Protocol):
    def score(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        expectations: dict[str, str],
    ) -> list[Score]: ...
