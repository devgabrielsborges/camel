from __future__ import annotations

from typing import Any, Protocol


class ScorerPort(Protocol):
    """A callable scorer compatible with MLflow's @scorer interface."""

    name: str

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
    ) -> Any: ...
