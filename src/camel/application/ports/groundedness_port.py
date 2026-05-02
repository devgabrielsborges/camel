from __future__ import annotations

from abc import ABC, abstractmethod


class GroundednessPort(ABC):
    """Abstract port for groundedness scoring."""

    @abstractmethod
    def score(self, source: str, statement: str) -> tuple[float, str]:
        """Score how grounded a statement is in the source content.

        Returns:
            A tuple of (score, reasoning) where score is 0.0-1.0 and
            reasoning is a chain-of-thought explanation.
        """
        ...
