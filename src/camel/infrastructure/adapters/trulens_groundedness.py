from __future__ import annotations

import logging

from trulens.providers.litellm import LiteLLM as TruLensLiteLLM

from camel.application.ports.groundedness_port import GroundednessPort

logger = logging.getLogger(__name__)


class TruLensGroundednessAdapter(GroundednessPort):
    """Adapter implementing GroundednessPort using TruLens LiteLLM provider."""

    def __init__(self, model_engine: str) -> None:
        self._provider = TruLensLiteLLM(model_engine=model_engine)

    def score(self, source: str, statement: str) -> tuple[float, str]:
        """Score groundedness of statement against source content.

        Returns (score 0.0-1.0, chain-of-thought reasoning).
        """
        try:
            result = self._provider.groundedness_measure_with_cot_reasons(
                source=source,
                statement=statement,
            )
            score_value = float(result[0]) if result[0] is not None else 0.0
            reasons = str(result[1]) if len(result) > 1 else ""
            return (min(max(score_value, 0.0), 1.0), reasons)
        except Exception:
            logger.warning("Groundedness scoring failed", exc_info=True)
            return (0.0, "scoring_error")
