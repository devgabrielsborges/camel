from __future__ import annotations

import json
import logging
from pathlib import Path

from camel.domain.value_objects.threshold_profile import ThresholdProfile

logger = logging.getLogger(__name__)


class ThresholdProfileRepository:
    """JSON disk persistence for ThresholdProfile artifacts."""

    def save(self, profile: ThresholdProfile, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        logger.info("ThresholdProfile saved to %s", target)

    def load(self, path: str) -> ThresholdProfile | None:
        target = Path(path)
        if not target.exists():
            logger.warning("ThresholdProfile not found at %s", target)
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
        logger.info("ThresholdProfile loaded from %s (version=%s)", target, data.get("version"))
        return ThresholdProfile.from_dict(data)
