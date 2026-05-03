from __future__ import annotations

from typing import Protocol

from camel.domain.value_objects.threshold_profile import ThresholdProfile


class ThresholdProfilePort(Protocol):
    def save(self, profile: ThresholdProfile, path: str) -> None: ...
    def load(self, path: str) -> ThresholdProfile | None: ...
