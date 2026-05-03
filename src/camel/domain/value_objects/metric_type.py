from __future__ import annotations

from enum import StrEnum


class MetricType(StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    COMPOSITE = "composite"
