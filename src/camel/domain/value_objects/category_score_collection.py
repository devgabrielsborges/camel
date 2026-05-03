from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryScoreCollection:
    category: str
    session_ids: tuple[str, ...]
    scores: dict[str, tuple[float, ...]]
