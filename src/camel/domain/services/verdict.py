from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from camel.domain.services.aggregation import AggregatedMetric, CategoryBreakdown


class Verdict(StrEnum):
    CAPABLE = "capable"
    NOT_CAPABLE = "not_capable"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class VerdictResult:
    verdict: Verdict
    positivo_overlap_mean: float
    negativo_overlap_mean: float
    positivo_refusal_rate: float
    negativo_refusal_rate: float
    discrimination_delta: float
    reasons: list[str]


_OVERLAP_THRESHOLD = 0.3
_REFUSAL_THRESHOLD = 0.3
_DISCRIMINATION_THRESHOLD = 0.1


def _find_metric(metrics: list[AggregatedMetric], scorer_name: str) -> AggregatedMetric | None:
    return next((m for m in metrics if m.scorer_name == scorer_name), None)


def compute_verdict(
    category_breakdowns: list[CategoryBreakdown],
    *,
    overlap_threshold: float = _OVERLAP_THRESHOLD,
    refusal_threshold: float = _REFUSAL_THRESHOLD,
    discrimination_threshold: float = _DISCRIMINATION_THRESHOLD,
) -> VerdictResult:
    """Compute a capability verdict based on per-category metrics.

    The model is considered capable if:
    1. Positivo: mean token_overlap_f1 >= overlap_threshold
    2. Negativo: refusal_rate >= refusal_threshold OR overlap < overlap_threshold
    3. Discrimination: abs(positivo_overlap - negativo_overlap) >= discrimination_threshold
    """
    positivo = next((b for b in category_breakdowns if b.category == "positivo"), None)
    negativo = next((b for b in category_breakdowns if b.category == "negativo"), None)

    reasons: list[str] = []

    pos_overlap = _find_metric(positivo.metrics, "token_overlap_f1") if positivo else None
    neg_overlap = _find_metric(negativo.metrics, "token_overlap_f1") if negativo else None
    pos_refusal = _find_metric(positivo.metrics, "refusal_detection") if positivo else None
    neg_refusal = _find_metric(negativo.metrics, "refusal_detection") if negativo else None

    pos_overlap_mean = pos_overlap.mean if pos_overlap else 0.0
    neg_overlap_mean = neg_overlap.mean if neg_overlap else 0.0
    pos_refusal_rate = pos_refusal.mean if pos_refusal else 0.0
    neg_refusal_rate = neg_refusal.mean if neg_refusal else 0.0

    if positivo is None or negativo is None:
        reasons.append("Missing category data — cannot determine capability")
        return VerdictResult(
            verdict=Verdict.INCONCLUSIVE,
            positivo_overlap_mean=pos_overlap_mean,
            negativo_overlap_mean=neg_overlap_mean,
            positivo_refusal_rate=pos_refusal_rate,
            negativo_refusal_rate=neg_refusal_rate,
            discrimination_delta=0.0,
            reasons=reasons,
        )

    positivo_capable = pos_overlap_mean >= overlap_threshold
    if positivo_capable:
        reasons.append(
            f"Positivo overlap ({pos_overlap_mean:.3f}) >= threshold ({overlap_threshold})"
        )
    else:
        reasons.append(
            f"Positivo overlap ({pos_overlap_mean:.3f}) < threshold ({overlap_threshold})"
        )

    negativo_handles = neg_refusal_rate >= refusal_threshold or neg_overlap_mean < overlap_threshold
    if neg_refusal_rate >= refusal_threshold:
        reasons.append(
            f"Negativo refusal rate ({neg_refusal_rate:.3f}) >= threshold ({refusal_threshold})"
        )
    elif neg_overlap_mean < overlap_threshold:
        reasons.append(
            f"Negativo overlap ({neg_overlap_mean:.3f}) < threshold — avoids hallucination"
        )
    else:
        reasons.append(
            f"Negativo: low refusal ({neg_refusal_rate:.3f}) and high overlap ({neg_overlap_mean:.3f}) — hallucination risk"
        )

    discrimination_delta = abs(pos_overlap_mean - neg_overlap_mean)
    discriminates = discrimination_delta >= discrimination_threshold
    if discriminates:
        reasons.append(
            f"Discrimination delta ({discrimination_delta:.3f}) >= threshold ({discrimination_threshold})"
        )
    else:
        reasons.append(
            f"Discrimination delta ({discrimination_delta:.3f}) < threshold ({discrimination_threshold}) — model does not differentiate categories"
        )

    if positivo_capable and negativo_handles and discriminates:
        verdict = Verdict.CAPABLE
    elif not positivo_capable and not negativo_handles:
        verdict = Verdict.NOT_CAPABLE
    else:
        verdict = Verdict.INCONCLUSIVE

    return VerdictResult(
        verdict=verdict,
        positivo_overlap_mean=pos_overlap_mean,
        negativo_overlap_mean=neg_overlap_mean,
        positivo_refusal_rate=pos_refusal_rate,
        negativo_refusal_rate=neg_refusal_rate,
        discrimination_delta=discrimination_delta,
        reasons=reasons,
    )
