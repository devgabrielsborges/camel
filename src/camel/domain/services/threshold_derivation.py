from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.threshold_profile import ThresholdProfile

_METRIC_TYPES: dict[str, MetricType] = {
    "token_overlap_f1": MetricType.CONTINUOUS,
    "groundedness": MetricType.CONTINUOUS,
    "refusal_detection": MetricType.BINARY,
    "class_exact_match": MetricType.BINARY,
    "pass_at_k": MetricType.BINARY,
}

_TEST_MAP: dict[MetricType, tuple[str, str]] = {
    MetricType.CONTINUOUS: ("wilcoxon", "mannwhitneyu"),
    MetricType.BINARY: ("mcnemar", "chi2"),
    MetricType.COMPOSITE: ("bootstrap", "bootstrap"),
}


def _bootstrap_percentile_ci(
    scores: Sequence[float],
    *,
    bootstrap_b: int,
    seed: int,
    percentile: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Compute a bootstrap percentile threshold with CI.

    Returns (threshold_value, ci_lower, ci_upper).
    """
    arr = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)

    boot_means = np.empty(bootstrap_b, dtype=np.float64)
    for i in range(bootstrap_b):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    threshold = float(np.percentile(boot_means, percentile))
    ci_lower_pct = alpha / 2 * 100
    ci_upper_pct = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(boot_means, ci_lower_pct))
    ci_upper = float(np.percentile(boot_means, ci_upper_pct))

    return threshold, ci_lower, ci_upper


def derive_thresholds(
    collections: list[CategoryScoreCollection],
    *,
    bootstrap_b: int,
    seed: int,
    percentile: int,
    alpha: float,
    benchmark: str,
    reference_models: tuple[str, ...],
    reference_run_ids: tuple[str, ...],
) -> ThresholdProfile:
    """Derive empirical thresholds from reference model score collections."""
    category_thresholds: dict[str, list[MetricThreshold]] = {}
    total_sessions = 0

    categories_data: dict[str, dict[str, tuple[float, ...]]] = {}

    for collection in collections:
        total_sessions += len(collection.session_ids)
        thresholds: list[MetricThreshold] = []
        categories_data[collection.category] = collection.scores

        for metric_name, scores in collection.scores.items():
            if len(scores) == 0:
                continue

            mt = _METRIC_TYPES.get(metric_name, MetricType.CONTINUOUS)
            paired, unpaired = _TEST_MAP.get(mt, ("wilcoxon", "mannwhitneyu"))

            threshold_val, ci_low, ci_up = _bootstrap_percentile_ci(
                scores,
                bootstrap_b=bootstrap_b,
                seed=seed,
                percentile=percentile,
                alpha=alpha,
            )

            arr = np.asarray(scores, dtype=np.float64)
            thresholds.append(
                MetricThreshold(
                    metric_name=metric_name,
                    value=threshold_val,
                    ci_lower=ci_low,
                    ci_upper=ci_up,
                    metric_type=mt,
                    test_paired=paired,
                    test_unpaired=unpaired,
                    reference_mean=float(np.mean(arr)),
                    reference_std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    sample_count=len(scores),
                )
            )

        category_thresholds[collection.category] = thresholds

    global_thresholds = _compute_discrimination_delta(
        categories_data,
        bootstrap_b=bootstrap_b,
        seed=seed,
        percentile=percentile,
        alpha=alpha,
    )

    return ThresholdProfile(
        version="1.0.0",
        created_at=ThresholdProfile.make_timestamp(),
        benchmark=benchmark,
        reference_models=reference_models,
        reference_run_ids=reference_run_ids,
        total_sessions=total_sessions,
        bootstrap_b=bootstrap_b,
        bootstrap_seed=seed,
        threshold_percentile=percentile,
        alpha=alpha,
        correction_method="benjamini-hochberg",
        category_thresholds=category_thresholds,
        global_thresholds=global_thresholds,
    )


def _compute_discrimination_delta(
    categories_data: dict[str, dict[str, tuple[float, ...]]],
    *,
    bootstrap_b: int,
    seed: int,
    percentile: int,
    alpha: float,
) -> list[MetricThreshold]:
    """Compute discrimination delta as |mean(positivo) - mean(negativo)| for token_overlap_f1."""
    pos_scores = categories_data.get("positivo", {}).get("token_overlap_f1")
    neg_scores = categories_data.get("negativo", {}).get("token_overlap_f1")

    if pos_scores is None or neg_scores is None:
        return []

    pos_arr = np.asarray(pos_scores, dtype=np.float64)
    neg_arr = np.asarray(neg_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n_pos, n_neg = len(pos_arr), len(neg_arr)
    boot_deltas = np.empty(bootstrap_b, dtype=np.float64)

    for i in range(bootstrap_b):
        pos_sample = pos_arr[rng.integers(0, n_pos, size=n_pos)]
        neg_sample = neg_arr[rng.integers(0, n_neg, size=n_neg)]
        boot_deltas[i] = abs(float(np.mean(pos_sample)) - float(np.mean(neg_sample)))

    threshold_val = float(np.percentile(boot_deltas, percentile))
    ci_lower_pct = alpha / 2 * 100
    ci_upper_pct = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(boot_deltas, ci_lower_pct))
    ci_upper = float(np.percentile(boot_deltas, ci_upper_pct))

    actual_delta = abs(float(np.mean(pos_arr)) - float(np.mean(neg_arr)))
    combined_n = n_pos + n_neg
    combined_scores = list(pos_scores) + list(neg_scores)
    combined_arr = np.asarray(combined_scores, dtype=np.float64)

    return [
        MetricThreshold(
            metric_name="discrimination_delta",
            value=threshold_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            metric_type=MetricType.COMPOSITE,
            test_paired="bootstrap",
            test_unpaired="bootstrap",
            reference_mean=actual_delta,
            reference_std=float(np.std(boot_deltas, ddof=1)),
            sample_count=combined_n,
        )
    ]
