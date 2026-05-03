from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
from statsmodels.stats.multitest import multipletests

from camel.domain.services.effect_sizes import classify_magnitude, cohens_d, odds_ratio
from camel.domain.value_objects.category_score_collection import CategoryScoreCollection
from camel.domain.value_objects.metric_threshold import MetricThreshold
from camel.domain.value_objects.metric_type import MetricType
from camel.domain.value_objects.test_result import TestResult
from camel.domain.value_objects.threshold_profile import ThresholdProfile

logger = logging.getLogger(__name__)

_DISCORDANT_EXACT_THRESHOLD = 25


def detect_pairing(
    candidate_ids: Sequence[str],
    reference_ids: Sequence[str],
    threshold: float = 0.80,
) -> tuple[bool, tuple[str, ...]]:
    """Detect whether two sets of session IDs are sufficiently paired.

    Returns (is_paired, common_session_ids sorted).
    """
    cand_set = set(candidate_ids)
    ref_set = set(reference_ids)
    common = cand_set & ref_set
    if len(cand_set) == 0:
        return False, ()
    overlap_ratio = len(common) / len(cand_set)
    is_paired = overlap_ratio >= threshold
    return is_paired, tuple(sorted(common))


def run_mcnemar(
    candidate_binary: Sequence[float],
    reference_binary: Sequence[float],
    session_ids: Sequence[str],
    *,
    metric_name: str,
    category: str,
) -> TestResult:
    """Run McNemar's test on paired binary data."""
    cand = np.asarray(candidate_binary, dtype=int)
    ref = np.asarray(reference_binary, dtype=int)
    n = len(cand)

    a = int(np.sum((cand == 1) & (ref == 1)))
    b = int(np.sum((cand == 1) & (ref == 0)))
    c = int(np.sum((cand == 0) & (ref == 1)))
    d = int(np.sum((cand == 0) & (ref == 0)))

    table = np.array([[a, b], [c, d]])
    discordant = b + c
    use_exact = discordant < _DISCORDANT_EXACT_THRESHOLD

    result = statsmodels_mcnemar(table, exact=use_exact)
    p_value = float(result.pvalue)
    statistic = float(result.statistic) if result.statistic is not None else 0.0

    or_val = odds_ratio(((a, b), (c, d)))
    magnitude = classify_magnitude(or_val, "binary")

    cand_rate = float(np.mean(cand))
    ref_rate = float(np.mean(ref))

    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="mcnemar",
        statistic=statistic,
        p_value=p_value,
        p_value_adjusted=p_value,
        reject_null=False,
        effect_size=or_val,
        effect_magnitude=magnitude,
        ci_lower=0.0,
        ci_upper=0.0,
        candidate_value=cand_rate,
        threshold_value=ref_rate,
        sample_size_candidate=n,
        sample_size_reference=n,
        is_paired=True,
    )


def run_wilcoxon(
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    metric_name: str,
    category: str,
) -> TestResult:
    """Run Wilcoxon signed-rank test on paired continuous data."""
    cand = np.asarray(candidate_scores, dtype=np.float64)
    ref = np.asarray(reference_scores, dtype=np.float64)
    diffs = cand - ref
    nonzero_diffs = diffs[diffs != 0]

    if len(nonzero_diffs) == 0:
        statistic_val = 0.0
        p_value = 1.0
    else:
        scipy_result = stats.wilcoxon(nonzero_diffs)
        statistic_val = float(scipy_result.statistic)
        p_value = float(scipy_result.pvalue)

    d = cohens_d(tuple(cand.tolist()), tuple(ref.tolist()))
    magnitude = classify_magnitude(d, "continuous")

    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="wilcoxon",
        statistic=statistic_val,
        p_value=p_value,
        p_value_adjusted=p_value,
        reject_null=False,
        effect_size=d,
        effect_magnitude=magnitude,
        ci_lower=0.0,
        ci_upper=0.0,
        candidate_value=float(np.mean(cand)),
        threshold_value=float(np.mean(ref)),
        sample_size_candidate=len(cand),
        sample_size_reference=len(ref),
        is_paired=True,
    )


def run_mannwhitneyu(
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    metric_name: str,
    category: str,
) -> TestResult:
    """Run Mann-Whitney U test on unpaired continuous data."""
    cand = np.asarray(candidate_scores, dtype=np.float64)
    ref = np.asarray(reference_scores, dtype=np.float64)

    scipy_result = stats.mannwhitneyu(cand, ref, alternative="two-sided")
    statistic_val = float(scipy_result.statistic)
    p_value = float(scipy_result.pvalue)

    d = cohens_d(tuple(cand.tolist()), tuple(ref.tolist()))
    magnitude = classify_magnitude(d, "continuous")

    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="mannwhitneyu",
        statistic=statistic_val,
        p_value=p_value,
        p_value_adjusted=p_value,
        reject_null=False,
        effect_size=d,
        effect_magnitude=magnitude,
        ci_lower=0.0,
        ci_upper=0.0,
        candidate_value=float(np.mean(cand)),
        threshold_value=float(np.mean(ref)),
        sample_size_candidate=len(cand),
        sample_size_reference=len(ref),
        is_paired=False,
    )


def run_chi2(
    candidate_binary: Sequence[float],
    reference_binary: Sequence[float],
    *,
    metric_name: str,
    category: str,
) -> TestResult:
    """Run chi-squared test of independence on unpaired binary data."""
    cand = np.asarray(candidate_binary, dtype=int)
    ref = np.asarray(reference_binary, dtype=int)

    cand_pos = int(np.sum(cand == 1))
    cand_neg = len(cand) - cand_pos
    ref_pos = int(np.sum(ref == 1))
    ref_neg = len(ref) - ref_pos

    table = np.array([[cand_pos, cand_neg], [ref_pos, ref_neg]])
    if table.min() == 0 and table.max() == 0:
        return TestResult(
            metric_name=metric_name, category=category, test_name="chi2",
            statistic=0.0, p_value=1.0, p_value_adjusted=1.0,
            reject_null=False, effect_size=1.0, effect_magnitude="negligible",
            ci_lower=0.0, ci_upper=0.0,
            candidate_value=0.0, threshold_value=0.0,
            sample_size_candidate=len(cand), sample_size_reference=len(ref),
            is_paired=False,
        )

    chi2_stat, p_value, _, _ = stats.chi2_contingency(table, correction=True)

    or_table = ((cand_pos, cand_neg), (ref_pos, ref_neg))
    or_val = odds_ratio(or_table)
    magnitude = classify_magnitude(or_val, "binary")

    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="chi2",
        statistic=float(chi2_stat),
        p_value=float(p_value),
        p_value_adjusted=float(p_value),
        reject_null=False,
        effect_size=or_val,
        effect_magnitude=magnitude,
        ci_lower=0.0,
        ci_upper=0.0,
        candidate_value=float(np.mean(cand)),
        threshold_value=float(np.mean(ref)),
        sample_size_candidate=len(cand),
        sample_size_reference=len(ref),
        is_paired=False,
    )


def run_bootstrap_ci(
    candidate_scores: Sequence[float],
    threshold_value: float,
    *,
    bootstrap_b: int,
    seed: int,
    metric_name: str,
    category: str,
) -> TestResult:
    """Run bootstrap CI test: reject if CI entirely below threshold."""
    arr = np.asarray(candidate_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)

    boot_means = np.empty(bootstrap_b, dtype=np.float64)
    for i in range(bootstrap_b):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    candidate_mean = float(np.mean(arr))

    reject = ci_upper < threshold_value

    relative_diff = (candidate_mean - threshold_value) / threshold_value if threshold_value != 0 else 0.0

    return TestResult(
        metric_name=metric_name,
        category=category,
        test_name="bootstrap",
        statistic=candidate_mean,
        p_value=0.0,
        p_value_adjusted=0.0,
        reject_null=reject,
        effect_size=relative_diff,
        effect_magnitude=classify_magnitude(relative_diff, "continuous"),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        candidate_value=candidate_mean,
        threshold_value=threshold_value,
        sample_size_candidate=n,
        sample_size_reference=0,
        is_paired=False,
    )


def apply_bh_correction(
    test_results: list[TestResult],
    alpha: float,
) -> list[TestResult]:
    """Apply Benjamini-Hochberg FDR correction to a list of TestResults."""
    if not test_results:
        return []

    raw_pvals = [r.p_value for r in test_results]
    reject_arr, adjusted_arr, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")

    corrected: list[TestResult] = []
    for result, adj_p, rej in zip(test_results, adjusted_arr, reject_arr):
        corrected.append(
            replace(result, p_value_adjusted=float(adj_p), reject_null=bool(rej))
        )
    return corrected


def _run_test_for_metric(
    metric_threshold: MetricThreshold,
    category: str,
    candidate_scores: tuple[float, ...],
    reference_scores: tuple[float, ...] | None,
    common_ids: tuple[str, ...] | None,
    is_paired: bool,
    profile: ThresholdProfile,
) -> TestResult:
    """Route a single metric to the appropriate statistical test."""
    mt = metric_threshold.metric_type
    mn = metric_threshold.metric_name

    if mt == MetricType.COMPOSITE:
        return run_bootstrap_ci(
            candidate_scores,
            metric_threshold.value,
            bootstrap_b=profile.bootstrap_b,
            seed=profile.bootstrap_seed,
            metric_name=mn,
            category=category,
        )

    if is_paired and reference_scores is not None and common_ids is not None:
        if mt == MetricType.BINARY:
            return run_mcnemar(
                candidate_scores, reference_scores, common_ids,
                metric_name=mn, category=category,
            )
        return run_wilcoxon(
            candidate_scores, reference_scores,
            metric_name=mn, category=category,
        )

    if reference_scores is not None:
        if mt == MetricType.BINARY:
            return run_chi2(
                candidate_scores, reference_scores,
                metric_name=mn, category=category,
            )
        return run_mannwhitneyu(
            candidate_scores, reference_scores,
            metric_name=mn, category=category,
        )

    return run_bootstrap_ci(
        candidate_scores,
        metric_threshold.value,
        bootstrap_b=profile.bootstrap_b,
        seed=profile.bootstrap_seed,
        metric_name=mn,
        category=category,
    )


def run_all_tests(
    candidate_collections: list[CategoryScoreCollection],
    reference_collections: list[CategoryScoreCollection],
    profile: ThresholdProfile,
) -> list[TestResult]:
    """Orchestrate all hypothesis tests for a candidate model against a profile."""
    cand_by_cat = {c.category: c for c in candidate_collections}
    ref_by_cat = {c.category: c for c in reference_collections}

    all_results: list[TestResult] = []

    for category, thresholds in profile.category_thresholds.items():
        cand_col = cand_by_cat.get(category)
        ref_col = ref_by_cat.get(category)

        if cand_col is None:
            logger.warning("No candidate data for category '%s' — skipping", category)
            continue

        is_paired = False
        common_ids: tuple[str, ...] | None = None
        if ref_col is not None:
            is_paired, common_ids = detect_pairing(cand_col.session_ids, ref_col.session_ids)

        for mt in thresholds:
            cand_scores = cand_col.scores.get(mt.metric_name)
            if cand_scores is None:
                logger.warning(
                    "Metric '%s' not found in candidate category '%s' — skipping",
                    mt.metric_name, category,
                )
                continue

            ref_scores: tuple[float, ...] | None = None
            if ref_col is not None:
                ref_scores = ref_col.scores.get(mt.metric_name)

            if is_paired and ref_scores is not None and common_ids is not None and ref_col is not None:
                cand_id_idx = {sid: i for i, sid in enumerate(cand_col.session_ids)}
                ref_id_idx = {sid: i for i, sid in enumerate(ref_col.session_ids)}
                paired_cand = tuple(cand_scores[cand_id_idx[sid]] for sid in common_ids if sid in cand_id_idx)
                paired_ref = tuple(ref_scores[ref_id_idx[sid]] for sid in common_ids if sid in ref_id_idx)
                result = _run_test_for_metric(
                    mt, category, paired_cand, paired_ref, common_ids, True, profile,
                )
            else:
                result = _run_test_for_metric(
                    mt, category, cand_scores, ref_scores, None, False, profile,
                )

            all_results.append(result)

    for mt in profile.global_thresholds:
        cand_scores_for_global = _collect_global_metric(mt.metric_name, cand_by_cat)
        if cand_scores_for_global is not None:
            result = _run_test_for_metric(
                mt, "global", cand_scores_for_global, None, None, False, profile,
            )
            all_results.append(result)

    non_bootstrap = [r for r in all_results if r.test_name != "bootstrap"]
    bootstrap_results = [r for r in all_results if r.test_name == "bootstrap"]

    if non_bootstrap:
        corrected = apply_bh_correction(non_bootstrap, profile.alpha)
    else:
        corrected = []

    return corrected + bootstrap_results


def _collect_global_metric(
    metric_name: str,
    cand_by_cat: dict[str, CategoryScoreCollection],
) -> tuple[float, ...] | None:
    """Compute discrimination_delta from positivo/negativo overlap scores."""
    if metric_name != "discrimination_delta":
        return None

    pos = cand_by_cat.get("positivo")
    neg = cand_by_cat.get("negativo")
    if pos is None or neg is None:
        return None

    pos_overlap = pos.scores.get("token_overlap_f1")
    neg_overlap = neg.scores.get("token_overlap_f1")
    if pos_overlap is None or neg_overlap is None:
        return None

    pos_mean = float(np.mean(pos_overlap))
    neg_mean = float(np.mean(neg_overlap))
    delta = abs(pos_mean - neg_mean)
    return (delta,)
