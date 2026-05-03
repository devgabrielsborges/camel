from __future__ import annotations

from camel.domain.services.effect_sizes import classify_magnitude, cohens_d, odds_ratio
from camel.domain.services.hypothesis_tests import (
    apply_bh_correction,
    detect_pairing,
    run_all_tests,
    run_bootstrap_ci,
    run_chi2,
    run_mannwhitneyu,
    run_mcnemar,
    run_wilcoxon,
)
from camel.domain.services.score_collector import collect_raw_scores
from camel.domain.services.statistical_verdict import compute_statistical_verdict
from camel.domain.services.threshold_derivation import derive_thresholds

__all__ = [
    "apply_bh_correction",
    "classify_magnitude",
    "cohens_d",
    "collect_raw_scores",
    "compute_statistical_verdict",
    "derive_thresholds",
    "detect_pairing",
    "odds_ratio",
    "run_all_tests",
    "run_bootstrap_ci",
    "run_chi2",
    "run_mannwhitneyu",
    "run_mcnemar",
    "run_wilcoxon",
]
