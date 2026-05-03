from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def compute_descriptive_stats(series: pd.Series) -> dict[str, float | int]:
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "variance": float("nan"),
            "cv": float("nan"),
            "q1": float("nan"),
            "q2": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if n > 1 else float("nan")
    variance = float(np.var(clean, ddof=1)) if n > 1 else float("nan")

    cv: float
    if math.isnan(std) or mean == 0:
        cv = float("nan")
    else:
        cv = std / mean

    q1 = float(np.percentile(clean, 25))
    q2 = float(np.percentile(clean, 50))
    q3 = float(np.percentile(clean, 75))

    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "cv": cv,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "count": n,
    }


def compute_z_scores(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if len(clean) < 2:
        return pd.Series(float("nan"), index=series.index)

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))

    if std == 0:
        return pd.Series(float("nan"), index=series.index)

    result = (series - mean) / std
    return result


def compute_comparison_stats(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict[str, Any]:
    stats_a = compute_descriptive_stats(series_a)
    stats_b = compute_descriptive_stats(series_b)

    delta = stats_a["mean"] - stats_b["mean"]

    if math.isnan(stats_b["mean"]) or stats_b["mean"] == 0:
        relative_delta_pct = float("nan")
    else:
        relative_delta_pct = (delta / stats_b["mean"]) * 100

    clean_a = series_a.dropna()
    clean_b = series_b.dropna()
    if len(clean_a) < 2 or len(clean_b) < 2:
        p_value = float("nan")
    else:
        _, p_value = ttest_ind(clean_a, clean_b, equal_var=False)
        p_value = float(p_value)

    significant = False if math.isnan(p_value) else p_value < 0.05

    return {
        "model_a": stats_a,
        "model_b": stats_b,
        "delta": delta,
        "relative_delta_pct": relative_delta_pct,
        "p_value": p_value,
        "significant": significant,
    }


def build_stats_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "run_id",
) -> pd.DataFrame:
    records: dict[str, dict[tuple[str, str], float | int]] = {}

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        row: dict[tuple[str, str], float | int] = {}
        for group_name, group_df in df.groupby(group_col):
            stats = compute_descriptive_stats(group_df[metric])
            for stat_name, stat_val in stats.items():
                row[(str(group_name), stat_name)] = stat_val
        records[metric] = row

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).T
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=[group_col, "statistic"])
    result.index.name = "metric"
    return result


def build_comparison_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    model_a: str,
    model_b: str,
    group_col: str = "run_id",
) -> pd.DataFrame:
    df_a = df[df[group_col] == model_a]
    df_b = df[df[group_col] == model_b]

    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        comp = compute_comparison_stats(df_a[metric], df_b[metric])
        rows.append(
            {
                "metric": metric,
                f"{model_a}_mean": comp["model_a"]["mean"],
                f"{model_a}_std": comp["model_a"]["std"],
                f"{model_b}_mean": comp["model_b"]["mean"],
                f"{model_b}_std": comp["model_b"]["std"],
                "delta": comp["delta"],
                "relative_delta_pct": comp["relative_delta_pct"],
                "p_value": comp["p_value"],
                "significant": comp["significant"],
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("metric")
    return result
