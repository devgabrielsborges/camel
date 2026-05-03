from __future__ import annotations

import math

import numpy as np
import pandas as pd


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
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=[group_col, "statistic"]
    )
    result.index.name = "metric"
    return result
