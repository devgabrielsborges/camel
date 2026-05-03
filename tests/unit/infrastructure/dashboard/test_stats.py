from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from camel.infrastructure.dashboard.stats import (
    build_comparison_table,
    build_stats_table,
    compute_comparison_stats,
    compute_descriptive_stats,
    compute_z_scores,
)


class TestComputeDescriptiveStats:
    def test_normal_series(self) -> None:
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_descriptive_stats(data)

        assert result["count"] == 5
        assert round(result["mean"], 4) == round(float(np.mean(data)), 4)
        assert round(result["std"], 4) == round(float(np.std(data, ddof=1)), 4)
        assert round(result["variance"], 4) == round(float(np.var(data, ddof=1)), 4)
        assert round(result["q1"], 4) == round(float(np.percentile(data, 25)), 4)
        assert round(result["q2"], 4) == round(float(np.percentile(data, 50)), 4)
        assert round(result["q3"], 4) == round(float(np.percentile(data, 75)), 4)
        assert round(result["iqr"], 4) == round(result["q3"] - result["q1"], 4)
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_cv_computation(self) -> None:
        data = pd.Series([10.0, 20.0, 30.0])
        result = compute_descriptive_stats(data)
        expected_cv = result["std"] / result["mean"]
        assert round(result["cv"], 4) == round(expected_cv, 4)

    def test_all_nan_series(self) -> None:
        data = pd.Series([float("nan"), float("nan"), float("nan")])
        result = compute_descriptive_stats(data)

        assert result["count"] == 0
        assert math.isnan(result["mean"])
        assert math.isnan(result["std"])
        assert math.isnan(result["variance"])
        assert math.isnan(result["cv"])
        assert math.isnan(result["q1"])
        assert math.isnan(result["q2"])
        assert math.isnan(result["q3"])
        assert math.isnan(result["iqr"])
        assert math.isnan(result["min"])
        assert math.isnan(result["max"])

    def test_single_value_series(self) -> None:
        data = pd.Series([42.0])
        result = compute_descriptive_stats(data)

        assert result["count"] == 1
        assert result["mean"] == 42.0
        assert math.isnan(result["std"])
        assert math.isnan(result["variance"])
        assert math.isnan(result["cv"])
        assert result["q1"] == 42.0
        assert result["q2"] == 42.0
        assert result["q3"] == 42.0
        assert result["min"] == 42.0
        assert result["max"] == 42.0

    def test_mixed_nan_series(self) -> None:
        data = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0])
        result = compute_descriptive_stats(data)

        clean = data.dropna()
        assert result["count"] == 3
        assert round(result["mean"], 4) == round(float(np.mean(clean)), 4)
        assert round(result["std"], 4) == round(float(np.std(clean, ddof=1)), 4)

    def test_zero_mean_cv_is_nan(self) -> None:
        data = pd.Series([-1.0, 0.0, 1.0])
        result = compute_descriptive_stats(data)
        assert result["mean"] == 0.0
        assert math.isnan(result["cv"])

    def test_empty_series(self) -> None:
        data = pd.Series([], dtype=float)
        result = compute_descriptive_stats(data)
        assert result["count"] == 0
        assert math.isnan(result["mean"])


class TestComputeZScores:
    def test_normal_z_scores(self) -> None:
        data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        z = compute_z_scores(data)

        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        expected = (data - mean) / std

        pd.testing.assert_series_equal(z, expected, atol=1e-10)

    def test_nan_preserved(self) -> None:
        data = pd.Series([1.0, float("nan"), 3.0, 4.0, 5.0])
        z = compute_z_scores(data)

        assert math.isnan(z.iloc[1])
        assert not math.isnan(z.iloc[0])
        assert not math.isnan(z.iloc[2])

    def test_constant_series_returns_nan(self) -> None:
        data = pd.Series([5.0, 5.0, 5.0])
        z = compute_z_scores(data)
        assert all(math.isnan(v) for v in z)

    def test_single_value_returns_nan(self) -> None:
        data = pd.Series([42.0])
        z = compute_z_scores(data)
        assert all(math.isnan(v) for v in z)


class TestBuildStatsTable:
    def test_single_group(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["model_a"] * 5,
                "metric_x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        table = build_stats_table(df, metric_cols=["metric_x"])

        assert "metric_x" in table.index
        assert ("model_a", "mean") in table.columns
        assert table.loc["metric_x", ("model_a", "mean")] == pytest.approx(3.0)

    def test_multiple_groups(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["a"] * 3 + ["b"] * 3,
                "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        table = build_stats_table(df, metric_cols=["score"])

        assert ("a", "mean") in table.columns
        assert ("b", "mean") in table.columns
        assert table.loc["score", ("a", "mean")] == pytest.approx(2.0)
        assert table.loc["score", ("b", "mean")] == pytest.approx(5.0)

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"run_id": ["a"], "x": [1.0]})
        table = build_stats_table(df, metric_cols=["x", "nonexistent"])
        assert "x" in table.index
        assert "nonexistent" not in table.index


class TestComputeComparisonStats:
    def test_identical_series(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_comparison_stats(a, b)

        assert result["delta"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["significant"] is False

    def test_significantly_different_series(self) -> None:
        a = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_comparison_stats(a, b)

        assert result["delta"] == pytest.approx(99.0)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_one_series_too_small_for_ttest(self) -> None:
        a = pd.Series([5.0])
        b = pd.Series([1.0, 2.0, 3.0])
        result = compute_comparison_stats(a, b)

        assert math.isnan(result["p_value"])
        assert result["significant"] is False

    def test_zero_mean_b_relative_delta_nan(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([-1.0, 0.0, 1.0])
        result = compute_comparison_stats(a, b)

        assert result["model_b"]["mean"] == pytest.approx(0.0)
        assert math.isnan(result["relative_delta_pct"])

    def test_contains_descriptive_stats(self) -> None:
        a = pd.Series([10.0, 20.0, 30.0])
        b = pd.Series([5.0, 15.0, 25.0])
        result = compute_comparison_stats(a, b)

        assert "model_a" in result
        assert "model_b" in result
        assert result["model_a"]["mean"] == pytest.approx(20.0)
        assert result["model_b"]["mean"] == pytest.approx(15.0)
        assert result["relative_delta_pct"] == pytest.approx((5.0 / 15.0) * 100, abs=0.01)


class TestBuildComparisonTable:
    def test_output_columns(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["a"] * 5 + ["b"] * 5,
                "score": [0.8, 0.9, 0.7, 0.85, 0.75, 0.5, 0.6, 0.4, 0.55, 0.45],
            }
        )
        table = build_comparison_table(df, ["score"], model_a="a", model_b="b")

        expected_cols = {
            "a_mean",
            "a_std",
            "b_mean",
            "b_std",
            "delta",
            "relative_delta_pct",
            "p_value",
            "significant",
        }
        assert set(table.columns) == expected_cols
        assert "score" in table.index

    def test_delta_correct(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["a"] * 3 + ["b"] * 3,
                "m": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
            }
        )
        table = build_comparison_table(df, ["m"], model_a="a", model_b="b")
        assert table.loc["m", "delta"] == pytest.approx(5.0)
        assert table.loc["m", "relative_delta_pct"] == pytest.approx(100.0)
