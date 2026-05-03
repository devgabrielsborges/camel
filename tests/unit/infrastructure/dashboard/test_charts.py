from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from camel.infrastructure.dashboard.charts import (
    box_plots,
    failure_mode_bars,
    performance_vs_complexity,
    radar_chart,
)

_METRIC_COLS = ["correctness", "groundedness", "token_overlap_f1"]


def _make_scores_df(n_rows: int = 10, run_id: str = "model_a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [run_id] * n_rows,
            "data_category_QA": ["positivo"] * (n_rows // 2)
            + ["negativo"] * (n_rows - n_rows // 2),
            "language_label": ["English"] * n_rows,
            "failure_mode": ["none"] * (n_rows // 2) + ["incorrect"] * (n_rows - n_rows // 2),
            "correctness": [1.0, 0.0] * (n_rows // 2),
            "groundedness": [0.8, 0.6, 0.9, 0.7, 0.5] * (n_rows // 5 or 1),
            "token_overlap_f1": [0.7, 0.3, 0.8, 0.4, 0.6] * (n_rows // 5 or 1),
        }
    )


class TestRadarChart:
    def test_returns_figure(self) -> None:
        df = _make_scores_df()
        fig = radar_chart(df, _METRIC_COLS)
        assert isinstance(fig, go.Figure)

    def test_single_group_trace_count(self) -> None:
        df = _make_scores_df()
        fig = radar_chart(df, _METRIC_COLS)
        assert len(fig.data) == 1

    def test_multiple_groups_trace_count(self) -> None:
        df_a = _make_scores_df(run_id="model_a")
        df_b = _make_scores_df(run_id="model_b")
        df = pd.concat([df_a, df_b], ignore_index=True)
        fig = radar_chart(df, _METRIC_COLS)
        assert len(fig.data) == 2

    def test_theta_labels_match_metrics(self) -> None:
        df = _make_scores_df()
        fig = radar_chart(df, _METRIC_COLS)
        theta = list(fig.data[0].theta)
        for metric in _METRIC_COLS:
            assert metric in theta


class TestFailureModeBars:
    def test_returns_figure(self) -> None:
        df = _make_scores_df()
        fig = failure_mode_bars(df)
        assert isinstance(fig, go.Figure)

    def test_has_bar_traces(self) -> None:
        df = _make_scores_df()
        fig = failure_mode_bars(df)
        assert len(fig.data) > 0

    def test_grouped_by_column(self) -> None:
        df = _make_scores_df()
        fig = failure_mode_bars(df, group_col="data_category_QA")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestBoxPlots:
    def test_returns_figure(self) -> None:
        df = _make_scores_df()
        fig = box_plots(df, _METRIC_COLS)
        assert isinstance(fig, go.Figure)

    def test_single_color_group(self) -> None:
        df = _make_scores_df()
        fig = box_plots(df, _METRIC_COLS)
        assert len(fig.data) > 0

    def test_multi_model_side_by_side(self) -> None:
        df_a = _make_scores_df(run_id="model_a")
        df_b = _make_scores_df(run_id="model_b")
        df = pd.concat([df_a, df_b], ignore_index=True)
        fig = box_plots(df, _METRIC_COLS)
        assert len(fig.data) > 0


def _make_complexity_df(n_rows: int = 20, run_id: str = "model_a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [run_id] * n_rows,
            "input_len": list(range(10, 10 + n_rows)),
            "correctness": [0.5 + (i % 5) * 0.1 for i in range(n_rows)],
            "groundedness": [0.4 + (i % 4) * 0.15 for i in range(n_rows)],
        }
    )


class TestPerformanceVsComplexity:
    def test_returns_figure(self) -> None:
        df = _make_complexity_df()
        fig = performance_vs_complexity(df, ["correctness", "groundedness"])
        assert isinstance(fig, go.Figure)

    def test_single_model_traces(self) -> None:
        df = _make_complexity_df()
        fig = performance_vs_complexity(df, ["correctness"])
        assert len(fig.data) >= 1

    def test_multi_model_traces(self) -> None:
        df_a = _make_complexity_df(run_id="model_a")
        df_b = _make_complexity_df(run_id="model_b")
        df = pd.concat([df_a, df_b], ignore_index=True)
        fig = performance_vs_complexity(df, ["correctness"])
        assert len(fig.data) >= 2

    def test_missing_input_len_returns_empty(self) -> None:
        df = _make_scores_df()
        fig = performance_vs_complexity(df, ["correctness"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
