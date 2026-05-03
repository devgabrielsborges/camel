from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from camel.infrastructure.dashboard.charts import (
    box_plots,
    cost_performance_scatter,
    failure_mode_bars,
    performance_vs_complexity,
    radar_chart,
    sankey_diagram,
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


def _make_scatter_df(n_rows: int = 10, run_id: str = "model_a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [run_id] * n_rows,
            "session_id": [f"sess_{i}" for i in range(n_rows)],
            "output_len": list(range(50, 50 + n_rows)),
            "groundedness": [0.3 + (i % 5) * 0.15 for i in range(n_rows)],
            "failure_mode": ["none"] * (n_rows // 2) + ["incorrect"] * (n_rows - n_rows // 2),
            "language_label": ["English"] * n_rows,
        }
    )


class TestCostPerformanceScatter:
    def test_returns_figure(self) -> None:
        df = _make_scatter_df()
        fig = cost_performance_scatter(df)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_traces(self) -> None:
        df = _make_scatter_df()
        fig = cost_performance_scatter(df)
        assert len(fig.data) > 0

    def test_multi_model_color(self) -> None:
        df_a = _make_scatter_df(run_id="model_a")
        df_b = _make_scatter_df(run_id="model_b")
        df = pd.concat([df_a, df_b], ignore_index=True)
        fig = cost_performance_scatter(df)
        assert len(fig.data) >= 2

    def test_hover_data_includes_session_id(self) -> None:
        df = _make_scatter_df()
        fig = cost_performance_scatter(df)
        trace = fig.data[0]
        assert trace.customdata is not None

    def test_missing_x_col_returns_empty(self) -> None:
        df = _make_scores_df()
        fig = cost_performance_scatter(df, x_col="output_len")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


def _make_sankey_df(n_rows: int = 20) -> pd.DataFrame:
    categories = ["positivo", "negativo"] * (n_rows // 2)
    refusals = [True, False, False, True] * (n_rows // 4)
    failures = ["none", "incorrect", "none", "false_refusal"] * (n_rows // 4)
    return pd.DataFrame(
        {
            "run_id": ["model_a"] * n_rows,
            "data_category_QA": categories[:n_rows],
            "refusal_detection": refusals[:n_rows],
            "failure_mode": failures[:n_rows],
        }
    )


class TestSankeyDiagram:
    def test_returns_figure(self) -> None:
        df = _make_sankey_df()
        fig = sankey_diagram(df)
        assert isinstance(fig, go.Figure)

    def test_has_sankey_trace(self) -> None:
        df = _make_sankey_df()
        fig = sankey_diagram(df)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Sankey)

    def test_node_labels_contain_categories(self) -> None:
        df = _make_sankey_df()
        fig = sankey_diagram(df)
        labels = list(fig.data[0].node.label)
        assert "positivo" in labels
        assert "negativo" in labels

    def test_node_labels_contain_failure_modes(self) -> None:
        df = _make_sankey_df()
        fig = sankey_diagram(df)
        labels = list(fig.data[0].node.label)
        assert "none" in labels
        assert "incorrect" in labels

    def test_links_sum_to_total_rows(self) -> None:
        df = _make_sankey_df(n_rows=20)
        fig = sankey_diagram(df)
        total_link_value = sum(fig.data[0].link.value)
        assert total_link_value == 2 * len(df)

    def test_missing_columns_returns_empty(self) -> None:
        df = pd.DataFrame({"run_id": ["a"], "correctness": [1.0]})
        fig = sankey_diagram(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
