from __future__ import annotations

import argparse

import pandas as pd
import streamlit as st

from camel.infrastructure.dashboard.charts import (
    box_plots,
    cost_performance_scatter,
    failure_mode_bars,
    performance_vs_complexity,
    radar_chart,
    sankey_diagram,
)
from camel.infrastructure.dashboard.data_loader import METRIC_COLS, load_merged_data
from camel.infrastructure.dashboard.filters import (
    apply_filters,
    render_sidebar_filters,
    show_filter_summary,
)
from camel.infrastructure.dashboard.stats import (
    build_comparison_table,
    build_stats_table,
)

_GROUP_COL = "model_label"
_MAX_RADAR_MODELS = 4
_PAGE_TITLE = "CAMEL Evaluation Dashboard"
_TAB_NAMES = ["Overview", "Comparison", "Distributions", "Failure Modes", "Deep Dive"]


def _parse_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="Path to DuckDB file")
    args, _ = parser.parse_known_args()
    return str(args.db_path)


def main() -> None:
    st.set_page_config(page_title=_PAGE_TITLE, layout="wide")
    st.title(_PAGE_TITLE)

    db_path = _parse_args()

    try:
        df = load_merged_data(db_path)
    except FileNotFoundError:
        st.error(
            f"DuckDB file not found at **{db_path}**. "
            "Run `camel prepare && camel run` first, then `cd dbt && dbt run`."
        )
        st.stop()
        return

    if df.empty:
        st.warning("No evaluation data found in the database.")
        return

    filters = render_sidebar_filters(df)
    df_filtered = apply_filters(df, filters)
    show_filter_summary(df, df_filtered)

    if df_filtered.empty:
        st.warning("No data matches the current filters. Adjust filters in the sidebar.")
        return

    for model_name in df_filtered[_GROUP_COL].unique():
        model_count = len(df_filtered[df_filtered[_GROUP_COL] == model_name])
        if model_count < 2:
            st.warning(
                f"Model **{model_name}** has only {model_count} row(s). "
                "Statistics like std, CV, and t-test require at least 2 samples."
            )

    tabs = st.tabs(_TAB_NAMES)

    _render_overview(tabs[0], df_filtered)
    _render_comparison(tabs[1], df_filtered)
    _render_distributions(tabs[2], df_filtered)
    _render_failure_modes(tabs[3], df_filtered)
    _render_deep_dive(tabs[4], df_filtered)


def _get_available_metrics(df: pd.DataFrame) -> list[str]:
    return [c for c in METRIC_COLS if c in df.columns and df[c].notna().any()]


def _render_overview(tab: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    with tab:
        st.header("Overview")

        available_metrics = _get_available_metrics(df)
        if not available_metrics:
            st.info("No metric columns contain data.")
            return

        n_models = df[_GROUP_COL].nunique()
        if n_models > _MAX_RADAR_MODELS:
            st.warning(
                f"{n_models} models selected — radar chart may be crowded. "
                f"Consider selecting {_MAX_RADAR_MODELS} or fewer."
            )

        cols = st.columns(min(len(available_metrics), 4))
        for i, metric in enumerate(available_metrics):
            series = df[metric].dropna()
            if not series.empty:
                mean_val = float(series.mean())
                std_val = float(series.std())
                cols[i % len(cols)].metric(
                    metric.replace("_", " ").title(),
                    f"{mean_val:.3f}",
                    delta=f"\u00b1 {std_val:.3f}" if not pd.isna(std_val) else None,
                    delta_color="off",
                )

        st.subheader("Metric Radar")
        fig = radar_chart(df, available_metrics, group_col=_GROUP_COL)
        st.plotly_chart(fig, use_container_width=True)


def _highlight_significant(row: pd.Series) -> list[str]:
    if row.get("significant", False):
        return ["background-color: #d4edda"] * len(row)
    return [""] * len(row)


def _render_comparison(tab: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    with tab:
        st.header("Statistics")
        available_metrics = _get_available_metrics(df)
        models = sorted(df[_GROUP_COL].unique())

        stats_df = build_stats_table(df, available_metrics, group_col=_GROUP_COL)
        if not stats_df.empty:
            st.subheader("Descriptive Statistics")
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No metrics available for statistics.")

        if len(models) >= 2:
            st.subheader("Pairwise Comparison")
            col_a, col_b = st.columns(2)
            with col_a:
                model_a = st.selectbox("Model A", models, index=0, key="cmp_model_a")
            with col_b:
                default_b = 1 if len(models) > 1 else 0
                model_b = st.selectbox("Model B", models, index=default_b, key="cmp_model_b")

            if model_a == model_b:
                st.warning("Select two different models to compare.")
            else:
                cmp_df = build_comparison_table(
                    df,
                    available_metrics,
                    model_a=model_a,
                    model_b=model_b,
                    group_col=_GROUP_COL,
                )
                if not cmp_df.empty:
                    styled = cmp_df.style.apply(_highlight_significant, axis=1)
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.info("No comparison data available.")


def _render_distributions(tab: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    with tab:
        st.header("Score Distributions")
        available_metrics = _get_available_metrics(df)
        fig = box_plots(df, available_metrics, group_col=_GROUP_COL)
        st.plotly_chart(fig, use_container_width=True)


def _render_failure_modes(tab: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    with tab:
        st.header("Failure Modes")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("By model")
            fig = failure_mode_bars(df, group_col=_GROUP_COL)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("By category")
            fig = failure_mode_bars(df, group_col="data_category_QA")
            st.plotly_chart(fig, use_container_width=True)

        _render_sankey_section(df)


def _render_sankey_section(df: pd.DataFrame) -> None:
    st.subheader("Trajectory Flow")

    required_cols = {"data_category_QA", "refusal_detection", "failure_mode"}
    if not required_cols.issubset(df.columns):
        st.info("Sankey diagram requires category, refusal, and failure mode columns.")
        return

    models = sorted(df[_GROUP_COL].unique())
    if len(models) <= 1:
        fig = sankey_diagram(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        selected_model = st.selectbox(
            "Select model for trajectory view",
            models,
            key="sankey_model",
        )
        model_df = df[df[_GROUP_COL] == selected_model]
        fig = sankey_diagram(model_df)
        st.plotly_chart(fig, use_container_width=True)


def _render_deep_dive(tab: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    with tab:
        st.header("Deep Dive")

        available_metrics = _get_available_metrics(df)

        if "output_len" in df.columns and available_metrics:
            st.subheader("Cost-Performance Scatter")
            scatter_cols = st.columns(2)
            with scatter_cols[0]:
                x_col = st.selectbox(
                    "X-axis (cost proxy)",
                    ["output_len", "input_len"],
                    index=0,
                    key="scatter_x",
                )
            with scatter_cols[1]:
                y_col = st.selectbox(
                    "Y-axis (metric)",
                    available_metrics,
                    index=0,
                    key="scatter_y",
                )
            fig = cost_performance_scatter(df, x_col=x_col, y_col=y_col, color_col=_GROUP_COL)
            st.plotly_chart(fig, use_container_width=True)

        if "input_len" in df.columns and available_metrics:
            st.subheader("Performance vs Input Complexity")
            fig = performance_vs_complexity(df, available_metrics, group_col=_GROUP_COL)
            st.plotly_chart(fig, use_container_width=True)

        _render_session_inspector(df)


def _render_session_inspector(df: pd.DataFrame) -> None:
    st.subheader("Session Inspector")

    has_text_cols = "input" in df.columns and "output" in df.columns
    if not has_text_cols:
        st.info("No input/output text columns available for inspection.")
        return

    display_cols = ["session_id", _GROUP_COL]
    for col in ["failure_mode", "language_label", "data_category_QA"] + [
        c for c in METRIC_COLS if c in df.columns
    ]:
        if col in df.columns and col not in display_cols:
            display_cols.append(col)

    st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True,
        key="session_table",
    )

    session_ids = df["session_id"].unique().tolist() if "session_id" in df.columns else []
    if session_ids:
        selected = st.selectbox("Select session to inspect", session_ids, key="inspect_session")
        row = df[df["session_id"] == selected].iloc[0]
        with st.expander(f"Session: {selected}", expanded=True):
            st.markdown("**Input**")
            st.text(str(row.get("input", "")))
            st.markdown("**Output**")
            st.text(str(row.get("output", "")))


if __name__ == "__main__":
    main()
