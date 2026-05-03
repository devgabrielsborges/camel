from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_PALETTE = px.colors.qualitative.Set2
CHART_TEMPLATE = "plotly_white"
DEFAULT_HEIGHT = 500
DEFAULT_MARGIN: dict[str, int] = {"l": 40, "r": 40, "t": 60, "b": 40}


def _apply_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=DEFAULT_HEIGHT,
        margin=DEFAULT_MARGIN,
    )
    return fig


def radar_chart(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "run_id",
) -> go.Figure:
    fig = go.Figure()
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return _apply_style(fig)

    for idx, (name, group) in enumerate(df.groupby(group_col)):
        means = [float(group[col].mean()) for col in available]
        means.append(means[0])
        theta = available + [available[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=means,
                theta=theta,
                fill="toself",
                name=str(name),
                line={"color": COLOR_PALETTE[idx % len(COLOR_PALETTE)]},
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        title="Metric Profile",
    )
    return _apply_style(fig)


def failure_mode_bars(
    df: pd.DataFrame,
    group_col: str = "run_id",
) -> go.Figure:
    if "failure_mode" not in df.columns:
        return _apply_style(go.Figure())

    counts = df.groupby([group_col, "failure_mode"], dropna=False).size().reset_index(name="count")
    counts["failure_mode"] = counts["failure_mode"].fillna("none")

    fig = px.bar(
        counts,
        x=group_col,
        y="count",
        color="failure_mode",
        barmode="stack",
        color_discrete_sequence=COLOR_PALETTE,
        title="Failure Mode Breakdown",
    )
    return _apply_style(fig)


def box_plots(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "run_id",
) -> go.Figure:
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return _apply_style(go.Figure())

    melted = df.melt(
        id_vars=[group_col],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )

    fig = px.box(
        melted,
        x="metric",
        y="value",
        color=group_col,
        color_discrete_sequence=COLOR_PALETTE,
        title="Score Distributions",
    )
    return _apply_style(fig)


def cost_performance_scatter(
    df: pd.DataFrame,
    x_col: str = "output_len",
    y_col: str = "groundedness",
    color_col: str = "run_id",
) -> go.Figure:
    if x_col not in df.columns or y_col not in df.columns:
        return _apply_style(go.Figure())

    hover_cols = [c for c in ["session_id", "failure_mode", "language_label"] if c in df.columns]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        hover_data=hover_cols or None,
        color_discrete_sequence=COLOR_PALETTE,
        title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
    )
    return _apply_style(fig)


def sankey_diagram(df: pd.DataFrame) -> go.Figure:
    required = {"data_category_QA", "refusal_detection", "failure_mode"}
    if not required.issubset(df.columns):
        return _apply_style(go.Figure())

    tmp = df.copy()
    refusal_col = tmp["refusal_detection"]
    tmp["refusal_label"] = refusal_col.apply(
        lambda v: "refused" if v in (True, 1, 1.0) else "not_refused"
    )
    tmp["failure_mode"] = tmp["failure_mode"].fillna("none")

    categories = sorted(tmp["data_category_QA"].unique())
    refusal_labels = sorted(tmp["refusal_label"].unique())
    failure_modes = sorted(tmp["failure_mode"].unique())

    all_labels = categories + refusal_labels + failure_modes
    label_idx = {label: i for i, label in enumerate(all_labels)}

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []

    cat_ref = tmp.groupby(["data_category_QA", "refusal_label"]).size().reset_index(name="count")
    for _, row in cat_ref.iterrows():
        sources.append(label_idx[row["data_category_QA"]])
        targets.append(label_idx[row["refusal_label"]])
        values.append(int(row["count"]))

    ref_fail = tmp.groupby(["refusal_label", "failure_mode"]).size().reset_index(name="count")
    for _, row in ref_fail.iterrows():
        sources.append(label_idx[row["refusal_label"]])
        targets.append(label_idx[row["failure_mode"]])
        values.append(int(row["count"]))

    fig = go.Figure(
        go.Sankey(
            node={
                "label": all_labels,
                "color": [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(all_labels))],
            },
            link={"source": sources, "target": targets, "value": values},
        )
    )
    fig.update_layout(title="Trajectory Flow: Category \u2192 Refusal \u2192 Failure Mode")
    return _apply_style(fig)


def performance_vs_complexity(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "run_id",
    n_bins: int = 5,
) -> go.Figure:
    if "input_len" not in df.columns:
        return _apply_style(go.Figure())

    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return _apply_style(go.Figure())

    tmp = df.copy()
    tmp["complexity_bin"] = pd.cut(tmp["input_len"], bins=n_bins)
    tmp["complexity_bin_label"] = tmp["complexity_bin"].astype(str)

    melted = tmp.melt(
        id_vars=[group_col, "complexity_bin_label", "complexity_bin"],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )

    agg = (
        melted.groupby([group_col, "complexity_bin_label", "metric"], observed=True)["value"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        agg,
        x="complexity_bin_label",
        y="value",
        color=group_col,
        facet_col="metric",
        color_discrete_sequence=COLOR_PALETTE,
        title="Performance vs Input Complexity",
        labels={"complexity_bin_label": "Input Length Bin", "value": "Mean Score"},
    )
    return _apply_style(fig)
