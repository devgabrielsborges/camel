from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from camel.infrastructure.dashboard.data_loader import FLOAT_COLS, get_filter_options


def render_sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    options = get_filter_options(df)

    st.sidebar.header("Filters")

    models = st.sidebar.multiselect(
        "Models",
        options=options["models"],
        default=[],
        key="filter_models",
    )

    categories = st.sidebar.multiselect(
        "Categories",
        options=options["categories"],
        default=[],
        key="filter_categories",
    )

    languages = st.sidebar.multiselect(
        "Languages",
        options=options["languages"],
        default=[],
        key="filter_languages",
    )

    failure_modes = st.sidebar.multiselect(
        "Failure modes",
        options=options["failure_modes"],
        default=[],
        key="filter_failure_modes",
    )

    score_ranges: dict[str, tuple[float, float]] = {}
    for metric in FLOAT_COLS:
        if metric not in df.columns:
            continue
        col_min = float(df[metric].min()) if df[metric].notna().any() else 0.0
        col_max = float(df[metric].max()) if df[metric].notna().any() else 1.0
        if col_min == col_max:
            continue
        score_ranges[metric] = st.sidebar.slider(
            metric.replace("_", " ").title(),
            min_value=col_min,
            max_value=col_max,
            value=(col_min, col_max),
            key=f"filter_range_{metric}",
        )

    return {
        "models": models,
        "categories": categories,
        "languages": languages,
        "failure_modes": failure_modes,
        "score_ranges": score_ranges,
    }


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if filters["models"]:
        model_col = "model_label" if "model_label" in df.columns else "run_id"
        mask &= df[model_col].isin(filters["models"])

    if filters["categories"] and "data_category_QA" in df.columns:
        mask &= df["data_category_QA"].isin(filters["categories"])

    if filters["languages"] and "language_label" in df.columns:
        mask &= df["language_label"].isin(filters["languages"])

    if filters["failure_modes"] and "failure_mode" in df.columns:
        mask &= df["failure_mode"].isin(filters["failure_modes"])

    for metric, (lo, hi) in filters.get("score_ranges", {}).items():
        if metric in df.columns:
            mask &= df[metric].isna() | ((df[metric] >= lo) & (df[metric] <= hi))

    return df[mask].copy()


def show_filter_summary(df_original: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    total = len(df_original)
    filtered = len(df_filtered)

    st.sidebar.divider()
    st.sidebar.metric("Rows", f"{filtered} / {total}")

    model_col = "model_label" if "model_label" in df_filtered.columns else "run_id"
    if model_col in df_filtered.columns:
        n_models = df_filtered[model_col].nunique()
        st.sidebar.metric("Models selected", str(n_models))
