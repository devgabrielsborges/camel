from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

BOOL_COLS: list[str] = [
    "correctness",
    "guidelines",
    "class_exact_match",
    "refusal_detection",
    "pass_at_k",
    "hedging_detection",
]
FLOAT_COLS: list[str] = [
    "token_overlap_f1",
    "groundedness",
    "pass_at_k_best_score",
    "question_response_overlap",
    "response_length_ratio",
    "rouge_l",
    "chunk_attribution",
    "self_consistency",
    "self_consistency_variance",
]
METRIC_COLS: list[str] = BOOL_COLS + FLOAT_COLS
LANGUAGE_MAP: dict[int, str] = {1: "English", 2: "Spanish", 3: "Portuguese"}

_SCORES_TABLE = "fct_evaluation_scores"
_INFERENCE_TABLE = "fct_inference_results"


def _validate_db_path(db_path: str) -> Path:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    return path


@st.cache_data(ttl=300)
def load_scores(db_path: str) -> pd.DataFrame:
    _validate_db_path(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.sql(f"SELECT * FROM {_SCORES_TABLE}").fetchdf()  # noqa: S608
    finally:
        con.close()

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df["language_label"] = df["language"].map(LANGUAGE_MAP).fillna("Unknown")
    return df


@st.cache_data(ttl=300)
def load_inference_results(db_path: str) -> pd.DataFrame:
    _validate_db_path(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.sql(f"SELECT * FROM {_INFERENCE_TABLE}").fetchdf()  # noqa: S608
    finally:
        con.close()

    df["input_len"] = df["input"].fillna("").str.len()
    df["output_len"] = df["output"].fillna("").str.len()
    return df


@st.cache_data(ttl=300)
def load_merged_data(db_path: str) -> pd.DataFrame:
    scores = load_scores(db_path)
    inference = load_inference_results(db_path)

    merge_cols = ["session_id", "run_id"]
    right_cols = merge_cols + ["input", "output", "input_len", "output_len", "model"]
    available_right = [c for c in right_cols if c in inference.columns]

    merged = scores.merge(
        inference[available_right],
        on=merge_cols,
        how="left",
    )

    if "model" in merged.columns:
        merged["model_label"] = merged["model"].fillna(merged["run_id"])
    else:
        merged["model_label"] = merged["run_id"]

    return merged


def get_available_models(df: pd.DataFrame) -> list[str]:
    col = "model_label" if "model_label" in df.columns else "run_id"
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def get_filter_options(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "categories": (
            sorted(df["data_category_QA"].dropna().unique().tolist())
            if "data_category_QA" in df.columns
            else []
        ),
        "languages": (
            sorted(df["language_label"].dropna().unique().tolist())
            if "language_label" in df.columns
            else []
        ),
        "failure_modes": (
            sorted(df["failure_mode"].dropna().unique().tolist())
            if "failure_mode" in df.columns
            else []
        ),
        "models": get_available_models(df),
    }
