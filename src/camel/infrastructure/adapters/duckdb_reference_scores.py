from __future__ import annotations

import logging
from collections import defaultdict

import duckdb

from camel.domain.value_objects.category_score_collection import CategoryScoreCollection

logger = logging.getLogger(__name__)

_SCORE_COLUMNS = (
    "token_overlap_f1",
    "refusal_detection",
    "class_exact_match",
    "groundedness",
    "pass_at_k",
    "hedging_detection",
    "question_response_overlap",
    "response_length_ratio",
    "rouge_l",
    "chunk_attribution",
    "self_consistency",
)

_QUERY = """
SELECT
    e.session_id,
    e.data_category_QA,
    e.token_overlap_f1,
    e.refusal_detection,
    e.class_exact_match,
    e.groundedness,
    e.pass_at_k,
    e.hedging_detection,
    e.question_response_overlap,
    e.response_length_ratio,
    e.rouge_l,
    e.chunk_attribution,
    e.self_consistency
FROM fct_evaluation_scores e
JOIN fct_inference_results i
    ON e.session_id = i.session_id AND e.run_id = i.run_id
WHERE i.model IN ({placeholders})
ORDER BY e.data_category_QA, e.session_id
"""


def load_reference_scores(
    db_path: str,
    models: list[str],
) -> list[CategoryScoreCollection]:
    """Load per-session scores from the DuckDB gold layer for the given models."""
    placeholders = ", ".join(["?" for _ in models])
    query = _QUERY.format(placeholders=placeholders)

    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute(query, models)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
    finally:
        conn.close()

    logger.info("Loaded %d scored rows for models %s", len(rows), models)

    cat_sessions: dict[str, list[str]] = defaultdict(list)
    cat_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        row_dict = dict(zip(columns, row))
        category = str(row_dict["data_category_QA"])
        session_id = str(row_dict["session_id"])

        cat_sessions[category].append(session_id)
        for col in _SCORE_COLUMNS:
            raw_val = row_dict.get(col)
            if raw_val is not None:
                if isinstance(raw_val, bool):
                    cat_scores[category][col].append(1.0 if raw_val else 0.0)
                else:
                    cat_scores[category][col].append(float(raw_val))

    collections: list[CategoryScoreCollection] = []
    for category in sorted(cat_sessions.keys()):
        scores_dict: dict[str, tuple[float, ...]] = {}
        n_sessions = len(cat_sessions[category])
        for metric, values in cat_scores[category].items():
            if len(values) == n_sessions:
                scores_dict[metric] = tuple(values)
            else:
                logger.warning(
                    "Metric '%s' in category '%s' has %d values but %d sessions — skipping",
                    metric,
                    category,
                    len(values),
                    n_sessions,
                )

        collections.append(
            CategoryScoreCollection(
                category=category,
                session_ids=tuple(cat_sessions[category]),
                scores=scores_dict,
            )
        )
        logger.info(
            "Category '%s': %d sessions, %d metrics",
            category,
            n_sessions,
            len(scores_dict),
        )

    return collections
