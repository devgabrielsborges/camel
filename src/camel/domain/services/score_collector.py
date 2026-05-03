from __future__ import annotations

from collections import defaultdict

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.category_score_collection import CategoryScoreCollection

_STATISTICAL_METRICS = frozenset(
    {
        "token_overlap_f1",
        "refusal_detection",
        "class_exact_match",
        "groundedness",
        "pass_at_k",
    }
)


def collect_raw_scores(evaluation: Evaluation) -> list[CategoryScoreCollection]:
    """Build CategoryScoreCollection list from an evaluated Evaluation entity.

    Extracts per-session raw scores from traces, grouped by data_category_QA.
    Only includes metrics in the statistical metrics set.
    """
    cat_sessions: dict[str, list[str]] = defaultdict(list)
    cat_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for session in evaluation.sessions:
        category = session.dataset_record.data_category_qa
        session_id = session.session_id
        cat_sessions[category].append(session_id)

        trace = session.traces[0] if session.traces else None
        if trace is None:
            continue

        score_map = {s.scorer_name: s for s in trace.scores}
        for metric_name in _STATISTICAL_METRICS:
            score = score_map.get(metric_name)
            if score is not None and score.value is not None:
                val = (
                    1.0
                    if score.value is True
                    else (0.0 if score.value is False else float(score.value))
                )
                cat_scores[category][metric_name].append(val)

    collections: list[CategoryScoreCollection] = []
    for category in sorted(cat_sessions.keys()):
        session_ids = tuple(cat_sessions[category])
        n = len(session_ids)
        scores_dict: dict[str, tuple[float, ...]] = {}
        for metric, values in cat_scores[category].items():
            if len(values) == n:
                scores_dict[metric] = tuple(values)

        collections.append(
            CategoryScoreCollection(
                category=category,
                session_ids=session_ids,
                scores=scores_dict,
            )
        )

    return collections
