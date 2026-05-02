from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.score import Score

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

_SCORER_COLUMN_MAP: dict[str, str] = {
    "token_overlap_f1": "token_overlap_f1",
    "class_exact_match": "class_exact_match",
    "refusal_detection": "refusal_detection",
    "correctness": "correctness_score",
    "guidelines": "guidelines_score",
    "groundedness": "groundedness_score",
}


def _score_value(score: Score) -> float | bool | None:
    if score.value is None:
        return None
    return score.value


class ExportResults:
    def execute(
        self,
        evaluation: Evaluation,
        output_path: str,
        run_id: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> int:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        total_sessions = len(evaluation.sessions)
        row_count = 0
        session_count = 0
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        with open(output, "a", encoding="utf-8") as f:
            for session in evaluation.sessions:
                record = session.dataset_record
                primary_trace = session.traces[0] if session.traces else None
                if primary_trace is None:
                    continue

                pk_score = next(
                    (s for s in primary_trace.scores if s.scorer_name == "pass_at_k"),
                    None,
                )

                for trace_obj in session.traces:
                    scores_by_name = {s.scorer_name: s for s in trace_obj.scores}
                    row: dict[str, object] = {
                        "id": record.id,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "question": record.question,
                        "prediction": trace_obj.output_text,
                        "data_category_QA": record.data_category_qa,
                        "language": record.language,
                        "model": trace_obj.model,
                    }
                    for scorer_name, col_name in _SCORER_COLUMN_MAP.items():
                        score = scores_by_name.get(scorer_name)
                        row[col_name] = _score_value(score) if score else None

                    if pk_score is not None:
                        row["pass_at_k"] = _score_value(pk_score)
                        row["pass_at_k_best_score"] = pk_score.metadata.get("best_score")
                    else:
                        row["pass_at_k"] = None
                        row["pass_at_k_best_score"] = None

                    fm_entry = scores_by_name.get("failure_mode")
                    row["failure_mode"] = (
                        fm_entry.metadata.get("failure_mode") if fm_entry else None
                    )

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    row_count += 1

                session_count += 1
                if on_progress is not None:
                    on_progress(session_count, total_sessions)

        logger.info("Exported %d rows to %s", row_count, output_path)
        return row_count
