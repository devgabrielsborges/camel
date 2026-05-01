from __future__ import annotations

import csv
import logging
from pathlib import Path

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.score import Score

logger = logging.getLogger(__name__)

_SCORER_COLUMN_MAP: dict[str, str] = {
    "token_overlap_f1": "token_overlap_f1",
    "class_exact_match": "class_exact_match",
    "refusal_detection": "refusal_detection",
    "correctness": "correctness_score",
    "guidelines": "guidelines_score",
}

_CSV_COLUMNS: list[str] = [
    "id",
    "question",
    "prediction",
    "data_category_QA",
    "language",
    "model",
    "correctness_score",
    "guidelines_score",
    "token_overlap_f1",
    "class_exact_match",
    "refusal_detection",
]


def _score_value_str(score: Score) -> str:
    if score.value is None:
        return "N/A"
    return str(score.value)


class ExportResults:
    def execute(
        self,
        evaluation: Evaluation,
        output_path: str,
    ) -> int:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        row_count = 0
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()

            for session in evaluation.sessions:
                record = session.dataset_record
                for trace_obj in session.traces:
                    scores_by_name = {s.scorer_name: s for s in trace_obj.scores}
                    row: dict[str, str] = {
                        "id": record.id,
                        "question": record.question,
                        "prediction": trace_obj.output_text,
                        "data_category_QA": record.data_category_qa,
                        "language": str(record.language),
                        "model": trace_obj.model,
                    }
                    for scorer_name, col_name in _SCORER_COLUMN_MAP.items():
                        score = scores_by_name.get(scorer_name)
                        row[col_name] = _score_value_str(score) if score else ""

                    writer.writerow(row)
                    row_count += 1

        logger.info("Exported %d rows to %s", row_count, output_path)
        return row_count
