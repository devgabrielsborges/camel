from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.genai

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)


class MLflowTrackerAdapter:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)

    def enable_autolog(self) -> None:
        mlflow.openai.autolog()

    def disable_autolog(self) -> None:
        mlflow.openai.autolog(disable=True)

    def start_run(self, evaluation: Evaluation) -> str:
        mlflow.set_experiment(evaluation.experiment_name)
        run = mlflow.start_run(
            run_name=evaluation.evaluation_id,
            tags={
                "model": evaluation.eval_model.model_name,
                "prompt_version": evaluation.prompt_version,
                "dataset": evaluation.dataset_name,
            },
        )
        return str(run.info.run_id)

    def end_run(self, run_id: str) -> None:
        mlflow.end_run()

    def set_run_tags(self, run_id: str, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def register_prompt(self, template: PromptTemplate) -> str:
        raw_content = Path(template.template_path).read_text(encoding="utf-8")
        prompt_version = mlflow.genai.register_prompt(
            name=Path(template.template_path).stem,
            template=raw_content,
            commit_message=f"Register prompt from {template.template_path}",
            tags={"template_engine": "jinja2"},
        )
        return f"prompts:/{prompt_version.name}/{prompt_version.version}"

    def register_dataset(
        self,
        name: str,
        records: list[dict[str, object]],
    ) -> None:
        dataset = mlflow.genai.create_dataset(name=name)
        dataset.merge_records(records)
        logger.info("Registered dataset '%s' with %d records", name, len(records))

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics, run_id=run_id)

    def search_traces(
        self,
        experiment_name: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        traces = mlflow.search_traces(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_id = '{run_id}'",
        )

        results: list[dict[str, Any]] = []
        for _, row in traces.iterrows():
            results.append(
                {
                    "trace_id": row.get("trace_id", ""),
                    "request": row.get("request", ""),
                    "response": row.get("response", ""),
                    "request_metadata": row.get("request_metadata", {}),
                    "tags": row.get("tags", {}),
                }
            )
        return results
