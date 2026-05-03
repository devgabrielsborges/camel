from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.genai

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)

_PATCHED = False


def _patch_livespan_set_tag() -> None:
    """Workaround for mlflow 3.11.x bug: _agent_tracer calls LiveSpan.set_tag
    but LiveSpan only exposes set_attribute. Alias set_tag to set_attribute."""
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return
    try:
        from mlflow.tracing.fluent import LiveSpan  # type: ignore[attr-defined]

        if not hasattr(LiveSpan, "set_tag"):
            LiveSpan.set_tag = LiveSpan.set_attribute  # type: ignore[attr-defined]
            logger.debug("Patched LiveSpan.set_tag -> set_attribute")
    except ImportError:
        pass
    _PATCHED = True


class MLflowTrackerAdapter:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)

    def enable_autolog(self) -> None:
        _patch_livespan_set_tag()
        mlflow.openai.autolog()
        self._register_litellm_callback()

    def disable_autolog(self) -> None:
        mlflow.openai.autolog(disable=True)
        self._unregister_litellm_callback()

    @staticmethod
    def _register_litellm_callback() -> None:
        import litellm
        from litellm.integrations.mlflow import MlflowLogger

        if not any(isinstance(cb, MlflowLogger) for cb in litellm.callbacks):
            litellm.callbacks.append(MlflowLogger())  # type: ignore[no-untyped-call]

    @staticmethod
    def _unregister_litellm_callback() -> None:
        import litellm
        from litellm.integrations.mlflow import MlflowLogger

        litellm.callbacks = [cb for cb in litellm.callbacks if not isinstance(cb, MlflowLogger)]

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
        prompt_name = Path(template.template_path).stem

        existing = self._load_latest_prompt(prompt_name)
        if existing is not None and existing.template == raw_content:
            logger.info(
                "Reusing existing prompt '%s' version %s (template unchanged)",
                existing.name,
                existing.version,
            )
            return f"prompts:/{existing.name}/{existing.version}"

        prompt_version = mlflow.genai.register_prompt(
            name=prompt_name,
            template=raw_content,
            commit_message=f"Register prompt from {template.template_path}",
            tags={"template_engine": "jinja2"},
        )
        return f"prompts:/{prompt_version.name}/{prompt_version.version}"

    @staticmethod
    def _load_latest_prompt(name: str) -> Any:
        try:
            return mlflow.genai.load_prompt(name)
        except Exception:
            return None

    def register_dataset(
        self,
        name: str,
        records: list[dict[str, object]],
    ) -> None:
        dataset = self._get_or_create_dataset(name)
        dataset.merge_records(records)
        logger.info("Registered dataset '%s' with %d records", name, len(records))

    @staticmethod
    def _get_or_create_dataset(name: str) -> Any:
        try:
            dataset = mlflow.genai.get_dataset(name=name)
            logger.info("Reusing existing dataset '%s'", name)
            return dataset
        except Exception:
            logger.info("Creating new dataset '%s'", name)
            return mlflow.genai.create_dataset(name=name)

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
