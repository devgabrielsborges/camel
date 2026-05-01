from __future__ import annotations

import os

import pytest

from camel.domain.entities.evaluation import Evaluation
from camel.domain.value_objects.model_config import ModelConfig
from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

pytestmark = pytest.mark.skipif(
    not MLFLOW_TRACKING_URI,
    reason="MLFLOW_TRACKING_URI not set",
)


@pytest.fixture()
def tracker() -> MLflowTrackerAdapter:
    return MLflowTrackerAdapter(tracking_uri=MLFLOW_TRACKING_URI)


def _make_evaluation() -> Evaluation:
    return Evaluation(
        evaluation_id="test-eval-001",
        experiment_name="test_integration_experiment",
        eval_model=ModelConfig(model_name="gpt-4o-mini", temperature=0.0),
        prompt_version="prompts:/test/1",
        dataset_name="test_dataset",
    )


def test_start_and_end_run(tracker: MLflowTrackerAdapter) -> None:
    evaluation = _make_evaluation()
    run_id = tracker.start_run(evaluation)
    assert run_id
    tracker.end_run(run_id)


def test_log_metrics(tracker: MLflowTrackerAdapter) -> None:
    evaluation = _make_evaluation()
    run_id = tracker.start_run(evaluation)
    tracker.log_metrics(run_id, {"test_metric": 0.95})
    tracker.end_run(run_id)


def test_register_prompt(tracker: MLflowTrackerAdapter) -> None:
    template = PromptTemplate(
        template_path="prompts/system_prompt.j2",
        version_uri="prompts:/test/1",
        rendered_content="Test rendered content",
        token_count=10,
    )
    version_uri = tracker.register_prompt(template)
    assert version_uri.startswith("prompts:/")
