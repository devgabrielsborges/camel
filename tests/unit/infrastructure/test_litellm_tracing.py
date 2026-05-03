from __future__ import annotations

from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.integrations.mlflow import MlflowLogger

from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter


@pytest.fixture()
def tracker() -> MLflowTrackerAdapter:
    with patch("mlflow.set_tracking_uri"):
        return MLflowTrackerAdapter(tracking_uri="http://fake:5000")


@pytest.fixture(autouse=True)
def _clean_litellm_callbacks() -> None:
    """Ensure litellm.callbacks is clean before and after each test."""
    original = litellm.callbacks[:]
    litellm.callbacks.clear()
    yield  # type: ignore[misc]
    litellm.callbacks.clear()
    litellm.callbacks.extend(original)


class TestLiteLLMCallbackRegistration:
    def test_enable_autolog_registers_mlflow_logger(self, tracker: MLflowTrackerAdapter) -> None:
        with (
            patch("mlflow.openai.autolog"),
            patch("camel.infrastructure.adapters.mlflow_tracker._patch_livespan_set_tag"),
        ):
            tracker.enable_autolog()

        assert any(isinstance(cb, MlflowLogger) for cb in litellm.callbacks)

    def test_disable_autolog_removes_mlflow_logger(self, tracker: MLflowTrackerAdapter) -> None:
        litellm.callbacks.append(MlflowLogger())

        with patch("mlflow.openai.autolog"):
            tracker.disable_autolog()

        assert not any(isinstance(cb, MlflowLogger) for cb in litellm.callbacks)

    def test_enable_autolog_is_idempotent(self, tracker: MLflowTrackerAdapter) -> None:
        with (
            patch("mlflow.openai.autolog"),
            patch("camel.infrastructure.adapters.mlflow_tracker._patch_livespan_set_tag"),
        ):
            tracker.enable_autolog()
            tracker.enable_autolog()

        mlflow_loggers = [cb for cb in litellm.callbacks if isinstance(cb, MlflowLogger)]
        assert len(mlflow_loggers) == 1

    def test_enable_preserves_existing_non_mlflow_callbacks(
        self, tracker: MLflowTrackerAdapter
    ) -> None:
        other_callback = MagicMock()
        litellm.callbacks.append(other_callback)

        with (
            patch("mlflow.openai.autolog"),
            patch("camel.infrastructure.adapters.mlflow_tracker._patch_livespan_set_tag"),
        ):
            tracker.enable_autolog()

        assert other_callback in litellm.callbacks
        assert len(litellm.callbacks) == 2

    def test_disable_preserves_non_mlflow_callbacks(self, tracker: MLflowTrackerAdapter) -> None:
        other_callback = MagicMock()
        litellm.callbacks.append(other_callback)
        litellm.callbacks.append(MlflowLogger())

        with patch("mlflow.openai.autolog"):
            tracker.disable_autolog()

        assert other_callback in litellm.callbacks
        assert not any(isinstance(cb, MlflowLogger) for cb in litellm.callbacks)
