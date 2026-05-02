from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from camel.domain.value_objects.prompt_template import PromptTemplate
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter


@pytest.fixture()
def tracker() -> MLflowTrackerAdapter:
    with patch("mlflow.set_tracking_uri"):
        return MLflowTrackerAdapter(tracking_uri="http://fake:5000")


class TestRegisterPromptReuse:
    def test_reuses_existing_prompt_when_template_unchanged(
        self, tracker: MLflowTrackerAdapter, tmp_path: Path
    ) -> None:
        template_file = tmp_path / "system_prompt.j2"
        template_content = "You are a helpful assistant."
        template_file.write_text(template_content, encoding="utf-8")

        existing_prompt = SimpleNamespace(
            name="system_prompt",
            version=3,
            template=template_content,
        )

        tpl = PromptTemplate(template_path=str(template_file), version_uri="")

        with (
            patch.object(
                MLflowTrackerAdapter,
                "_load_latest_prompt",
                return_value=existing_prompt,
            ),
            patch("mlflow.genai.register_prompt") as mock_register,
        ):
            uri = tracker.register_prompt(tpl)

        assert uri == "prompts:/system_prompt/3"
        mock_register.assert_not_called()

    def test_registers_new_version_when_template_changed(
        self, tracker: MLflowTrackerAdapter, tmp_path: Path
    ) -> None:
        template_file = tmp_path / "system_prompt.j2"
        new_content = "You are a NEW assistant."
        template_file.write_text(new_content, encoding="utf-8")

        existing_prompt = SimpleNamespace(
            name="system_prompt",
            version=2,
            template="You are an OLD assistant.",
        )

        new_version = SimpleNamespace(name="system_prompt", version=3)

        tpl = PromptTemplate(template_path=str(template_file), version_uri="")

        with (
            patch.object(
                MLflowTrackerAdapter,
                "_load_latest_prompt",
                return_value=existing_prompt,
            ),
            patch("mlflow.genai.register_prompt", return_value=new_version) as mock_register,
        ):
            uri = tracker.register_prompt(tpl)

        assert uri == "prompts:/system_prompt/3"
        mock_register.assert_called_once()

    def test_registers_new_prompt_when_none_exists(
        self, tracker: MLflowTrackerAdapter, tmp_path: Path
    ) -> None:
        template_file = tmp_path / "my_prompt.j2"
        template_file.write_text("First version", encoding="utf-8")

        new_version = SimpleNamespace(name="my_prompt", version=1)
        tpl = PromptTemplate(template_path=str(template_file), version_uri="")

        with (
            patch.object(
                MLflowTrackerAdapter,
                "_load_latest_prompt",
                return_value=None,
            ),
            patch("mlflow.genai.register_prompt", return_value=new_version) as mock_register,
        ):
            uri = tracker.register_prompt(tpl)

        assert uri == "prompts:/my_prompt/1"
        mock_register.assert_called_once()


class TestRegisterDatasetReuse:
    def test_reuses_existing_dataset(self, tracker: MLflowTrackerAdapter) -> None:
        mock_dataset = MagicMock()

        with patch("mlflow.genai.get_dataset", return_value=mock_dataset) as mock_get:
            tracker.register_dataset("my_dataset", [{"inputs": {"q": "hi"}}])

        mock_get.assert_called_once_with(name="my_dataset")
        mock_dataset.merge_records.assert_called_once()

    def test_creates_new_dataset_when_not_found(self, tracker: MLflowTrackerAdapter) -> None:
        mock_dataset = MagicMock()

        with (
            patch("mlflow.genai.get_dataset", side_effect=Exception("Not found")),
            patch("mlflow.genai.create_dataset", return_value=mock_dataset) as mock_create,
        ):
            tracker.register_dataset("new_dataset", [{"inputs": {"q": "hi"}}])

        mock_create.assert_called_once_with(name="new_dataset")
        mock_dataset.merge_records.assert_called_once()
