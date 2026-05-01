from __future__ import annotations

from camel.domain.services.prompt_renderer import PromptRenderer, count_tokens
from camel.domain.value_objects.dataset_record import DatasetRecord


def test_count_tokens_returns_positive_int() -> None:
    assert count_tokens("Hello world") > 0


def test_render_includes_persona(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.rendered_content is not None
    assert "Alice" in result.rendered_content
    assert "friendly" in result.rendered_content.lower()
    assert "Customer support agent" in result.rendered_content


def test_render_includes_instructions(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.rendered_content is not None
    assert "Be polite" in result.rendered_content
    assert "Answer only based on the knowledge base" in result.rendered_content


def test_render_includes_tool_use_instruction(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.rendered_content is not None
    assert "search_knowledge_base" in result.rendered_content


def test_render_does_not_include_raw_chunks(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.rendered_content is not None
    assert "Returns are accepted within 30 days" not in result.rendered_content


def test_render_sets_token_count(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.token_count > 0


def test_render_sets_version_uri(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/weni/1")

    assert result.version_uri == "prompts:/weni/1"


def test_render_language_english(sample_dataset_record: DatasetRecord) -> None:
    renderer = PromptRenderer("prompts/system_prompt.j2")
    result = renderer.render(sample_dataset_record, version_uri="prompts:/test/1")

    assert result.rendered_content is not None
    assert "English" in result.rendered_content
