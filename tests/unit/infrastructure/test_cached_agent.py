from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock

import pytest

from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.infrastructure.adapters.cached_agent import CachedAgentAdapter


def _make_record(record_id: str = "rec-1") -> DatasetRecord:
    return DatasetRecord(
        id=record_id,
        question="What is the return policy?",
        content="Returns within 30 days.",
        context_metadata="{}",
        name="Alice",
        occupation="Agent",
        adjective="Friendly",
        chatbot_goal="Help users",
        instructions=["Be polite"],
        chunks_big=[],
        classes=[],
        chosen_class_id="P1",
        language=1,
        data_category_qa="positivo",
        content_base_uuids="uuid-1",
    )


def _make_trace(record_id: str = "rec-1", output: str = "response-1") -> Trace:
    return Trace(
        trace_id="trace-1",
        session_id=record_id,
        input_text="What is the return policy?",
        output_text=output,
        model="gpt-4o-mini",
        latency_ms=500,
        token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
    )


class TestCachedAgentAdapter:
    @pytest.fixture()
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir  # type: ignore[misc]

    @pytest.fixture()
    def mock_agent(self) -> AsyncMock:
        agent = AsyncMock()
        agent.invoke = AsyncMock(return_value=_make_trace())
        agent.flush = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_first_call_is_cache_miss(self, cache_dir: str, mock_agent: AsyncMock) -> None:
        adapter = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        record = _make_record()

        result = await adapter.invoke(record, "system prompt")

        assert result.output_text == "response-1"
        assert adapter.miss_count == 1
        assert adapter.hit_count == 0
        mock_agent.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_call_is_cache_hit(self, cache_dir: str, mock_agent: AsyncMock) -> None:
        adapter = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        record = _make_record()

        await adapter.invoke(record, "system prompt")

        adapter2 = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        result = await adapter2.invoke(record, "system prompt")

        assert result.output_text == "response-1"
        assert adapter2.hit_count == 1
        assert adapter2.miss_count == 0

    @pytest.mark.asyncio
    async def test_different_prompts_are_separate_keys(
        self, cache_dir: str, mock_agent: AsyncMock
    ) -> None:
        adapter = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        record = _make_record()

        await adapter.invoke(record, "prompt A")
        await adapter.invoke(record, "prompt B")

        assert adapter.miss_count == 2
        assert mock_agent.invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_pass_at_k_caches_multiple_responses(
        self, cache_dir: str, mock_agent: AsyncMock
    ) -> None:
        responses = [_make_trace(output=f"resp-{i}") for i in range(3)]
        mock_agent.invoke = AsyncMock(side_effect=responses)

        adapter = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        record = _make_record()
        prompt = "system prompt"

        r1 = await adapter.invoke(record, prompt)
        r2 = await adapter.invoke(record, prompt)
        r3 = await adapter.invoke(record, prompt)

        assert r1.output_text == "resp-0"
        assert r2.output_text == "resp-1"
        assert r3.output_text == "resp-2"
        assert adapter.miss_count == 3

        adapter2 = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        c1 = await adapter2.invoke(record, prompt)
        c2 = await adapter2.invoke(record, prompt)
        c3 = await adapter2.invoke(record, prompt)

        assert c1.output_text == "resp-0"
        assert c2.output_text == "resp-1"
        assert c3.output_text == "resp-2"
        assert adapter2.hit_count == 3
        assert adapter2.miss_count == 0

    @pytest.mark.asyncio
    async def test_flush_delegates_to_inner_agent(
        self, cache_dir: str, mock_agent: AsyncMock
    ) -> None:
        adapter = CachedAgentAdapter(agent=mock_agent, cache_dir=cache_dir, model="gpt-4o-mini")
        await adapter.flush()
        mock_agent.flush.assert_called_once()
