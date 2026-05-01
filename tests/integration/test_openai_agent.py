from __future__ import annotations

import os

import pytest

from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.infrastructure.adapters.openai_agent import OpenAIAgentAdapter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture()
def adapter() -> OpenAIAgentAdapter:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return OpenAIAgentAdapter(model=model)


@pytest.mark.asyncio
async def test_invoke_returns_trace(
    adapter: OpenAIAgentAdapter,
    sample_dataset_record: DatasetRecord,
) -> None:
    trace = await adapter.invoke(
        record=sample_dataset_record,
        system_prompt="You are a helpful assistant. Answer briefly.",
    )

    assert trace.trace_id
    assert trace.session_id == sample_dataset_record.id
    assert trace.output_text
    assert trace.token_usage.total_tokens > 0
    assert trace.latency_ms > 0
    assert trace.model


@pytest.mark.asyncio
async def test_flush_succeeds(adapter: OpenAIAgentAdapter) -> None:
    await adapter.flush()
