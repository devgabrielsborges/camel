from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from agents import Agent, ModelResponse, Runner, flush_traces, trace

from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage, ToolCall
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.infrastructure.adapters.tools import build_knowledge_tool


def _aggregate_usage(raw_responses: Sequence[ModelResponse]) -> TokenUsage:
    total_in = 0
    total_out = 0
    total_all = 0
    for resp in raw_responses:
        if resp.usage is not None:
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            total_all += resp.usage.total_tokens
    return TokenUsage(
        input_tokens=total_in,
        output_tokens=total_out,
        total_tokens=total_all,
    )


def _extract_tool_calls(new_items: Sequence[object]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in new_items:
        if hasattr(item, "raw_item") and hasattr(item.raw_item, "type"):
            raw = item.raw_item
            if raw.type == "function_call":
                raw_args = getattr(raw, "arguments", "{}")
                try:
                    parsed_args: dict[str, Any] = json.loads(str(raw_args))
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {"raw": str(raw_args)}
                calls.append(
                    ToolCall(
                        tool_name=getattr(raw, "name", "unknown"),
                        arguments=parsed_args,
                        result=None,
                    )
                )
    return calls


class OpenAIAgentAdapter:
    def __init__(self, model: str) -> None:
        self._model = model

    async def invoke(self, record: DatasetRecord, system_prompt: str) -> Trace:
        kb_tool = build_knowledge_tool(record.chunks_big)

        agent = Agent(
            name="weni_eval_agent",
            instructions=system_prompt,
            model=self._model,
            tools=[kb_tool],
        )

        session_id = record.id
        t0 = time.perf_counter()

        with trace(
            workflow_name="weni_eval",
            group_id=session_id,
            metadata={
                "model": self._model,
                "data_category_QA": record.data_category_qa,
            },
        ) as current_trace:
            result = await Runner.run(agent, input=record.question)
            latency_ms = int((time.perf_counter() - t0) * 1000)

        return Trace(
            trace_id=current_trace.trace_id,
            session_id=session_id,
            input_text=record.question,
            output_text=str(result.final_output),
            tool_calls=_extract_tool_calls(result.new_items),
            token_usage=_aggregate_usage(result.raw_responses),
            model=self._model,
            latency_ms=latency_ms,
        )

    async def flush(self) -> None:
        flush_traces()
