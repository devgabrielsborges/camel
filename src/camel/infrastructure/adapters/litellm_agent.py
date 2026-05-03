from __future__ import annotations

import json
import time
import uuid
from typing import Any

import litellm

from camel.domain.entities.trace import Trace
from camel.domain.value_objects import Chunk, TokenUsage, ToolCall
from camel.domain.value_objects.dataset_record import DatasetRecord


def _build_knowledge_tool_spec() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the knowledge base for reference material. "
                "Call this tool before answering the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what information is needed.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def _build_knowledge_content(chunks: list[Chunk]) -> str:
    content_blocks = [chunk.content for chunk in chunks]
    return "\n\n---\n\n".join(content_blocks) if content_blocks else ""


class LiteLLMAgentAdapter:
    def __init__(self, model: str, api_base: str | None = None) -> None:
        self._model = model
        self._api_base = api_base

    async def invoke(self, record: DatasetRecord, system_prompt: str) -> Trace:
        knowledge_content = _build_knowledge_content(record.chunks_big)
        tool_spec = _build_knowledge_tool_spec()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.question},
        ]

        t0 = time.perf_counter()
        tool_calls_collected: list[ToolCall] = []
        total_input = 0
        total_output = 0

        max_iterations = 10
        for _ in range(max_iterations):
            response = await litellm.acompletion(
                model=self._model,
                messages=messages,
                tools=[tool_spec],
                tool_choice="auto",
                api_base=self._api_base,
                metadata={
                    "record_id": record.id,
                    "pipeline_stage": "inference",
                },
            )

            choice = response.choices[0]
            message = choice.message

            if response.usage:
                total_input += response.usage.prompt_tokens
                total_output += response.usage.completion_tokens

            if message.tool_calls:
                messages.append(message.model_dump())

                for tc in message.tool_calls:
                    fn = tc.function
                    raw_args = fn.arguments or "{}"
                    try:
                        parsed_args: dict[str, Any] = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"raw": raw_args}

                    if not knowledge_content:
                        result_text = "No knowledge base content available for this topic."
                    else:
                        result_text = knowledge_content

                    tool_calls_collected.append(
                        ToolCall(
                            tool_name=fn.name or "unknown",
                            arguments=parsed_args,
                            result=result_text,
                        )
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )
            else:
                break

        latency_ms = int((time.perf_counter() - t0) * 1000)
        final_output = message.content or ""

        return Trace(
            trace_id=str(uuid.uuid4()),
            session_id=record.id,
            input_text=record.question,
            output_text=final_output,
            tool_calls=tool_calls_collected,
            token_usage=TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            model=self._model,
            latency_ms=latency_ms,
        )

    async def flush(self) -> None:
        pass
