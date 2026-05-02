from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock

from camel.application.ports.agent_port import AgentPort
from camel.domain.entities.trace import Trace
from camel.domain.value_objects import TokenUsage
from camel.domain.value_objects.dataset_record import DatasetRecord

logger = logging.getLogger(__name__)


class CachedAgentAdapter:
    """Wraps an AgentPort and caches inference results to disk.

    Cache key is based on (record_id, model, prompt_hash). Each record
    can have multiple cached responses (for Pass@k). Responses are served
    in order; once all cached responses are exhausted, new calls go to
    the underlying agent.
    """

    def __init__(self, agent: AgentPort, cache_dir: str, model: str) -> None:
        self._agent = agent
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._counters: dict[str, int] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    def _cache_key(self, record_id: str, prompt_hash: str) -> str:
        return hashlib.sha256(
            f"{record_id}:{self._model}:{prompt_hash}".encode()
        ).hexdigest()[:16]

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.jsonl"

    def _load_cached_traces(self, key: str) -> list[dict[str, object]]:
        path = self._cache_path(key)
        if not path.exists():
            return []
        traces: list[dict[str, object]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces

    def _append_trace(self, key: str, trace: Trace) -> None:
        path = self._cache_path(key)
        entry = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "input_text": trace.input_text,
            "output_text": trace.output_text,
            "model": trace.model,
            "latency_ms": trace.latency_ms,
            "token_usage": {
                "input_tokens": trace.token_usage.input_tokens,
                "output_tokens": trace.token_usage.output_tokens,
                "total_tokens": trace.token_usage.total_tokens,
            },
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _trace_from_cache(self, data: dict[str, object], record: DatasetRecord) -> Trace:
        tu = data.get("token_usage", {})
        assert isinstance(tu, dict)
        return Trace(
            trace_id=str(data.get("trace_id", "")),
            session_id=record.id,
            input_text=str(data.get("input_text", "")),
            output_text=str(data.get("output_text", "")),
            model=str(data.get("model", self._model)),
            latency_ms=int(str(data.get("latency_ms", 0))),
            token_usage=TokenUsage(
                input_tokens=int(str(tu.get("input_tokens", 0))),
                output_tokens=int(str(tu.get("output_tokens", 0))),
                total_tokens=int(str(tu.get("total_tokens", 0))),
            ),
        )

    async def invoke(self, record: DatasetRecord, system_prompt: str) -> Trace:
        p_hash = self._prompt_hash(system_prompt)
        key = self._cache_key(record.id, p_hash)

        with self._lock:
            attempt = self._counters.get(key, 0)
            self._counters[key] = attempt + 1

        cached = self._load_cached_traces(key)
        if attempt < len(cached):
            self._hits += 1
            logger.debug("Cache HIT for record=%s attempt=%d", record.id, attempt)
            return self._trace_from_cache(cached[attempt], record)

        self._misses += 1
        logger.debug("Cache MISS for record=%s attempt=%d", record.id, attempt)
        trace = await self._agent.invoke(record, system_prompt)
        self._append_trace(key, trace)
        return trace

    async def flush(self) -> None:
        await self._agent.flush()
        logger.info(
            "Inference cache stats: %d hits, %d misses",
            self._hits,
            self._misses,
        )
