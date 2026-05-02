from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Callable, Iterator

from camel.application.ports.agent_port import AgentPort
from camel.domain.entities.evaluation import Evaluation, EvaluationStatus
from camel.domain.entities.session import Session
from camel.domain.entities.trace import Trace
from camel.domain.services.prompt_renderer import PromptRenderer
from camel.domain.value_objects.dataset_record import DatasetRecord
from camel.infrastructure.adapters.duckdb_dataset import DuckDBDatasetAdapter
from camel.infrastructure.adapters.mlflow_tracker import MLflowTrackerAdapter

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def _chunked(iterator: Iterator[DatasetRecord], size: int) -> Iterator[list[DatasetRecord]]:
    batch: list[DatasetRecord] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class RunInference:
    def __init__(
        self,
        dataset_adapter: DuckDBDatasetAdapter,
        agent_adapter: AgentPort,
        tracker_adapter: MLflowTrackerAdapter,
        prompt_renderer: PromptRenderer,
        batch_size: int,
        concurrency: int,
        pass_at_k: int = 1,
    ) -> None:
        self._dataset = dataset_adapter
        self._agent = agent_adapter
        self._tracker = tracker_adapter
        self._renderer = prompt_renderer
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._pass_at_k = pass_at_k

    async def execute(
        self,
        evaluation: Evaluation,
        categories: list[str],
        limit: int | None = None,
        prompt_version_uri: str = "",
        on_progress: ProgressCallback | None = None,
        total: int | None = None,
    ) -> Evaluation:
        evaluation.transition_to(EvaluationStatus.INFERRING)

        effective_total = total or 0

        try:
            rows = self._dataset.load_filtered(categories)
            semaphore = asyncio.Semaphore(self._concurrency)
            processed = 0

            for batch in _chunked(rows, self._batch_size):
                tasks: list[tuple[DatasetRecord, list[asyncio.Task[Trace]]]] = []
                for record in batch:
                    if limit is not None and processed >= limit:
                        break

                    prompt_tpl = self._renderer.render(record, prompt_version_uri)
                    record_tasks: list[asyncio.Task[Trace]] = []

                    for _attempt in range(self._pass_at_k):

                        async def _invoke(
                            rec: DatasetRecord = record,
                            prompt: str = prompt_tpl.rendered_content or "",
                        ) -> Trace:
                            async with semaphore:
                                return await self._agent.invoke(rec, prompt)

                        record_tasks.append(asyncio.ensure_future(_invoke()))

                    tasks.append((record, record_tasks))
                    processed += 1

                if not tasks:
                    break

                all_coros = [t for _, rts in tasks for t in rts]
                await asyncio.gather(*all_coros, return_exceptions=True)

                for record, record_tasks in tasks:
                    session = Session(
                        session_id=record.id,
                        evaluation_id=evaluation.evaluation_id,
                        dataset_record=record,
                    )
                    for task in record_tasks:
                        if task.done() and not task.cancelled():
                            exc = task.exception()
                            if exc is not None:
                                logger.error("Inference failed for record: %s", exc)
                                continue
                            trace_obj = task.result()
                            assert isinstance(trace_obj, Trace)
                            session.add_trace(trace_obj)
                        else:
                            logger.error("Task not completed for record %s", record.id)

                    if session.traces:
                        evaluation.add_session(session)

                await self._agent.flush()
                gc.collect()

                if on_progress is not None:
                    on_progress(processed, effective_total)

                logger.info(
                    "Processed %d/%s records",
                    processed,
                    limit or "all",
                )

        except Exception:
            evaluation.transition_to(EvaluationStatus.FAILED)
            raise

        return evaluation
