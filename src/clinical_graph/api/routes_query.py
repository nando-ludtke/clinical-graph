"""Hybrid clinical query endpoint combining metadata filtering, graph retrieval, and synthesis."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from clinical_graph.api.deps import get_openai_api_key, get_query_pipeline
from clinical_graph.api.rate_limit import check_rate_limit
from clinical_graph.pipelines.query import QueryPipeline
from clinical_graph.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post(
    "/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def query_clinical_graph(
    payload: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    api_key: str | None = Depends(get_openai_api_key),
) -> QueryResponse:
    """Execute filtered hybrid graph retrieval and synthesize a grounded clinical answer."""
    return await pipeline.run(payload, api_key=api_key)


@router.post(
    "/v1/query/stream",
    dependencies=[Depends(check_rate_limit)],
)
async def stream_query_clinical_graph(
    payload: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    api_key: str | None = Depends(get_openai_api_key),
) -> StreamingResponse:
    """Stream Server-Sent Events (SSE) for each stage of the hybrid GraphRAG query."""
    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

    def on_event(event_name: str, data: dict[str, Any]) -> None:
        queue.put_nowait((event_name, json.dumps(data)))

    async def _run_pipeline() -> None:
        try:
            response = await pipeline.run(
                payload,
                api_key=api_key,
                on_event=on_event,
            )
            await queue.put(("done", response.model_dump_json()))
        except Exception as exc:
            await queue.put(("error", json.dumps({"detail": str(exc)})))
        finally:
            await queue.put(None)

    async def event_generator() -> AsyncIterator[str]:
        task = asyncio.create_task(_run_pipeline())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_name, data_json = item
                yield f"event: {event_name}\ndata: {data_json}\n\n"
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
