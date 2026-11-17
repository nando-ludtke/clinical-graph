"""Hybrid clinical query endpoint combining metadata filtering, graph retrieval, and synthesis."""

from fastapi import APIRouter, Depends

from clinical_graph.api.deps import get_openai_api_key, get_query_pipeline
from clinical_graph.pipelines.query import QueryPipeline
from clinical_graph.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/v1/query", response_model=QueryResponse)
async def query_clinical_graph(
    payload: QueryRequest,
    pipeline: QueryPipeline = Depends(get_query_pipeline),
    api_key: str | None = Depends(get_openai_api_key),
) -> QueryResponse:
    """Execute filtered hybrid graph retrieval and synthesize a grounded clinical answer."""
    return await pipeline.run(payload, api_key=api_key)
