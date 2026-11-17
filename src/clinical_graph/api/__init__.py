"""API routers for the clinical-graph microservice."""

from clinical_graph.api.routes_documents import router as documents_router
from clinical_graph.api.routes_graph import benchmarks_router, graph_router, system_router
from clinical_graph.api.routes_query import router as query_router

__all__ = [
    "benchmarks_router",
    "documents_router",
    "graph_router",
    "query_router",
    "system_router",
]
