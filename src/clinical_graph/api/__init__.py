"""API routers for the clinical-graph microservice."""

from clinical_graph.api.routes_documents import router as documents_router
from clinical_graph.api.routes_query import router as query_router

__all__ = ["documents_router", "query_router"]
