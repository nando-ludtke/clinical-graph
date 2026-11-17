"""FastAPI application entrypoint for clinical-graph."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from clinical_graph.agents import AnswerAgent, DocumentClassifier, QueryRewriter
from clinical_graph.api import (
    benchmarks_router,
    documents_router,
    graph_router,
    query_router,
    system_router,
)
from clinical_graph.api.rate_limit import SlidingWindowRateLimiter
from clinical_graph.config import Settings, get_settings
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    KnowledgeManager,
    LiteLLMEmbedder,
    LocalVectorStore,
)
from clinical_graph.llm import LiteLLMClient
from clinical_graph.pipelines.jobs import JobStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    app.state.settings = settings
    app.state.job_store = JobStore()
    app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    embedder = LiteLLMEmbedder(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = LocalVectorStore(embedder=embedder)
    graph = ClinicalKnowledgeGraph(vector_store=vector_store)
    app.state.knowledge_manager = KnowledgeManager(
        graph=graph,
        storage_dir=settings.document_storage_dir,
        graph_dir=settings.graph_storage_dir,
    )
    app.state.knowledge_manager.load()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    app.state.llm_client = llm_client
    app.state.answer_agent = AnswerAgent(
        llm_client=llm_client,
        model=settings.answer_generation_model,
    )
    app.state.query_rewriter = QueryRewriter(
        llm_client=llm_client,
        model=settings.query_rewrite_model,
    )
    app.state.document_classifier = DocumentClassifier(
        llm_client=llm_client,
        model=settings.classifier_model,
    )
    logger.info(
        "Starting %s v%s (environment=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    yield
    app.state.knowledge_manager.save()
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings: Settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    application.include_router(documents_router)
    application.include_router(query_router)
    application.include_router(graph_router)
    application.include_router(system_router)
    application.include_router(benchmarks_router)

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Return service health status and metadata."""
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
        }

    frontend_dir = Path("frontend")
    if not frontend_dir.is_dir():
        repo_frontend = Path(__file__).resolve().parents[2] / "frontend"
        if repo_frontend.is_dir():
            frontend_dir = repo_frontend
    if frontend_dir.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )

    return application


app = create_app()


if __name__ == "__main__":
    runtime_settings = get_settings()
    uvicorn.run(
        "clinical_graph.main:app",
        host="0.0.0.0",
        port=8000,
        reload=runtime_settings.environment == "dev",
    )
