"""FastAPI dependency helpers for retrieving application state services."""

from fastapi import Header, Request

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.classifier import DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.agents.query_rewriter import QueryRewriter
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.pipelines.ingestion import IngestionPipeline
from clinical_graph.pipelines.jobs import JobStore
from clinical_graph.pipelines.query import QueryPipeline


def get_knowledge_manager(request: Request) -> KnowledgeManager:
    """Return the ``KnowledgeManager`` instance stored on ``app.state``."""
    return request.app.state.knowledge_manager


def get_job_store(request: Request) -> JobStore:
    """Return the ``JobStore`` instance stored on ``app.state``."""
    return request.app.state.job_store


def get_answer_agent(request: Request) -> AnswerAgent:
    """Return the ``AnswerAgent`` instance stored on ``app.state``."""
    return request.app.state.answer_agent


def get_query_rewriter(request: Request) -> QueryRewriter:
    """Return the ``QueryRewriter`` instance stored on ``app.state``."""
    return request.app.state.query_rewriter


def get_document_classifier(request: Request) -> DocumentClassifier:
    """Return the ``DocumentClassifier`` instance stored on ``app.state``."""
    return request.app.state.document_classifier


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Construct an ``IngestionPipeline`` wired to ``app.state`` services."""
    settings = request.app.state.settings
    llm_client = getattr(request.app.state, "llm_client", None) or LiteLLMClient(
        max_concurrent_calls=settings.max_concurrent_llm_calls
    )
    extractor = GraphExtractor(
        llm_client=llm_client,
        node_model=settings.node_extraction_model,
        edge_model=settings.edge_extraction_model,
    )
    return IngestionPipeline(
        manager=request.app.state.knowledge_manager,
        classifier=request.app.state.document_classifier,
        extractor=extractor,
    )


def get_query_pipeline(request: Request) -> QueryPipeline:
    """Construct a ``QueryPipeline`` wired to ``app.state`` services."""
    return QueryPipeline(
        manager=request.app.state.knowledge_manager,
        query_rewriter=request.app.state.query_rewriter,
        answer_agent=request.app.state.answer_agent,
    )


def get_openai_api_key(
    request: Request,
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-API-Key"),
) -> str | None:
    """Return per-request ``X-OpenAI-API-Key`` header or fallback to configured setting."""
    return x_openai_api_key or request.app.state.settings.openai_api_key
