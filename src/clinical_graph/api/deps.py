"""FastAPI dependency helpers for retrieving application state services."""

from fastapi import Header, Request

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.classifier import DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.agents.query_rewriter import QueryRewriter
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.guardrails.prompt_guard import ClinicalGuardClient
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
    settings = request.app.state.settings
    guard_client = (
        ClinicalGuardClient(
            base_url=settings.clinical_guard_url,
            timeout=settings.clinical_guard_timeout_seconds,
        )
        if settings.clinical_guard_url
        else None
    )
    return QueryPipeline(
        manager=request.app.state.knowledge_manager,
        query_rewriter=request.app.state.query_rewriter,
        answer_agent=request.app.state.answer_agent,
        clinical_guard_client=guard_client,
    )


def get_openai_api_key(
    request: Request,
    x_openai_api_key: str | None = Header(default=None, alias="X-OpenAI-API-Key"),
) -> str | None:
    """Return per-request ``X-OpenAI-API-Key`` header or fallback to configured setting."""
    return x_openai_api_key or request.app.state.settings.openai_api_key


def extract_client_ip(request: Request) -> str | None:
    """Return the visitor IP from ``X-Forwarded-For`` (first hop) or ``request.client.host``."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    if request.client is not None and request.client.host:
        return request.client.host
    return None
