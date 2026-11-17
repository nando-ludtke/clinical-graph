"""Pydantic v2 schemas for clinical query requests, filters, citations, and responses."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from clinical_graph.schemas.document import DocType, LocalRestriction
from clinical_graph.schemas.metrics import PipelineMetrics

RetrievalStrategy = Literal["auto", "rag", "hybrid"]
ResolvedRetrievalMode = Literal["rag", "hybrid"]


class ChatMessage(BaseModel):
    """Single conversational turn in a multi-turn clinical query."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class PatientContext(BaseModel):
    """Optional patient-specific clinical context for query scoping."""

    age_group: Literal["pediatric", "adult", "geriatric"] | None = None
    pregnancy_status: bool | None = None
    renal_impairment: bool | None = None
    allergies: list[str] = Field(default_factory=list)


class QueryFilters(BaseModel):
    """Optional retrieval-time filters restricting candidate documents."""

    document_ids: list[str] | None = None
    doc_types: list[DocType] | None = None
    local_restrictions: list[LocalRestriction] | None = None
    conditions: list[str] | None = None
    target_populations: list[str] | None = None


class QueryRequest(BaseModel):
    """Request payload for clinical hybrid retrieval and answer synthesis."""

    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(default_factory=list)
    k_hops: int = Field(default=2, ge=1, le=10)
    top_k_seeds: int = Field(default=5, ge=1, le=10)
    filters: QueryFilters | None = None
    patient_context: PatientContext | None = None
    allow_web_fallback: bool | None = None
    include_trace: bool = False
    retrieval_mode: RetrievalStrategy = "auto"


class CitationPassage(BaseModel):
    """Exact cited quote anchored within a source chunk with highlight offsets."""

    chunk_id: str | None = None
    quote: str
    chunk_text: str
    highlighted_chunk: str
    match_start: int | None = None
    match_end: int | None = None


class Citation(BaseModel):
    """Source document citation supporting a clinical answer."""

    doc_id: str
    slug: str
    title: str
    doc_type: DocType
    local_restriction: LocalRestriction
    excerpt: str | None = None
    excerpts: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    passages: list[CitationPassage] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    """Diagnostic trace of graph and document retrieval for a query."""

    search_queries: list[str] = Field(default_factory=list)
    retrieval_mode: Literal["focused", "multi_hop", "rag", "hybrid"] | None = None
    requested_mode: RetrievalStrategy | None = None
    chosen_mode: ResolvedRetrievalMode | None = None
    seed_nodes: list[str] = Field(default_factory=list)
    traversed_edges: list[dict[str, Any]] = Field(default_factory=list)
    matched_doc_ids: list[str] = Field(default_factory=list)
    assembled_context: str | None = None


class QueryResponse(BaseModel):
    """Response payload containing the synthesized answer, citations, and optional trace."""

    session_id: str
    answer: str
    chosen_mode: ResolvedRetrievalMode | None = None
    citations: list[Citation] = Field(default_factory=list)
    used_web_fallback: bool = False
    audit_passed: bool | None = None
    trace: RetrievalTrace | None = None
    metrics: PipelineMetrics | None = None
