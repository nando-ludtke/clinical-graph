"""Pydantic v2 schemas for verified web fallback search, guarding, auditing, and selection."""

from typing import Literal

from pydantic import BaseModel, Field

from clinical_graph.schemas.metrics import LLMUsage


class WebSearchResult(BaseModel):
    """Single search result returned by Tavily."""

    result_id: str
    url: str
    title: str
    snippet: str = ""
    raw_content: str | None = None
    domain: str
    tier: Literal["dominant", "regional"] = "dominant"
    is_pdf: bool = False


class WebPageCacheEntry(BaseModel):
    """In-memory per-query store of fetched webpage/PDF full text and metadata."""

    result_id: str
    url: str
    title: str
    domain: str
    full_text: str
    content_type: Literal["html", "pdf"] = "html"


class CandidateLink(BaseModel):
    """Allowlisted subpage or PDF link discovered on a fetched landing page."""

    url: str
    anchor_text: str = ""
    is_pdf: bool = False


class AuditFinding(BaseModel):
    """Structured verdict and extracted clinical chunks produced by the Source Auditor."""

    result_id: str
    url: str
    title: str
    domain: str
    content_type: Literal["html", "pdf"] = "html"
    tier: Literal["dominant", "regional"] = "dominant"
    guard_passed: bool = True
    guard_reason: str | None = None
    readable: bool = True
    unreadable_reason: str | None = None
    relevant: bool = False
    reason: str = ""
    extracted_chunks: list[str] = Field(default_factory=list)
    candidate_links: list[CandidateLink] = Field(default_factory=list)
    followed_from_id: str | None = None


class WebEvidenceItem(BaseModel):
    """Noise-filtered web evidence block selected for AnswerAgent synthesis."""

    result_id: str
    url: str
    title: str
    domain: str
    content_type: Literal["html", "pdf"] = "html"
    tier: Literal["dominant", "regional"] = "dominant"
    chunks: list[str] = Field(default_factory=list)


class WebSearchOutcome(BaseModel):
    """Aggregate result of the Sufficiency + Web Search + Guard + Auditor workflow."""

    triggered: bool = False
    sufficient_locally: bool = True
    sufficiency_reason: str = ""
    searches_performed: int = 0
    queries_used: list[str] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    selected_evidence: list[WebEvidenceItem] = Field(default_factory=list)
    page_cache: dict[str, WebPageCacheEntry] = Field(default_factory=dict)
    guard_checked: bool = False
    guard_reachable: bool | None = None
    guard_blocked_count: int = 0
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    usage: LLMUsage = Field(default_factory=LLMUsage)
