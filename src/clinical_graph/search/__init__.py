"""Verified web fallback search, prompt-injection guarding, and Source Auditor."""

from clinical_graph.search.allowlist import (
    ALLOWED_DOMAINS,
    DOMINANT_DOMAINS,
    REGIONAL_DOMAINS,
    classify_domain_tier,
    is_url_allowed,
)
from clinical_graph.search.auditor import SourceAuditorAgent
from clinical_graph.search.decision_client import OpenAIDecisionClient, SufficiencyDecision
from clinical_graph.search.fetcher import FetchedPage, fetch_allowlisted_url, strip_html_to_text
from clinical_graph.search.schemas import (
    AuditFinding,
    CandidateLink,
    WebEvidenceItem,
    WebPageCacheEntry,
    WebSearchOutcome,
    WebSearchResult,
)
from clinical_graph.search.tavily_client import TavilySearchClient
from clinical_graph.search.web_search_agent import WebSearchAgent

__all__ = [
    "ALLOWED_DOMAINS",
    "AuditFinding",
    "CandidateLink",
    "DOMINANT_DOMAINS",
    "FetchedPage",
    "OpenAIDecisionClient",
    "REGIONAL_DOMAINS",
    "SourceAuditorAgent",
    "SufficiencyDecision",
    "TavilySearchClient",
    "WebEvidenceItem",
    "WebPageCacheEntry",
    "WebSearchAgent",
    "WebSearchOutcome",
    "WebSearchResult",
    "classify_domain_tier",
    "fetch_allowlisted_url",
    "is_url_allowed",
    "strip_html_to_text",
]
