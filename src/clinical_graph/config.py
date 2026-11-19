"""Application configuration using Pydantic v2 BaseSettings."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the clinical-graph microservice."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLINICAL_GRAPH_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "clinical-graph"
    app_version: str = "0.1.0"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # Knowledge graph traversal depth
    graph_k_hops: int = Field(default=2, ge=1)

    # Per-client API rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    # Outbound LLM concurrency limit
    max_concurrent_llm_calls: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices(
            "MAX_CONCURRENT_LLM_CALLS",
            "CLINICAL_GRAPH_MAX_CONCURRENT_LLM_CALLS",
        ),
    )

    # Optional on-disk persistence for raw document text and knowledge graph
    document_storage_dir: str | None = "data/documents"
    graph_storage_dir: str | None = "data/graph"

    # LiteLLM model routing
    classifier_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "CLASSIFIER_MODEL",
            "CLINICAL_GRAPH_CLASSIFIER_MODEL",
        ),
    )
    query_rewrite_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "QUERY_REWRITE_MODEL",
            "CLINICAL_GRAPH_QUERY_REWRITE_MODEL",
        ),
    )
    node_extraction_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "NODE_EXTRACTION_MODEL",
            "CLINICAL_GRAPH_NODE_EXTRACTION_MODEL",
        ),
    )
    edge_extraction_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "EDGE_EXTRACTION_MODEL",
            "CLINICAL_GRAPH_EDGE_EXTRACTION_MODEL",
        ),
    )
    answer_generation_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices(
            "ANSWER_GENERATION_MODEL",
            "CLINICAL_GRAPH_ANSWER_GENERATION_MODEL",
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices(
            "EMBEDDING_MODEL",
            "CLINICAL_GRAPH_EMBEDDING_MODEL",
        ),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            "CLINICAL_GRAPH_OPENAI_API_KEY",
        ),
    )

    # Clinical-guard prompt-injection microservice
    clinical_guard_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices(
            "CLINICAL_GUARD_URL",
            "CLINICAL_GRAPH_CLINICAL_GUARD_URL",
        ),
    )
    clinical_guard_timeout_seconds: float = Field(
        default=2.5,
        gt=0.0,
        validation_alias=AliasChoices(
            "CLINICAL_GUARD_TIMEOUT_SECONDS",
            "CLINICAL_GRAPH_CLINICAL_GUARD_TIMEOUT_SECONDS",
        ),
    )
    clinical_guard_health_poll_seconds: int = Field(
        default=30,
        ge=0,
        validation_alias=AliasChoices(
            "CLINICAL_GUARD_HEALTH_POLL_SECONDS",
            "CLINICAL_GRAPH_CLINICAL_GUARD_HEALTH_POLL_SECONDS",
        ),
    )
    auto_prune_non_frozen_hours: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "AUTO_PRUNE_NON_FROZEN_HOURS",
            "CLINICAL_GRAPH_AUTO_PRUNE_NON_FROZEN_HOURS",
        ),
    )

    # Tavily Search API & 1-shot verified web fallback
    tavily_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TAVILY_API_KEY",
            "CLINICAL_GRAPH_TAVILY_API_KEY",
        ),
    )
    web_search_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "WEB_SEARCH_MODEL",
            "CLINICAL_GRAPH_WEB_SEARCH_MODEL",
        ),
    )
    decision_model: str = Field(
        default="gpt-6-luna",
        validation_alias=AliasChoices(
            "DECISION_MODEL",
            "CLINICAL_GRAPH_DECISION_MODEL",
        ),
    )
    enable_web_fallback: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ENABLE_WEB_FALLBACK",
            "CLINICAL_GRAPH_ENABLE_WEB_FALLBACK",
        ),
    )
    enable_web_pdf_fetch: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ENABLE_WEB_PDF_FETCH",
            "CLINICAL_GRAPH_ENABLE_WEB_PDF_FETCH",
        ),
    )
    include_regional_domains: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "INCLUDE_REGIONAL_DOMAINS",
            "CLINICAL_GRAPH_INCLUDE_REGIONAL_DOMAINS",
        ),
    )
    web_label_similarity_threshold: float = Field(
        default=0.48,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "WEB_LABEL_SIMILARITY_THRESHOLD",
            "CLINICAL_GRAPH_WEB_LABEL_SIMILARITY_THRESHOLD",
        ),
    )
    web_search_results_per_tier: int = Field(default=5, ge=1, le=10)
    web_search_results_per_tier_with_pdf: int = Field(default=3, ge=1, le=10)
    web_fetch_timeout_seconds: float = Field(default=8.0, gt=0.0)
    web_pdf_preview_chars: int = Field(default=3000, ge=500)
    web_pdf_max_pages: int = Field(default=40, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
