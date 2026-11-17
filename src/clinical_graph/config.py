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
        default=5,
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

    # Planned capability flags (not yet implemented)
    enable_guardrails: bool = False
    enable_web_fallback: bool = False
    enable_audit_agent: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
