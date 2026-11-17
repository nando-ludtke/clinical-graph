"""Pydantic v2 schemas for clinical document ingestion, storage, and filtering."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DocType(str, Enum):
    """Supported clinical document types."""

    GUIDELINE = "guideline"
    DRUG_LABEL = "drug_label"
    PEER_REVIEWED = "peer_reviewed"
    CLINICAL_TRIAL = "clinical_trial"
    INTERNAL_SOP = "internal_sop"
    WEB_SNIPPET = "web_snippet"


class SourceType(str, Enum):
    """Origin format or ingestion channel of a clinical document."""

    UPLOADED_PDF = "uploaded_pdf"
    RAW_TEXT = "raw_text"
    MARKDOWN = "markdown"
    API_IMPORT = "api_import"
    WEB_SCRAPE = "web_scrape"


class LocalRestriction(str, Enum):
    """Jurisdictional or regional applicability restriction."""

    NONE = "none"
    US = "US"
    DE = "DE"
    EU = "EU"
    UK = "UK"
    CH = "CH"


class DocumentStatus(str, Enum):
    """Lifecycle status of a stored clinical document."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class DocumentMetadata(BaseModel):
    """Structured metadata attached to a clinical document."""

    title: str
    doc_type: DocType
    source_type: SourceType
    local_restriction: LocalRestriction = LocalRestriction.NONE
    publisher: str | None = None
    source_url: str | None = None
    language: str = "en"
    effective_year: int | None = None
    version: str | None = None
    conditions: list[str] = Field(default_factory=list)
    target_populations: list[str] = Field(default_factory=list)
    specialties: list[str] = Field(default_factory=list)
    description: str | None = None
    content_hash: str | None = None
    status: DocumentStatus = DocumentStatus.ACTIVE
    is_frozen: bool = False

    @field_validator("conditions", "target_populations", "specialties", mode="after")
    @classmethod
    def _normalize_string_lists(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]


class DocumentCreate(BaseModel):
    """Payload for creating/ingesting a clinical document."""

    content: str = Field(min_length=1)
    slug: str | None = None
    metadata: DocumentMetadata


class OmissionRecord(BaseModel):
    """Audit record describing a deterministic text span omitted before chunking."""

    omission_id: str
    reason: Literal[
        "references",
        "boilerplate_section",
        "citation_density",
        "repeated_header_footer",
    ]
    heading: str | None = None
    char_count: int
    preview: str


class ChunkSpan(BaseModel):
    """Character span of an extracted chunk within the cleaned document text."""

    chunk_id: str
    start_char: int
    end_char: int


class TableRecord(BaseModel):
    """Structured table extracted from a clinical document with surrounding context."""

    table_id: str
    doc_id: str
    markdown: str
    summary: str = ""
    context_before: str = ""
    context_after: str = ""
    page_number: int | None = None


class DocumentRecord(BaseModel):
    """Persisted clinical document record with generated identifiers and timestamps."""

    doc_id: str
    slug: str
    content: str
    metadata: DocumentMetadata
    storage_path: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_spans: list[ChunkSpan] = Field(default_factory=list)
    omissions: list[OmissionRecord] = Field(default_factory=list)
    tables: list[TableRecord] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class DocumentFilter(BaseModel):
    """Faceted AND filter criteria for querying stored documents."""

    doc_types: list[DocType] | None = None
    source_types: list[SourceType] | None = None
    local_restrictions: list[LocalRestriction] | None = None
    conditions: list[str] | None = None
    target_populations: list[str] | None = None
    specialties: list[str] | None = None
    status: DocumentStatus | None = DocumentStatus.ACTIVE


class DeletionReport(BaseModel):
    """Summary of coordinated document, graph, vector, and disk deletion."""

    doc_id: str
    deleted_from_db: bool
    deleted_from_disk: bool = False
    removed_nodes: int
    removed_edges: int
    removed_embeddings: int


class JobStatus(str, Enum):
    """Lifecycle status of an asynchronous background ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    """Status and result payload for a background document ingestion job."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    current_step: str | None = None
    filename: str | None = None
    title: str | None = None
    doc_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
