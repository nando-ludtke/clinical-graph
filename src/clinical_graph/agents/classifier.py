"""Document type classification and deterministic PII/PHI ingestion gate."""

from enum import Enum
import re

from pydantic import BaseModel, Field

from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.prompts import load_prompt
from clinical_graph.schemas.document import DocType, LocalRestriction
from clinical_graph.schemas.metrics import LLMUsage


class BlockedDocumentCategory(str, Enum):
    """Categories of documents rejected from the clinical knowledge graph."""

    PATIENT_RECORD = "patient_record"
    EHR_NOTE = "ehr_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_REPORT = "lab_report"
    PII_PHI_DETECTED = "pii_phi_detected"
    PROMOTIONAL_OR_UNVERIFIED = "promotional_or_unverified"
    NON_CLINICAL = "non_clinical"


ALLOWED_GRAPH_DOC_TYPES: set[DocType] = {
    DocType.GUIDELINE,
    DocType.DRUG_LABEL,
    DocType.PEER_REVIEWED,
    DocType.CLINICAL_TRIAL,
    DocType.INTERNAL_SOP,
}

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("SSN pattern detected", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "Medical Record Number (MRN) detected",
        re.compile(r"\b(?:MRN|Medical\s+Record\s+Number)\b(?:\s*[:#-]\s*[A-Z0-9-]+)?", re.IGNORECASE),
    ),
    (
        "Patient identifier (Patient Name / DOB) detected",
        re.compile(r"\b(?:Patient\s+Name|DOB)\s*:", re.IGNORECASE),
    ),
    (
        "Email address detected",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "Phone number detected",
        re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    ),
)

class DocumentClassification(BaseModel):
    """Structured classification decision for document ingestion screening."""

    is_allowed: bool
    predicted_doc_type: DocType | None = None
    blocked_category: BlockedDocumentCategory | None = None
    contains_pii_or_phi: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = ""
    inferred_conditions: list[str] = Field(default_factory=list)
    inferred_target_populations: list[str] = Field(default_factory=list)
    inferred_specialties: list[str] = Field(default_factory=list)
    inferred_local_restriction: LocalRestriction | None = None
    inferred_description: str | None = None


class DocumentClassifier:
    """Two-stage document classifier: deterministic PII/PHI regex gate followed by LLM classification."""

    def __init__(
        self,
        llm_client: LiteLLMClient | None = None,
        model: str = "gpt-4o-mini",
        prompt_version: str = "v1",
    ) -> None:
        self.llm_client: LiteLLMClient | None = llm_client
        self.model = model
        self.prompt_version = prompt_version
        self._system_prompt = load_prompt(
            "document_classifier", version=prompt_version
        ).system_prompt

    @staticmethod
    def screen_pii_regex(text: str) -> str | None:
        """Return a reason string if deterministic PII/PHI patterns match, else ``None``."""
        for reason, pattern in _PII_PATTERNS:
            if pattern.search(text):
                return reason
        return None

    async def classify_async(
        self,
        text: str,
        title: str = "",
        api_key: str | None = None,
    ) -> tuple[DocumentClassification, LLMUsage]:
        """Screen for PII/PHI and classify document eligibility before graph ingestion."""
        pii_reason = self.screen_pii_regex(f"{title}\n{text[:4000]}")
        if pii_reason is not None:
            return (
                DocumentClassification(
                    is_allowed=False,
                    blocked_category=BlockedDocumentCategory.PII_PHI_DETECTED,
                    contains_pii_or_phi=True,
                    confidence=1.0,
                    reason=pii_reason,
                ),
                LLMUsage(),
            )

        if self.llm_client is None:
            return (
                DocumentClassification(
                    is_allowed=True,
                    predicted_doc_type=DocType.GUIDELINE,
                    reason="Offline default allow",
                ),
                LLMUsage(),
            )

        snippet = f"Title: {title}\n\n{text[:2000]}" if title else text[:2000]
        raw, usage = await complete_with_usage_compat(
            self.llm_client,
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": snippet},
            ],
            temperature=0.0,
            response_format=DocumentClassification,
            api_key=api_key,
        )
        parsed = DocumentClassification.model_validate_json(raw)
        enforced_allowed = (
            parsed.is_allowed
            and not parsed.contains_pii_or_phi
            and parsed.predicted_doc_type in ALLOWED_GRAPH_DOC_TYPES
            and parsed.blocked_category is None
        )
        if not enforced_allowed and parsed.is_allowed:
            updates: dict[str, object] = {"is_allowed": False}
            if parsed.contains_pii_or_phi and parsed.blocked_category is None:
                updates["blocked_category"] = BlockedDocumentCategory.PII_PHI_DETECTED
            parsed = parsed.model_copy(update=updates)

        return parsed, usage
