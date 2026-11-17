"""Unit tests for Pydantic v2 request, response, and document schemas."""

import pytest
from pydantic import ValidationError

from clinical_graph.schemas import (
    ChatMessage,
    Citation,
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    PatientContext,
    QueryFilters,
    QueryRequest,
    QueryResponse,
    ResolvedRetrievalMode,
    RetrievalStrategy,
    RetrievalTrace,
    SourceType,
)


def test_query_request_defaults_and_valid_bounds() -> None:
    req = QueryRequest(query="First-line therapy for T2DM?", session_id="sess-1")
    assert req.k_hops == 2
    assert req.top_k_seeds == 5
    assert req.messages == []
    assert req.filters is None
    assert req.patient_context is None
    assert req.allow_web_fallback is None
    assert req.include_trace is False
    assert req.retrieval_mode == "auto"

    req_min = QueryRequest(query="q", session_id="s", k_hops=1, top_k_seeds=1)
    assert req_min.k_hops == 1
    assert req_min.top_k_seeds == 1

    req_max = QueryRequest(query="q" * 4000, session_id="s" * 128, k_hops=10, top_k_seeds=10)
    assert req_max.k_hops == 10
    assert req_max.top_k_seeds == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "", "session_id": "sess-1"},
        {"query": "a" * 4001, "session_id": "sess-1"},
        {"query": "valid", "session_id": ""},
        {"query": "valid", "session_id": "s" * 129},
        {"query": "valid", "session_id": "sess-1", "k_hops": 0},
        {"query": "valid", "session_id": "sess-1", "k_hops": 11},
        {"query": "valid", "session_id": "sess-1", "top_k_seeds": 0},
        {"query": "valid", "session_id": "sess-1", "top_k_seeds": 11},
        {"query": "valid", "session_id": "sess-1", "retrieval_mode": "graph"},
    ],
)
def test_query_request_invalid_bounds(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(**kwargs)


def test_chat_message_and_patient_context_validation() -> None:
    msg = ChatMessage(role="user", content="Check renal dosing")
    assert msg.role == "user"

    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="hi")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")

    ctx = PatientContext(age_group="pediatric", pregnancy_status=False, renal_impairment=True)
    assert ctx.age_group == "pediatric"

    with pytest.raises(ValidationError):
        PatientContext(age_group="elderly")  # type: ignore[arg-type]


def test_document_metadata_normalizes_string_lists() -> None:
    meta = DocumentMetadata(
        title="Pediatric Diabetes Guideline",
        doc_type=DocType.GUIDELINE,
        source_type=SourceType.MARKDOWN,
        conditions=["  Type 1 Diabetes ", "DKA", "   "],
        target_populations=[" Pediatric ", "Adolescents"],
        specialties=[" Endocrinology ", ""],
    )
    assert meta.conditions == ["type 1 diabetes", "dka"]
    assert meta.target_populations == ["pediatric", "adolescents"]
    assert meta.specialties == ["endocrinology"]


def test_document_create_requires_non_empty_content() -> None:
    meta = DocumentMetadata(
        title="Empty Doc",
        doc_type=DocType.INTERNAL_SOP,
        source_type=SourceType.RAW_TEXT,
    )
    with pytest.raises(ValidationError):
        DocumentCreate(content="", metadata=meta)


def test_query_response_with_citations_and_trace() -> None:
    resp = QueryResponse(
        session_id="sess-1",
        answer="Metformin is contraindicated in severe renal impairment.",
        citations=[
            Citation(
                doc_id="doc_1234567890ab",
                slug="metformin-label-2024",
                title="Metformin Label",
                doc_type=DocType.DRUG_LABEL,
                local_restriction=LocalRestriction.DE,
                excerpt="Contraindicated when eGFR < 30 mL/min.",
            )
        ],
        trace=RetrievalTrace(
            seed_nodes=["Metformin"],
            traversed_edges=[{"source": "Metformin", "target": "Renal Impairment", "rel": "contraindicated_in"}],
            matched_doc_ids=["doc_1234567890ab"],
        ),
    )
    assert len(resp.citations) == 1
    assert resp.trace is not None
    assert resp.trace.seed_nodes == ["Metformin"]
    assert QueryFilters(local_restrictions=[LocalRestriction.DE]).local_restrictions == [
        LocalRestriction.DE
    ]


def test_retrieval_mode_and_chosen_mode_schemas() -> None:
    strategy: RetrievalStrategy = "rag"
    resolved: ResolvedRetrievalMode = "rag"
    req = QueryRequest(query="Dose?", session_id="s1", retrieval_mode=strategy)
    assert req.retrieval_mode == "rag"

    req_hybrid = QueryRequest(query="Dose?", session_id="s1", retrieval_mode="hybrid")
    assert req_hybrid.retrieval_mode == "hybrid"

    resp = QueryResponse(
        session_id="s1",
        answer="500 mg",
        chosen_mode=resolved,
        trace=RetrievalTrace(requested_mode="auto", chosen_mode="hybrid"),
    )
    assert resp.chosen_mode == "rag"
    assert resp.trace is not None
    assert resp.trace.requested_mode == "auto"
    assert resp.trace.chosen_mode == "hybrid"

    with pytest.raises(ValidationError):
        QueryResponse(session_id="s1", answer="x", chosen_mode="auto")  # type: ignore[arg-type]
