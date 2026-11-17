"""Direct unit tests for IngestionPipeline and QueryPipeline (no FastAPI/HTTP)."""

import json
from pathlib import Path

import pytest

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.classifier import DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.agents.query_rewriter import QueryRewriter
from clinical_graph.graph.knowledge_graph import ClinicalKnowledgeGraph
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.graph.vector_store import DeterministicHashEmbedder, LocalVectorStore
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.parser import chunk_text_with_spans
from clinical_graph.pipelines import (
    DocumentRejectedError,
    IngestionPipeline,
    IngestionResult,
    QueryPipeline,
)
from clinical_graph.schemas.document import (
    ChunkSpan,
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    SourceType,
    TableRecord,
)
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.schemas.query import QueryFilters, QueryRequest


def _make_manager(tmp_path: Path | None = None) -> KnowledgeManager:
    graph = ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder()))
    if tmp_path is None:
        return KnowledgeManager(graph=graph)
    return KnowledgeManager(
        graph=graph,
        storage_dir=tmp_path / "docs",
        graph_dir=tmp_path / "graph",
    )


class _QueuedLLMClient(LiteLLMClient):
    """Deterministic stub returning queued responses with token usage."""

    def __init__(self, responses: list[str], usage: LLMUsage | None = None) -> None:
        super().__init__()
        self._responses = list(responses)
        self._usage = usage or LLMUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost_usd=0.001
        )
        self.calls: list[dict] = []

    async def complete_with_usage(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> tuple[str, LLMUsage]:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("No more queued responses in _QueuedLLMClient.")
        return self._responses.pop(0), self._usage

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        text, _ = await self.complete_with_usage(model, messages, **kwargs)
        return text


@pytest.mark.anyio
async def test_ingestion_pipeline_rejects_disallowed_document_without_persisting(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "is_allowed": False,
                    "blocked_category": "non_clinical",
                    "reason": "Sports news article, not clinical guidance.",
                    "predicted_doc_type": None,
                }
            )
        ]
    )
    pipeline = IngestionPipeline(
        manager=manager,
        classifier=DocumentClassifier(llm_client=llm, model="gpt-4o-mini"),
    )
    payload = DocumentCreate(
        content="The championship match ended 2-1 in extra time.",
        metadata=DocumentMetadata(
            title="Match Report",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.RAW_TEXT,
        ),
    )

    with pytest.raises(DocumentRejectedError) as exc_info:
        await pipeline.ingest_document(payload, classify=True)

    assert exc_info.value.classification.is_allowed is False
    assert "non_clinical" in exc_info.value.detail
    assert manager.db.filter_documents() == []


@pytest.mark.anyio
async def test_ingestion_pipeline_backfills_metadata_and_overrides_doc_type(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "is_allowed": True,
                    "blocked_category": None,
                    "reason": "Clinical dosing protocol.",
                    "predicted_doc_type": "internal_sop",
                    "inferred_conditions": ["Sepsis", "sepsis"],
                    "inferred_target_populations": ["Adults"],
                    "inferred_specialties": ["Critical Care"],
                    "inferred_description": "Initial IV antibiotic protocol for sepsis.",
                    "inferred_local_restriction": "US",
                }
            )
        ]
    )
    pipeline = IngestionPipeline(
        manager=manager,
        classifier=DocumentClassifier(llm_client=llm, model="gpt-4o-mini"),
    )
    payload = DocumentCreate(
        content="Administer broad-spectrum IV antibiotics within 1 hour of sepsis recognition.",
        metadata=DocumentMetadata(
            title="Sepsis Protocol",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.MARKDOWN,
        ),
    )

    record = await pipeline.ingest_document(
        payload,
        classify=True,
        override_doc_type=True,
    )

    assert record.metadata.doc_type == DocType.INTERNAL_SOP
    assert record.metadata.conditions == ["sepsis"]
    assert record.metadata.target_populations == ["adults"]
    assert record.metadata.specialties == ["critical care"]
    assert record.metadata.description == "Initial IV antibiotic protocol for sepsis."
    assert record.metadata.local_restriction == LocalRestriction.US
    # Original payload must not be mutated in place
    assert payload.metadata.doc_type == DocType.GUIDELINE


@pytest.mark.anyio
async def test_ingestion_pipeline_file_bytes_with_extraction_and_deletion(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_name": "Metformin",
                            "node_type": "drug",
                            "aliases": ["Glucophage"],
                            "description": "First-line T2D agent.",
                        },
                        {
                            "canonical_name": "Type 2 Diabetes",
                            "node_type": "condition",
                            "aliases": ["T2D"],
                            "description": "Chronic metabolic disorder.",
                        },
                    ],
                    "edges": [
                        {
                            "source": "Metformin",
                            "target": "Type 2 Diabetes",
                            "relation": "TREATS",
                            "quote": "Metformin is first-line therapy for Type 2 Diabetes.",
                        }
                    ],
                }
            ),
        ]
    )
    pipeline = IngestionPipeline(
        manager=manager,
        extractor=GraphExtractor(llm_client=llm, node_model="gpt-4o-mini", edge_model="gpt-4o-mini"),
    )

    with pytest.raises(ValueError):
        await pipeline.ingest_file_bytes(b"data", "notes.docx", title="Invalid")

    with pytest.raises(ValueError, match="no extractable text"):
        await pipeline.ingest_file_bytes(b"   \n  ", "empty.txt", title="Empty")

    record = await pipeline.ingest_file_bytes(
        b"# Diabetes Guideline\n\nMetformin is first-line therapy for Type 2 Diabetes.",
        "diabetes.md",
        title="Diabetes Guideline",
        doc_type=DocType.GUIDELINE,
        conditions=["type 2 diabetes"],
        extract=True,
    )

    assert record.metadata.source_type == SourceType.MARKDOWN
    assert len(record.chunk_ids) == 1
    assert (tmp_path / "graph" / "documents.db").is_file()
    assert manager.graph.graph.number_of_nodes() >= 2
    assert manager.graph.graph.number_of_edges() >= 1

    report = await pipeline.delete_document(record.doc_id)
    assert report.deleted_from_db is True
    assert manager.db.get_by_id(record.doc_id) is None
    assert manager.graph.graph.number_of_nodes() == 0


@pytest.mark.anyio
async def test_query_pipeline_filters_merges_traverses_and_aggregates_metrics(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    doc_a = manager.ingest_document(
        DocumentCreate(
            content="Apixaban prevents stroke in atrial fibrillation.",
            metadata=DocumentMetadata(
                title="AFib Anticoagulation",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
                conditions=["atrial fibrillation"],
            ),
        )
    )
    manager.graph.upsert_node(
        node_id="apixaban",
        canonical_name="Apixaban",
        node_type="drug",
        aliases=["Eliquis"],
        doc_id=doc_a.doc_id,
    )
    manager.graph.upsert_node(
        node_id="afib",
        canonical_name="Atrial Fibrillation",
        node_type="condition",
        doc_id=doc_a.doc_id,
    )
    manager.graph.upsert_edge(
        source="apixaban",
        target="afib",
        relation="TREATS",
        doc_id=doc_a.doc_id,
        excerpt="Apixaban reduces stroke risk in AFib.",
    )
    doc_b = manager.ingest_document(
        DocumentCreate(
            content="Insulin treats diabetic ketoacidosis.",
            metadata=DocumentMetadata(
                title="DKA Protocol",
                doc_type=DocType.INTERNAL_SOP,
                source_type=SourceType.RAW_TEXT,
                conditions=["dka"],
            ),
        )
    )
    manager.graph.upsert_node(
        node_id="insulin",
        canonical_name="Insulin",
        node_type="drug",
        doc_id=doc_b.doc_id,
    )

    rewriter_llm = _QueuedLLMClient(
        [json.dumps({"search_queries": ["Apixaban atrial fibrillation", "Eliquis stroke"]})],
        usage=LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30, estimated_cost_usd=0.002),
    )
    answer_llm = _QueuedLLMClient(
        ["Apixaban is recommended for stroke prevention in atrial fibrillation."],
        usage=LLMUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65, estimated_cost_usd=0.005),
    )
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(llm_client=answer_llm, model="gpt-4o"),
    )

    response = await pipeline.run(
        QueryRequest(
            session_id="test-session",
            query="How is AFib treated?",
            filters=QueryFilters(doc_types=[DocType.GUIDELINE]),
            top_k_seeds=3,
            k_hops=1,
            include_trace=True,
        )
    )

    assert "Apixaban is recommended" in response.answer
    assert [c.doc_id for c in response.citations] == [doc_a.doc_id]
    assert response.citations[0].excerpt == "Apixaban reduces stroke risk in AFib."
    assert response.trace is not None
    assert response.trace.search_queries == ["Apixaban atrial fibrillation", "Eliquis stroke"]
    assert response.trace.matched_doc_ids == [doc_a.doc_id]
    assert response.trace.assembled_context is not None
    assert response.trace.assembled_context == answer_llm.calls[0]["messages"][1]["content"]
    assert "Query: How is AFib treated?" in response.trace.assembled_context
    assert response.metrics is not None
    assert set(response.metrics.latencies_ms.keys()) == {
        "filter_documents",
        "query_rewrite",
        "seed_search",
        "k_hop_traversal",
        "answer_synthesis",
        "total",
    }
    assert response.metrics.usage.prompt_tokens == 70
    assert response.metrics.usage.completion_tokens == 25
    assert response.metrics.usage.total_tokens == 95
    assert response.metrics.usage.estimated_cost_usd == pytest.approx(0.007)


def test_chunk_text_with_spans_offsets_and_chunk_id_on_edge_evidence() -> None:
    text = (
        "# Section A\n\n"
        "First paragraph about metformin dosing in renal impairment.\n\n"
        "# Section B\n\n"
        "Second paragraph about SGLT2 inhibitors and heart failure."
    )
    spans = chunk_text_with_spans(text, max_chars=90, overlap_paragraphs=1)
    assert len(spans) >= 2
    for chunk_str, start_char, end_char in spans:
        assert 0 <= start_char < end_char <= len(text)
        assert chunk_str.splitlines()[0] in text[start_char:end_char]
        assert chunk_str.splitlines()[-1] in text[start_char:end_char]


@pytest.mark.anyio
async def test_extraction_persists_chunk_spans_and_edge_chunk_id(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_name": "Metformin",
                            "node_type": "drug",
                            "aliases": [],
                            "description": "Biguanide antihyperglycemic.",
                        },
                        {
                            "canonical_name": "Lactic Acidosis",
                            "node_type": "condition",
                            "aliases": [],
                            "description": "Rare metabolic complication.",
                        },
                    ],
                    "edges": [
                        {
                            "source": "Metformin",
                            "target": "Lactic Acidosis",
                            "relation": "CAUSES",
                            "quote": "Metformin rarely causes lactic acidosis.",
                        }
                    ],
                }
            ),
        ]
    )
    pipeline = IngestionPipeline(
        manager=manager,
        extractor=GraphExtractor(llm_client=llm),
    )
    record = await pipeline.ingest_file_bytes(
        b"Metformin rarely causes lactic acidosis in severe renal failure.",
        "metformin.txt",
        title="Metformin Label",
        extract=True,
    )
    assert len(record.chunk_spans) == 1
    assert record.chunk_spans[0].chunk_id == f"{record.doc_id}:chunk:0"
    assert record.chunk_spans[0].start_char == 0

    edge_data = next(iter(manager.graph.graph.edges(data=True)))[2]["edge"]
    assert edge_data.evidence[0]["chunk_id"] == f"{record.doc_id}:chunk:0"


@pytest.mark.anyio
async def test_query_pipeline_focused_routing_ranking_seed_descriptions_and_passage_expansion(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    prefix_filler = " ".join(["background"] * 45)
    suffix_filler = " ".join(["followup"] * 45)
    doc_text = (
        f"{prefix_filler} "
        "Metformin initial dose is 500 mg twice daily with meals for type 2 diabetes. "
        f"{suffix_filler}\n\n"
        "Second paragraph discussing cardiovascular outcomes and renal monitoring."
    )
    doc = manager.ingest_document(
        DocumentCreate(
            content=doc_text,
            metadata=DocumentMetadata(
                title="T2D Dosing Guideline",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
            ),
        )
    )
    manager.record_processing_metadata(
        doc_id=doc.doc_id,
        chunk_ids=[f"{doc.doc_id}:chunk:0"],
        omissions=[],
        chunk_spans=[ChunkSpan(chunk_id=f"{doc.doc_id}:chunk:0", start_char=0, end_char=len(doc_text))],
    )
    manager.graph.vector_store.upsert(
        f"{doc.doc_id}:chunk:0",
        doc_text,
        entry_type="chunk",
        doc_id=doc.doc_id,
    )

    manager.graph.upsert_node(
        node_id="metformin",
        canonical_name="Metformin",
        node_type="drug",
        description="First-line oral biguanide for glycemic control.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="t2d",
        canonical_name="Type 2 Diabetes",
        node_type="condition",
        description="Chronic insulin resistance disorder.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="ckd",
        canonical_name="Chronic Kidney Disease",
        node_type="condition",
        description="Impaired renal function.",
        doc_id=doc.doc_id,
    )
    # 1-hop edge with direct query overlap + chunk_id
    manager.graph.upsert_edge(
        source="metformin",
        target="t2d",
        relation="TREATS",
        doc_id=doc.doc_id,
        excerpt="Metformin initial dose is 500 mg twice daily with meals for type 2 diabetes.",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )
    # 2-hop edge (should be excluded when retrieval_mode == "focused" caps k_hops to 1)
    manager.graph.upsert_edge(
        source="t2d",
        target="ckd",
        relation="RISK_FACTOR_FOR",
        doc_id=doc.doc_id,
        excerpt="Type 2 diabetes is a risk factor for chronic kidney disease.",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )

    rewriter_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": ["Metformin initial dose"],
                    "retrieval_mode": "hybrid",
                }
            )
        ]
    )
    answer_llm = _QueuedLLMClient(["Start metformin at 500 mg twice daily with meals."])
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(llm_client=answer_llm, model="gpt-4o"),
    )

    response = await pipeline.run(
        QueryRequest(
            session_id="focused-session",
            query="What is the initial dose of metformin?",
            top_k_seeds=1,
            k_hops=1,
            include_trace=True,
        )
    )

    assert response.chosen_mode == "hybrid"
    assert response.trace is not None
    assert response.trace.chosen_mode == "hybrid"
    # Only the 1-hop edge should be traversed when k_hops=1
    assert len(response.trace.traversed_edges) == 1
    assert response.trace.traversed_edges[0]["target"] == "t2d"

    user_prompt = answer_llm.calls[0]["messages"][1]["content"]
    assert "First-line oral biguanide for glycemic control." in user_prompt
    assert "Retrieved chunks:" in user_prompt
    assert "500 mg twice daily with meals" in user_prompt
    assert f"{doc.doc_id}:chunk:0" in user_prompt


@pytest.mark.anyio
async def test_edge_ranking_and_bm25_passage_fallback_and_answer_agent_sections(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    doc_text = (
        "Paragraph one discusses general hypertension lifestyle advice.\n\n"
        "Paragraph two states that apixaban requires renal dose adjustment when creatinine is elevated."
    )
    doc = manager.ingest_document(
        DocumentCreate(
            content=doc_text,
            metadata=DocumentMetadata(
                title="Anticoagulation SOP",
                doc_type=DocType.INTERNAL_SOP,
                source_type=SourceType.RAW_TEXT,
            ),
        )
    )
    manager.record_processing_metadata(
        doc_id=doc.doc_id,
        chunk_ids=[f"{doc.doc_id}:chunk:0"],
        omissions=[],
        chunk_spans=[ChunkSpan(chunk_id=f"{doc.doc_id}:chunk:0", start_char=0, end_char=len(doc_text))],
    )
    manager.graph.vector_store.upsert(
        f"{doc.doc_id}:chunk:0",
        doc_text,
        entry_type="chunk",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="apixaban",
        canonical_name="Apixaban",
        node_type="drug",
        description="Direct factor Xa inhibitor.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="hypertension",
        canonical_name="Hypertension",
        node_type="condition",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="renal-impairment",
        canonical_name="Renal Impairment",
        node_type="condition",
        doc_id=doc.doc_id,
    )
    # Lower-overlap edge
    manager.graph.upsert_edge(
        source="apixaban",
        target="hypertension",
        relation="INTERACTS_WITH",
        doc_id=doc.doc_id,
        excerpt="Monitor blood pressure.",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )
    # Higher-overlap edge whose excerpt is paraphrased (triggers paragraph BM25 fallback)
    manager.graph.upsert_edge(
        source="apixaban",
        target="renal-impairment",
        relation="REQUIRES_MONITORING",
        doc_id=doc.doc_id,
        excerpt="Apixaban renal dose adjustment elevated creatinine",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )

    rewriter_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": ["Apixaban renal dose adjustment creatinine"],
                    "retrieval_mode": "hybrid",
                }
            )
        ]
    )
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(),  # offline deterministic mode
    )
    response = await pipeline.run(
        QueryRequest(
            session_id="ranking-session",
            query="Does apixaban need renal dose adjustment for elevated creatinine?",
            top_k_seeds=1,
            k_hops=1,
            include_trace=True,
            retrieval_mode="hybrid",
        )
    )

    assert response.chosen_mode == "hybrid"
    assert response.trace is not None
    assert response.trace.retrieval_mode == "hybrid"
    assert response.trace.requested_mode == "hybrid"
    assert response.trace.chosen_mode == "hybrid"
    # Higher-overlap edge is ranked first and used for Citation.excerpt
    assert response.trace.traversed_edges[0]["target"] == "renal-impairment"
    assert response.citations[0].excerpt == "Apixaban renal dose adjustment elevated creatinine"
    # Offline AnswerAgent includes seed descriptions, ranked edges, and retrieved chunks
    assert "Direct factor Xa inhibitor." in response.answer
    assert "Retrieved chunks:" in response.answer
    assert "creatinine is elevated" in response.answer


@pytest.mark.anyio
async def test_ingestion_pipeline_file_bytes_with_report_aggregates_metrics_and_usage(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    classifier_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "is_allowed": True,
                    "blocked_category": None,
                    "reason": "Clinical guideline.",
                    "predicted_doc_type": "guideline",
                    "inferred_conditions": ["Type 2 Diabetes"],
                    "inferred_target_populations": ["Adults"],
                    "inferred_specialties": ["Endocrinology"],
                    "inferred_description": "Metformin guideline.",
                }
            )
        ],
        usage=LLMUsage(
            prompt_tokens=15,
            completion_tokens=10,
            total_tokens=25,
            estimated_cost_usd=0.001,
        ),
    )
    extractor_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_name": "Metformin",
                            "node_type": "drug",
                            "aliases": [],
                            "description": "First-line T2D agent.",
                        },
                        {
                            "canonical_name": "Type 2 Diabetes",
                            "node_type": "condition",
                            "aliases": [],
                            "description": "Metabolic disorder.",
                        },
                    ],
                    "edges": [
                        {
                            "source": "Metformin",
                            "target": "Type 2 Diabetes",
                            "relation": "TREATS",
                            "quote": "Metformin treats Type 2 Diabetes.",
                        }
                    ],
                }
            ),
        ],
        usage=LLMUsage(
            prompt_tokens=30,
            completion_tokens=20,
            total_tokens=50,
            estimated_cost_usd=0.002,
        ),
    )
    pipeline = IngestionPipeline(
        manager=manager,
        classifier=DocumentClassifier(llm_client=classifier_llm, model="gpt-4o-mini"),
        extractor=GraphExtractor(
            llm_client=extractor_llm,
            node_model="gpt-4o-mini",
            edge_model="gpt-4o-mini",
        ),
    )

    result = await pipeline.ingest_file_bytes_with_report(
        b"Metformin treats Type 2 Diabetes.",
        "t2d.txt",
        title="T2D Guideline",
        classify=True,
        extract=True,
    )

    assert isinstance(result, IngestionResult)
    assert result.record.metadata.doc_type == DocType.GUIDELINE
    assert result.extracted_node_count == 2
    assert result.extracted_edge_count == 1
    assert set(result.metrics.latencies_ms.keys()) == {
        "parse_ms",
        "classify_ms",
        "extract_ms",
        "ingest_ms",
        "total_ms",
    }
    assert all(v >= 0.0 for v in result.metrics.latencies_ms.values())
    # 1 classifier call (15/10/25/0.001) + 1 single-pass extractor call (30/20/50/0.002)
    assert result.metrics.usage.prompt_tokens == 45
    assert result.metrics.usage.completion_tokens == 30
    assert result.metrics.usage.total_tokens == 75
    assert result.metrics.usage.estimated_cost_usd == pytest.approx(0.003)


@pytest.mark.anyio
async def test_query_pipeline_retrieval_mode_auto_and_forced_routing(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    doc_text = (
        "Metformin initial dosing is 500 mg twice daily with meals.\n\n"
        "Type 2 diabetes increases risk for chronic kidney disease."
    )
    doc = manager.ingest_document(
        DocumentCreate(
            content=doc_text,
            metadata=DocumentMetadata(
                title="T2D Clinical Guide",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
            ),
        )
    )
    # Intentionally leave chunk_spans empty to also exercise doc.content BM25 fallback in rag mode.
    manager.graph.upsert_node(
        node_id="metformin",
        canonical_name="Metformin",
        node_type="drug",
        description="First-line oral biguanide.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="t2d",
        canonical_name="Type 2 Diabetes",
        node_type="condition",
        description="Chronic metabolic disorder.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="ckd",
        canonical_name="Chronic Kidney Disease",
        node_type="condition",
        description="Impaired renal function.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_edge(
        source="metformin",
        target="t2d",
        relation="TREATS",
        doc_id=doc.doc_id,
        excerpt="Metformin treats type 2 diabetes",
    )
    manager.graph.upsert_edge(
        source="t2d",
        target="ckd",
        relation="RISK_FACTOR_FOR",
        doc_id=doc.doc_id,
        excerpt="Type 2 diabetes increases risk for chronic kidney disease",
    )

    # Record chunk span and upsert chunk entry into vector_store so rag mode can retrieve it
    manager.record_processing_metadata(
        doc_id=doc.doc_id,
        chunk_ids=[f"{doc.doc_id}:chunk:0"],
        omissions=[],
        chunk_spans=[ChunkSpan(chunk_id=f"{doc.doc_id}:chunk:0", start_char=0, end_char=len(doc_text))],
    )
    manager.graph.vector_store.upsert(
        f"{doc.doc_id}:chunk:0",
        doc_text,
        entry_type="chunk",
        doc_id=doc.doc_id,
    )

    # 1. auto + rag -> rag (include_trace=False still surfaces chosen_mode)
    steps_auto_rag: list[str] = []
    pipeline_auto_rag = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(
            llm_client=_QueuedLLMClient(
                [json.dumps({"search_queries": ["Metformin initial dosing"], "retrieval_mode": "rag"})]
            ),
            model="gpt-4o-mini",
        ),
        answer_agent=AnswerAgent(),
    )
    resp_auto_rag = await pipeline_auto_rag.run(
        QueryRequest(
            session_id="s-auto-rag",
            query="What is the initial dosing of metformin?",
            top_k_seeds=1,
            k_hops=3,
            include_trace=False,
            retrieval_mode="auto",
        ),
        on_step=steps_auto_rag.append,
    )
    assert resp_auto_rag.chosen_mode == "rag"
    assert resp_auto_rag.trace is None
    assert "Retrieved chunks:" in resp_auto_rag.answer
    assert "500 mg twice daily" in resp_auto_rag.answer
    assert any("mode=rag" in s for s in steps_auto_rag)

    # 2. auto + hybrid -> hybrid (traverses k_hops + retrieves chunks)
    steps_auto_hybrid: list[str] = []
    pipeline_auto_hybrid = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(
            llm_client=_QueuedLLMClient(
                [json.dumps({"search_queries": ["Metformin T2D CKD"], "retrieval_mode": "hybrid"})]
            ),
            model="gpt-4o-mini",
        ),
        answer_agent=AnswerAgent(),
    )
    resp_auto_hybrid = await pipeline_auto_hybrid.run(
        QueryRequest(
            session_id="s-auto-hybrid",
            query="How does metformin relate to type 2 diabetes and chronic kidney disease?",
            top_k_seeds=1,
            k_hops=2,
            include_trace=True,
            retrieval_mode="auto",
        ),
        on_step=steps_auto_hybrid.append,
    )
    assert resp_auto_hybrid.chosen_mode == "hybrid"
    assert resp_auto_hybrid.trace is not None
    assert resp_auto_hybrid.trace.chosen_mode == "hybrid"
    assert resp_auto_hybrid.trace.requested_mode == "auto"
    assert len(resp_auto_hybrid.trace.traversed_edges) == 2
    assert any("mode=hybrid" in s for s in steps_auto_hybrid)

    # 3. Forced rag overrides rewriter hybrid and skips graph traversal
    pipeline_forced_rag = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(
            llm_client=_QueuedLLMClient(
                [json.dumps({"search_queries": ["Metformin"], "retrieval_mode": "hybrid"})]
            ),
            model="gpt-4o-mini",
        ),
        answer_agent=AnswerAgent(),
    )
    resp_forced_rag = await pipeline_forced_rag.run(
        QueryRequest(
            session_id="s-forced-rag",
            query="Metformin initial dosing?",
            top_k_seeds=1,
            k_hops=3,
            include_trace=True,
            retrieval_mode="rag",
        )
    )
    assert resp_forced_rag.chosen_mode == "rag"
    assert resp_forced_rag.trace is not None
    assert resp_forced_rag.trace.requested_mode == "rag"
    assert resp_forced_rag.trace.chosen_mode == "rag"
    assert len(resp_forced_rag.trace.traversed_edges) == 0
    assert "Retrieved chunks:" in resp_forced_rag.answer

    # 4. Forced hybrid overrides rewriter rag, traverses 2-hop
    pipeline_forced_hybrid = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(
            llm_client=_QueuedLLMClient(
                [json.dumps({"search_queries": ["Metformin"], "retrieval_mode": "rag"})]
            ),
            model="gpt-4o-mini",
        ),
        answer_agent=AnswerAgent(),
    )
    resp_forced_hybrid = await pipeline_forced_hybrid.run(
        QueryRequest(
            session_id="s-forced-hybrid",
            query="Metformin?",
            top_k_seeds=1,
            k_hops=2,
            include_trace=True,
            retrieval_mode="hybrid",
        )
    )
    assert resp_forced_hybrid.chosen_mode == "hybrid"
    assert resp_forced_hybrid.trace is not None
    assert resp_forced_hybrid.trace.requested_mode == "hybrid"
    assert resp_forced_hybrid.trace.chosen_mode == "hybrid"
    assert len(resp_forced_hybrid.trace.traversed_edges) == 2


@pytest.mark.anyio
async def test_single_newline_pdf_chunk_produces_compact_supporting_passages(
    tmp_path: Path,
) -> None:
    from clinical_graph.pipelines.query import _locate_excerpt_or_bm25_paragraph

    filler_line = "Routine outpatient follow-up documentation without specific dosing guidance.\n"
    target_lines = (
        "Metformin should be initiated at 500 mg once daily with the evening meal.\n"
        "Assess renal function eGFR before starting metformin therapy.\n"
    )
    # Single-newline PDF-style chunk > 3,000 chars with no blank-line paragraph breaks
    pdf_chunk = (filler_line * 25) + target_lines + (filler_line * 25)
    assert len(pdf_chunk) > 3000
    assert "\n\n" not in pdf_chunk

    # Verify hyphenated line-break excerpt matching keeps clean chunk offsets
    hyph_chunk = "Die Aufklärungs-\ninitiative empfiehlt regelmäßige Kontrollen."
    h_start, h_end = _locate_excerpt_or_bm25_paragraph(
        hyph_chunk,
        "Aufklärungsinitiative empfiehlt",
        "Aufklärungsinitiative",
    )
    assert "Aufklärungs-\ninitiative" in hyph_chunk[h_start:h_end]

    manager = _make_manager(tmp_path)
    doc = manager.ingest_document(
        DocumentCreate(
            content=pdf_chunk,
            metadata=DocumentMetadata(
                title="Single Newline PDF Guideline",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
            ),
        )
    )
    manager.record_processing_metadata(
        doc_id=doc.doc_id,
        chunk_ids=[f"{doc.doc_id}:chunk:0"],
        omissions=[],
        chunk_spans=[
            ChunkSpan(chunk_id=f"{doc.doc_id}:chunk:0", start_char=0, end_char=len(pdf_chunk))
        ],
    )
    manager.graph.vector_store.upsert(
        f"{doc.doc_id}:chunk:0",
        pdf_chunk,
        entry_type="chunk",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="metformin",
        canonical_name="Metformin",
        node_type="drug",
        description="First-line biguanide for type 2 diabetes.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="egfr-monitoring",
        canonical_name="eGFR Monitoring",
        node_type="monitoring",
        description="Renal function assessment before metformin.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_edge(
        source="metformin",
        target="egfr-monitoring",
        relation="REQUIRES_MONITORING",
        doc_id=doc.doc_id,
        excerpt="metformin renal eGFR dosing guidance",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )

    rewriter_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": ["Metformin eGFR renal function"],
                    "retrieval_mode": "hybrid",
                }
            )
        ]
    )
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(),
    )
    response = await pipeline.run(
        QueryRequest(
            session_id="pdf-window-session",
            query="How should Metformin be dosed and monitored with eGFR?",
            top_k_seeds=1,
            k_hops=1,
            retrieval_mode="hybrid",
        )
    )

    assert "Retrieved chunks:" in response.answer
    retrieved_section = response.answer.split("Retrieved chunks:", 1)[1]
    assert "500 mg once daily" in retrieved_section


@pytest.mark.anyio
async def test_query_pipeline_citation_collects_multiple_excerpts_and_chunk_ids(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    doc = manager.ingest_document(
        DocumentCreate(
            content=(
                "Metformin is first-line therapy for type 2 diabetes.\n\n"
                "Check eGFR before initiating metformin and monitor annually.\n\n"
                "Discontinue metformin if eGFR falls below 30 mL/min."
            ),
            metadata=DocumentMetadata(
                title="Diabetes Pharmacotherapy Guideline",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
                description="Fallback guideline description.",
            ),
        )
    )
    manager.graph.upsert_node(
        node_id="metformin",
        canonical_name="Metformin",
        node_type="drug",
        description="First-line biguanide for type 2 diabetes.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="type-2-diabetes",
        canonical_name="Type 2 Diabetes",
        node_type="condition",
        description="Chronic metabolic disorder.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="egfr-monitoring",
        canonical_name="eGFR Monitoring",
        node_type="monitoring",
        description="Renal function assessment.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_edge(
        source="metformin",
        target="type-2-diabetes",
        relation="TREATS",
        doc_id=doc.doc_id,
        excerpt="Metformin is first-line therapy for type 2 diabetes.",
        chunk_id=f"{doc.doc_id}:chunk:0",
    )
    manager.graph.upsert_edge(
        source="metformin",
        target="egfr-monitoring",
        relation="REQUIRES_MONITORING",
        doc_id=doc.doc_id,
        excerpt="Check eGFR before initiating metformin and monitor annually.",
        chunk_id=f"{doc.doc_id}:chunk:1",
    )

    rewriter_llm = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": ["Metformin type 2 diabetes eGFR"],
                    "retrieval_mode": "hybrid",
                }
            )
        ]
    )
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rewriter_llm, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(),
    )
    response = await pipeline.run(
        QueryRequest(
            session_id="multi-excerpt-citation-session",
            query="How is metformin used and monitored in type 2 diabetes?",
            top_k_seeds=2,
            k_hops=2,
            retrieval_mode="hybrid",
        )
    )

    assert len(response.citations) == 1
    citation = response.citations[0]
    assert len(citation.excerpts) == 2
    assert citation.excerpt == citation.excerpts[0]
    assert set(citation.excerpts) == {
        "Metformin is first-line therapy for type 2 diabetes.",
        "Check eGFR before initiating metformin and monitor annually.",
    }
    assert set(citation.chunk_ids) == {
        f"{doc.doc_id}:chunk:0",
        f"{doc.doc_id}:chunk:1",
    }


@pytest.mark.anyio
async def test_query_pipeline_rrf_fusion_rag_and_hybrid_dedup(tmp_path: Path) -> None:
    """Verify RRF fusion for chunk+table in rag mode and chunk_id deduplication in hybrid mode."""
    manager = _make_manager(tmp_path)
    chunk0_text = "Metformin initial dosing is 500 mg twice daily with meals."
    chunk1_text = "Assess eGFR before initiating metformin and discontinue if eGFR < 30."
    doc_text = f"{chunk0_text}\n\n{chunk1_text}"
    doc = manager.ingest_document(
        DocumentCreate(
            content=doc_text,
            metadata=DocumentMetadata(
                title="Metformin Renal & Dosing Guide",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
            ),
        )
    )
    c0_id = f"{doc.doc_id}:chunk:0"
    c1_id = f"{doc.doc_id}:chunk:1"
    t0_id = f"{doc.doc_id}:table:0"
    table_md = "| eGFR | Dosing |\n| --- | --- |\n| < 30 | Contraindicated |"
    table_summary = "eGFR >= 45: standard dose | eGFR 30-44: max 1000 mg/day | eGFR < 30: contraindicated"
    manager.record_processing_metadata(
        doc_id=doc.doc_id,
        chunk_ids=[c0_id, c1_id],
        omissions=[],
        chunk_spans=[
            ChunkSpan(chunk_id=c0_id, start_char=0, end_char=len(chunk0_text)),
            ChunkSpan(
                chunk_id=c1_id,
                start_char=len(chunk0_text) + 2,
                end_char=len(doc_text),
            ),
        ],
        tables=[
            TableRecord(
                table_id=t0_id,
                doc_id=doc.doc_id,
                markdown=table_md,
                summary=table_summary,
            )
        ],
    )
    manager.graph.vector_store.upsert(c0_id, chunk0_text, entry_type="chunk", doc_id=doc.doc_id)
    manager.graph.vector_store.upsert(c1_id, chunk1_text, entry_type="chunk", doc_id=doc.doc_id)
    manager.graph.vector_store.upsert(
        t0_id,
        table_summary,
        entry_type="table",
        doc_id=doc.doc_id,
    )

    manager.graph.upsert_node(
        node_id="metformin",
        canonical_name="Metformin",
        node_type="drug",
        description="First-line biguanide.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_node(
        node_id="egfr-monitoring",
        canonical_name="eGFR Monitoring",
        node_type="monitoring",
        description="Renal function assessment.",
        doc_id=doc.doc_id,
    )
    manager.graph.upsert_edge(
        source="metformin",
        target="egfr-monitoring",
        relation="REQUIRES_MONITORING",
        doc_id=doc.doc_id,
        excerpt="Assess eGFR before initiating metformin and discontinue if eGFR < 30.",
        chunk_id=c1_id,
    )

    # 1. rag mode: RRF fuses chunk + table results across multiple search queries, no graph traversal
    rag_rewriter = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": [
                        "Metformin initial dosing 500 mg",
                        "Metformin eGFR contraindicated table",
                    ],
                    "retrieval_mode": "rag",
                }
            )
        ]
    )
    rag_pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=rag_rewriter, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(),
    )
    rag_resp = await rag_pipeline.run(
        QueryRequest(
            session_id="rrf-rag",
            query="Metformin dosing and eGFR table?",
            include_trace=True,
            retrieval_mode="rag",
        )
    )
    assert rag_resp.chosen_mode == "rag"
    assert rag_resp.trace is not None
    assert rag_resp.trace.traversed_edges == []
    assert "Retrieved chunks:" in rag_resp.answer
    assert "Retrieved tables:" in rag_resp.answer
    assert "rag_retrieval" in rag_resp.metrics.latencies_ms

    # 2. hybrid mode: combines chunk/table RRF with DiGraph traversal and deduplicates chunk_ids
    hybrid_rewriter = _QueuedLLMClient(
        [
            json.dumps(
                {
                    "search_queries": [
                        "Metformin eGFR monitoring",
                        "eGFR contraindicated metformin",
                    ],
                    "retrieval_mode": "hybrid",
                }
            )
        ]
    )
    hybrid_pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=hybrid_rewriter, model="gpt-4o-mini"),
        answer_agent=AnswerAgent(),
    )
    hybrid_resp = await hybrid_pipeline.run(
        QueryRequest(
            session_id="rrf-hybrid",
            query="Metformin eGFR monitoring?",
            top_k_seeds=1,
            k_hops=1,
            include_trace=True,
            retrieval_mode="hybrid",
        )
    )
    assert hybrid_resp.chosen_mode == "hybrid"
    assert hybrid_resp.trace is not None
    assert len(hybrid_resp.trace.traversed_edges) == 1
    assert "Retrieved chunks:" in hybrid_resp.answer
    assert "Retrieved tables:" in hybrid_resp.answer
    # c1_id is referenced by both edge evidence and vector chunk search; it must be deduplicated
    assert hybrid_resp.answer.count(c1_id) == 1


def test_rank_traversed_edges_idf_bm25_and_bridge_bonus() -> None:
    from clinical_graph.pipelines.query import _rank_traversed_edges

    # Generic boilerplate edges mentioning "type 2 diabetes" on many edges (low IDF)
    boilerplate_edges = [
        {
            "source": "type-2-diabetes",
            "target": f"generic-target-{i}",
            "relation": "ASSOCIATED_WITH",
            "weight": 5.0,
            "relationship_description": "General type 2 diabetes management.",
            "evidence": [
                {"doc_id": "d1", "excerpt": "Type 2 diabetes routine care."},
                {"doc_id": "d2", "excerpt": "Type 2 diabetes follow-up."},
            ],
        }
        for i in range(5)
    ]
    # Specific clinical bridge edge between two seed nodes
    specific_edge = {
        "source": "empagliflozin",
        "target": "egfr",
        "relation": "CONTRAINDICATED_IF",
        "weight": 1.0,
        "relationship_description": "Empagliflozin initiation is not recommended when eGFR is below 20.",
        "evidence": [
            {
                "doc_id": "d1",
                "excerpt": "Do not initiate empagliflozin if eGFR < 20 mL/min/1.73m2.",
            }
        ],
    }
    zero_rel_edge = {
        "source": "aspirin",
        "target": "bleeding",
        "relation": "CAUSES",
        "weight": 10.0,
        "relationship_description": "Antiplatelet bleeding risk.",
        "evidence": [{"doc_id": "d3", "excerpt": "Monitor for GI bleeding."}],
    }

    ranked = _rank_traversed_edges(
        [*boilerplate_edges, zero_rel_edge, specific_edge],
        seed_nodes=[("empagliflozin", 0.95), ("egfr", 0.90), ("type-2-diabetes", 0.80)],
        query="Can empagliflozin be started in type 2 diabetes with low eGFR?",
        search_queries=["empagliflozin eGFR threshold type 2 diabetes"],
    )

    assert ranked[0]["source"] == "empagliflozin"
    assert ranked[0]["target"] == "egfr"
    assert ranked[-1]["source"] == "aspirin"


@pytest.mark.anyio
async def test_answer_agent_synthesized_citations_populate_passages_and_highlights() -> None:
    from clinical_graph.agents.answer_agent import CitationQuote, SynthesizedAnswer
    from clinical_graph.schemas.query import Citation

    doc_id = "doc-sglt2"
    chunk1_text = "General introduction to diabetes care and lifestyle counseling."
    chunk2_text = (
        "Empagliflozin is initiated at 10 mg once daily. "
        "Do not initiate if eGFR is below 20 mL/min/1.73m2."
    )
    retrieved_chunks = [
        {
            "doc_id": doc_id,
            "slug": "sglt2-sop",
            "title": "SGLT2 SOP",
            "chunk_id": f"{doc_id}:chunk:0",
            "text": chunk1_text,
        },
        {
            "doc_id": doc_id,
            "slug": "sglt2-sop",
            "title": "SGLT2 SOP",
            "chunk_id": f"{doc_id}:chunk:1",
            "text": chunk2_text,
        },
    ]
    default_citation = Citation(
        doc_id=doc_id,
        slug="sglt2-sop",
        title="SGLT2 SOP",
        doc_type=DocType.INTERNAL_SOP,
        local_restriction=LocalRestriction.NONE,
        excerpt=chunk1_text,
        excerpts=[chunk1_text],
        chunk_ids=[f"{doc_id}:chunk:0"],
    )

    quote = "Do not initiate if eGFR is below 20 mL/min/1.73m2."
    payload = SynthesizedAnswer(
        answer="Start empagliflozin 10 mg daily unless eGFR < 20 mL/min/1.73m2.",
        citations=[CitationQuote(ref_id="C2", quote=quote)],
    )
    llm_client = _QueuedLLMClient([payload.model_dump_json()])
    agent = AnswerAgent(llm_client=llm_client, model="gpt-4o")

    response, _ = await agent.synthesize_with_usage_async(
        request=QueryRequest(
            session_id="cite-passages-test",
            query="When should empagliflozin not be initiated?",
        ),
        seed_nodes=[("empagliflozin", 0.9)],
        traversed_edges=[],
        documents=[],
        citations=[default_citation],
        retrieved_chunks=retrieved_chunks,
    )

    assert response.answer == payload.answer
    assert len(response.citations) == 1
    cit = response.citations[0]
    assert cit.excerpt == quote
    assert cit.excerpts == [quote]
    assert cit.chunk_ids == [f"{doc_id}:chunk:1"]
    assert len(cit.passages) == 1

    passage = cit.passages[0]
    assert passage.chunk_id == f"{doc_id}:chunk:1"
    assert passage.quote == quote
    assert passage.chunk_text == chunk2_text
    assert f"<mark>{quote}</mark>" in passage.highlighted_chunk
    assert passage.match_start == chunk2_text.index(quote)
    assert passage.match_end == chunk2_text.index(quote) + len(quote)




