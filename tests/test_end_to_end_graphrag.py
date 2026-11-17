"""End-to-end tests for GraphRAG persistence, chunked extraction, and query API."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pydantic import ValidationError

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.extractor import ExtractedEdge, GraphExtractor
from clinical_graph.config import Settings
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    DeterministicHashEmbedder,
    KnowledgeManager,
    LocalVectorStore,
)
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.main import create_app
from clinical_graph.parser import chunk_text
from clinical_graph.schemas.document import (
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    SourceType,
)
from clinical_graph.schemas.metrics import LLMUsage


def test_chunk_text_preserves_markdown_tables_and_sections() -> None:
    """chunk_text splits on headings/paragraphs while keeping Markdown tables intact."""
    table = (
        "| Drug | Starting Dose | Max Dose |\n"
        "| --- | --- | --- |\n"
        "| Methylphenidate | 5 mg BID | 60 mg/day |\n"
        "| Atomoxetine | 40 mg/day | 100 mg/day |"
    )
    text = (
        "# ADHD Pharmacotherapy\n\n"
        "First-line stimulant therapy improves core symptoms in most patients.\n\n"
        f"{table}\n\n"
        "# Monitoring\n\n"
        "Assess blood pressure, heart rate, height, and weight at each follow-up visit."
    )

    chunks = chunk_text(text, max_chars=220, overlap_paragraphs=1)

    assert len(chunks) >= 2
    assert any(table in chunk for chunk in chunks)


@pytest.mark.anyio
async def test_graph_extractor_deduplicates_across_chunks() -> None:
    """GraphExtractor.extract chunks long text and merges duplicate nodes and edges."""
    long_text = (
        "# Section 1\n\n"
        + ("Methylphenidate (Ritalin) treats ADHD. " * 6)
        + "\n\n# Section 2\n\n"
        + ("Methylphenidate (Concerta) treats ADHD. " * 6)
    )

    graph_payload_1 = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Methylphenidate",
                    "node_type": "drug",
                    "aliases": ["Ritalin"],
                    "description": "CNS stimulant",
                },
                {
                    "canonical_name": "ADHD",
                    "node_type": "condition",
                    "aliases": ["Attention-Deficit/Hyperactivity Disorder"],
                    "description": "",
                },
            ],
            "edges": [
                {
                    "source": "Methylphenidate",
                    "target": "ADHD",
                    "relation": "TREATS",
                    "quote": "Methylphenidate (Ritalin) treats ADHD.",
                }
            ],
        }
    )
    graph_payload_2 = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Methylphenidate",
                    "node_type": "drug",
                    "aliases": ["Concerta"],
                    "description": "",
                }
            ],
            "edges": [
                {
                    "source": "Methylphenidate",
                    "target": "ADHD",
                    "relation": "TREATS",
                    "quote": "Methylphenidate (Ritalin) treats ADHD.",
                }
            ],
        }
    )

    mock_llm = MagicMock(spec=LiteLLMClient)
    mock_llm.complete = AsyncMock(side_effect=[graph_payload_1, graph_payload_2])

    extractor = GraphExtractor(llm_client=mock_llm)
    result = await extractor.extract(
        long_text,
        max_chars=350,
        overlap_paragraphs=0,
        max_extract_chars=350,
    )

    assert len(result.nodes) == 2
    drug_node = next(n for n in result.nodes if n.canonical_name == "Methylphenidate")
    assert drug_node.aliases == ["Ritalin", "Concerta"]
    assert drug_node.description == "CNS stimulant"
    assert len(result.edges) == 1


@pytest.mark.anyio
async def test_end_to_end_graphrag_persistence_and_query(tmp_path: Path) -> None:
    """Ingest with GraphExtractor, save to disk, reload fresh manager, and query via API."""
    graph_dir = tmp_path / "graph"
    docs_dir = tmp_path / "documents"

    verbatim_quote = (
        "Methylphenidate (Ritalin) is first-line pharmacotherapy for pediatric and adult ADHD."
    )
    doc_content = (
        "# ADHD Clinical Practice Guideline\n\n"
        f"{verbatim_quote}\n\n"
        "Monitor blood pressure and heart rate before initiation and during titration."
    )

    mock_extract_llm = MagicMock(spec=LiteLLMClient)
    mock_extract_llm.complete = AsyncMock(
        return_value=json.dumps(
            {
                "nodes": [
                    {
                        "canonical_name": "Methylphenidate",
                        "node_type": "drug",
                        "aliases": ["Ritalin", "Concerta"],
                        "description": "First-line CNS stimulant for ADHD.",
                    },
                    {
                        "canonical_name": "ADHD",
                        "node_type": "condition",
                        "aliases": ["Attention-Deficit/Hyperactivity Disorder"],
                        "description": "Neurodevelopmental disorder.",
                    },
                ],
                "edges": [
                    {
                        "source": "Methylphenidate",
                        "target": "ADHD",
                        "relation": "TREATS",
                        "quote": verbatim_quote,
                    }
                ],
            }
        )
    )

    manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=docs_dir,
        graph_dir=graph_dir,
    )
    extractor = GraphExtractor(llm_client=mock_extract_llm)
    payload = DocumentCreate(
        content=doc_content,
        metadata=DocumentMetadata(
            title="ADHD Clinical Practice Guideline",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.MARKDOWN,
            local_restriction=LocalRestriction.NONE,
            conditions=["ADHD"],
            target_populations=["pediatric", "adult"],
        ),
    )

    record, _ = await extractor.ingest_with_extraction(manager, payload)
    manager.save()

    assert (graph_dir / "documents.db").is_file()
    assert (graph_dir / "graph.json").is_file()
    assert (graph_dir / "vectors.npz").is_file()
    assert (graph_dir / "vector_texts.json").is_file()

    # Reload into a completely fresh KnowledgeManager instance
    fresh_manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=docs_dir,
        graph_dir=graph_dir,
    )
    fresh_manager.load()

    assert fresh_manager.db.count() == 1
    assert fresh_manager.graph.get_node("drug-methylphenidate") is not None

    mock_answer_llm = MagicMock(spec=LiteLLMClient)
    mock_answer_llm.complete = AsyncMock(
        return_value=(
            "Methylphenidate (Ritalin) is recommended as first-line pharmacotherapy "
            f"for pediatric and adult ADHD [{record.slug}]."
        )
    )

    app = create_app()
    with TestClient(app) as client:
        app.state.settings = Settings(_env_file=None)
        app.state.knowledge_manager = fresh_manager
        app.state.answer_agent = AnswerAgent(llm_client=mock_answer_llm, model="gpt-4o")

        resp = client.post(
            "/v1/query",
            json={
                "query": "What is Ritalin used for?",
                "session_id": "e2e-ritalin-session",
                "include_trace": True,
            },
        )

    assert resp.status_code == 200
    body = resp.json()

    assert "Methylphenidate (Ritalin)" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["doc_id"] == record.doc_id
    assert body["citations"][0]["excerpt"] == verbatim_quote

    trace = body["trace"]
    assert trace is not None
    assert "drug-methylphenidate" in trace["seed_nodes"]
    assert trace["traversed_edges"]
    evidences = [
        ev
        for edge in trace["traversed_edges"]
        for ev in edge.get("evidence", [])
    ]
    assert any(ev.get("excerpt") == verbatim_quote for ev in evidences)


def test_create_document_with_extract_flag(tmp_path: Path) -> None:
    """POST /v1/documents?extract=true runs GraphExtractor and persists to graph_dir."""
    graph_dir = tmp_path / "graph"
    manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=tmp_path / "docs",
        graph_dir=graph_dir,
    )

    with patch("clinical_graph.llm.client.LiteLLMClient.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.side_effect = [
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_name": "Metformin",
                            "node_type": "drug",
                            "aliases": ["Glucophage"],
                            "description": "First-line T2D agent.",
                        }
                    ]
                }
            ),
            json.dumps({"edges": []}),
        ]

        app = create_app()
        with TestClient(app) as client:
            app.state.settings = Settings(_env_file=None)
            app.state.knowledge_manager = manager

            resp = client.post(
                "/v1/documents?extract=true",
                json={
                    "content": "Metformin (Glucophage) is first-line therapy for type 2 diabetes.",
                    "metadata": {
                        "title": "T2D Guideline",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                    },
                },
            )

    assert resp.status_code == 201
    body = resp.json()
    assert len(body["chunk_ids"]) == 1
    assert body["chunk_ids"][0] == f"{body['doc_id']}:chunk:0"
    assert body["omissions"] == []
    assert manager.graph.get_node("drug-metformin") is not None
    assert (graph_dir / "graph.json").is_file()


def test_persisted_chunk_ids_and_omissions_reload_on_restart(tmp_path: Path) -> None:
    """chunk_ids and omissions persist to documents.db and reload on restart."""
    graph_dir = tmp_path / "graph"
    manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=tmp_path / "docs",
        graph_dir=graph_dir,
    )

    content = (
        "# Anticoagulation Protocol\n\n"
        "Apixaban is dosed at 5 mg twice daily for atrial fibrillation.\n\n"
        "## References\n\n"
        "[1] Granger CB et al. NEJM. 2011;365:981-992. doi:10.1056/NEJMoa1107039\n"
        "[2] Connolly SJ et al. NEJM. 2009;361:1139-1151. PMID: 19717844\n\n"
        "## Appendix A: Dosing Table\n\n"
        "| Drug | Standard Dose |\n"
        "| --- | --- |\n"
        "| Apixaban | 5 mg BID |"
    )

    with patch("clinical_graph.llm.client.LiteLLMClient.complete", new_callable=AsyncMock) as mock_complete:
        mock_complete.side_effect = [
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_name": "Apixaban",
                            "node_type": "drug",
                            "aliases": ["Eliquis"],
                            "description": "Factor Xa inhibitor.",
                        }
                    ]
                }
            ),
            json.dumps({"edges": []}),
        ]

        app = create_app()
        with TestClient(app) as client:
            app.state.settings = Settings(_env_file=None)
            app.state.knowledge_manager = manager

            resp = client.post(
                "/v1/documents/upload",
                data={
                    "title": "Anticoagulation Protocol",
                    "doc_type": "guideline",
                    "extract": "true",
                },
                files={"file": ("protocol.md", content.encode("utf-8"), "text/markdown")},
            )

    assert resp.status_code == 201
    uploaded = resp.json()
    doc_id = uploaded["doc_id"]
    assert uploaded["chunk_ids"] == [f"{doc_id}:chunk:0"]
    assert len(uploaded["omissions"]) == 1
    assert uploaded["omissions"][0]["reason"] == "references"
    assert uploaded["omissions"][0]["heading"] == "## References"

    assert (graph_dir / "documents.db").is_file()

    # Simulate restart by loading a fresh KnowledgeManager from graph_dir
    reloaded = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=tmp_path / "docs",
        graph_dir=graph_dir,
    )
    reloaded.load()
    restored = reloaded.db.get_by_id(doc_id)
    assert restored is not None
    assert restored.storage_path == str(tmp_path / "docs" / f"{doc_id}.txt")
    assert restored.chunk_ids == [f"{doc_id}:chunk:0"]
    assert len(restored.omissions) == 1
    assert restored.omissions[0].reason == "references"
    assert restored.omissions[0].heading == "## References"


@pytest.mark.anyio
async def test_extraction_result_metrics_latencies_and_usage(tmp_path: Path) -> None:
    """GraphExtractor tracks step latencies and aggregates LLMUsage across chunks."""
    graph_payload = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Metformin",
                    "node_type": "drug",
                    "aliases": ["Glucophage"],
                    "description": "First-line T2D agent.",
                }
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
    )

    from types import SimpleNamespace

    mock_acompletion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=graph_payload))],
            usage={"prompt_tokens": 130, "completion_tokens": 45, "total_tokens": 175},
        )
    )
    mock_cost = MagicMock(return_value=0.003)

    manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=tmp_path / "docs",
        graph_dir=tmp_path / "graph",
    )
    extractor = GraphExtractor(llm_client=LiteLLMClient())
    payload = DocumentCreate(
        content="Metformin treats Type 2 Diabetes.",
        metadata=DocumentMetadata(
            title="Metformin Guideline",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.MARKDOWN,
        ),
    )

    with (
        patch("clinical_graph.llm.client.litellm.acompletion", mock_acompletion),
        patch("clinical_graph.llm.client.litellm.completion_cost", mock_cost),
    ):
        _, result = await extractor.ingest_with_extraction(manager, payload)

    latencies = result.metrics.latencies_ms
    for step in (
        "cleaning",
        "chunking",
        "graph_extraction",
        "graph_upsert",
        "total",
    ):
        assert step in latencies
        assert latencies[step] >= 0.0

    assert result.metrics.usage.prompt_tokens == 130
    assert result.metrics.usage.completion_tokens == 45
    assert result.metrics.usage.total_tokens == 175
    assert result.metrics.usage.estimated_cost_usd == pytest.approx(0.003)


@pytest.mark.anyio
async def test_graph_extractor_respects_max_chunks() -> None:
    """GraphExtractor.extract caps processed chunks when max_chunks is provided."""
    mock_llm = AsyncMock()
    mock_llm.complete_with_usage = AsyncMock(
        return_value=(
            json.dumps({"nodes": [], "edges": []}),
            LLMUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
        )
    )
    extractor = GraphExtractor(llm_client=mock_llm)
    text = "\n\n".join(f"Paragraph {i}: " + ("x" * 60) for i in range(6))

    result = await extractor.extract(text, max_chars=90, overlap_paragraphs=0, max_chunks=1)
    assert len(result.chunk_ids) == 1
    assert mock_llm.complete_with_usage.await_count == 1

    zero_result = await extractor.extract(text, max_chars=90, overlap_paragraphs=0, max_chunks=0)
    assert zero_result.chunk_ids == []
    assert mock_llm.complete_with_usage.await_count == 1

    with pytest.raises(ValueError, match="max_chunks"):
        await extractor.extract(text, max_chunks=-1)


@pytest.mark.anyio
async def test_graph_extractor_alias_aware_merging_and_edge_remapping() -> None:
    """Nodes matching via aliases merge across chunks and edges remap to merged canonical_name."""
    long_text = (
        "# English Section\n\n"
        + ("Metformin is first-line therapy for Type 2 Diabetes. " * 4)
        + "\n\n# German Section\n\n"
        + ("Metformin wird bei Typ-2-Diabetes empfohlen. " * 4)
    )

    graph_payload_1 = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Metformin",
                    "node_type": "drug",
                    "aliases": ["Glucophage"],
                    "description": "Biguanide",
                },
                {
                    "canonical_name": "Type 2 Diabetes",
                    "node_type": "condition",
                    "aliases": ["T2D", "Typ-2-Diabetes"],
                    "description": "Metabolic disorder",
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
    )
    graph_payload_2 = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Typ-2-Diabetes",
                    "node_type": "condition",
                    "aliases": ["Diabetes mellitus Typ 2"],
                    "description": "Chronic metabolic disorder characterized by insulin resistance",
                }
            ],
            "edges": [
                {
                    "source": "Metformin",
                    "target": "Typ-2-Diabetes",
                    "relation": "TREATS",
                    "quote": "Metformin wird bei Typ-2-Diabetes empfohlen.",
                },
                {
                    "source": "Metformin",
                    "target": "Typ-2-Diabetes",
                    "relation": "RECOMMENDED_FOR",
                    "quote": "Metformin wird bei Typ-2-Diabetes empfohlen.",
                },
            ],
        }
    )

    mock_llm = MagicMock(spec=LiteLLMClient)
    mock_llm.complete = AsyncMock(side_effect=[graph_payload_1, graph_payload_2])

    extractor = GraphExtractor(llm_client=mock_llm)
    result = await extractor.extract(
        long_text,
        max_chars=250,
        overlap_paragraphs=0,
        max_extract_chars=250,
    )

    assert len(result.nodes) == 2
    condition_node = next(n for n in result.nodes if n.node_type == "condition")
    assert condition_node.canonical_name == "Type 2 Diabetes"
    assert condition_node.description == (
        "Chronic metabolic disorder characterized by insulin resistance"
    )
    assert set(condition_node.aliases) == {"T2D", "Typ-2-Diabetes", "Diabetes mellitus Typ 2"}

    # Two TREATS edges (from different chunks/quotes) deduplicate to one; RECOMMENDED_FOR remaps target.
    assert len(result.edges) == 2
    assert all(edge.target == "Type 2 Diabetes" for edge in result.edges)
    assert {edge.relation for edge in result.edges} == {"TREATS", "RECOMMENDED_FOR"}


def test_extracted_edge_rejects_invalid_relation_type() -> None:
    """ExtractedEdge rejects invalid relation typos via RelationType Literal validation."""
    with pytest.raises(ValidationError):
        ExtractedEdge(
            source="Metformin",
            target="Lactic acidosis",
            relation="CASES",
            quote="Metformin rarely causes lactic acidosis.",
        )


@pytest.mark.anyio
async def test_cross_document_alias_resolution_and_tight_deletion(tmp_path: Path) -> None:
    """Entities matching via _alias_index merge across documents and survive tight deletion."""
    manager = KnowledgeManager(
        graph=ClinicalKnowledgeGraph(LocalVectorStore(DeterministicHashEmbedder())),
        storage_dir=tmp_path / "docs",
        graph_dir=tmp_path / "graph",
    )

    doc1_graph = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Type 2 Diabetes",
                    "node_type": "condition",
                    "aliases": ["T2DM"],
                    "description": "Chronic metabolic disorder",
                }
            ],
            "edges": [],
        }
    )

    doc2_graph = json.dumps(
        {
            "nodes": [
                {
                    "canonical_name": "Metformin",
                    "node_type": "drug",
                    "aliases": ["Glucophage"],
                    "description": "First-line biguanide",
                },
                {
                    "canonical_name": "Typ-2-Diabetes",
                    "node_type": "condition",
                    "aliases": ["Type 2 Diabetes"],
                    "description": "Metabolic disorder",
                },
            ],
            "edges": [
                {
                    "source": "Metformin",
                    "target": "Typ-2-Diabetes",
                    "relation": "TREATS",
                    "quote": "Metformin treats Typ-2-Diabetes.",
                }
            ],
        }
    )

    mock_llm = MagicMock(spec=LiteLLMClient)
    mock_llm.complete = AsyncMock(side_effect=[doc1_graph, doc2_graph])
    extractor = GraphExtractor(llm_client=mock_llm)

    doc1, _ = await extractor.ingest_with_extraction(
        manager,
        DocumentCreate(
            content="Type 2 Diabetes (T2DM) requires glycemic monitoring.",
            metadata=DocumentMetadata(
                title="Doc 1 T2DM",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
                local_restriction=LocalRestriction.NONE,
            ),
        ),
    )

    # Document nodes must be ignored by resolve_existing_node_id.
    assert manager.graph.resolve_existing_node_id("Doc 1 T2DM") is None

    doc2, _ = await extractor.ingest_with_extraction(
        manager,
        DocumentCreate(
            content="Metformin treats Typ-2-Diabetes.",
            metadata=DocumentMetadata(
                title="Doc 2 Metformin",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
                local_restriction=LocalRestriction.NONE,
            ),
        ),
    )

    t2d_node = manager.graph.get_node("condition-type-2-diabetes")
    assert t2d_node is not None
    assert manager.graph.get_node("condition-typ-2-diabetes") is None
    assert t2d_node.canonical_name == "Type 2 Diabetes"
    assert "Typ-2-Diabetes" in t2d_node.aliases
    assert t2d_node.doc_ids == {doc1.doc_id, doc2.doc_id}

    assert manager.graph.get_node("drug-metformin") is not None
    assert manager.graph.graph.has_edge("drug-metformin", "condition-type-2-diabetes")
    edge_data = manager.graph.graph.edges["drug-metformin", "condition-type-2-diabetes"]["edge"]
    assert edge_data.relation == "TREATS"
    assert edge_data.doc_ids == {doc2.doc_id}

    report = manager.delete_document(doc2.doc_id)
    assert report.deleted_from_db is True
    assert report.removed_nodes == 2
    assert report.removed_edges == 1

    assert manager.graph.get_node("drug-metformin") is None
    assert not manager.graph.graph.has_edge("drug-metformin", "condition-type-2-diabetes")

    surviving_t2d = manager.graph.get_node("condition-type-2-diabetes")
    assert surviving_t2d is not None
    assert surviving_t2d.doc_ids == {doc1.doc_id}
