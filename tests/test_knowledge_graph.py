"""Tests for LocalVectorStore, ClinicalKnowledgeGraph, KnowledgeManager, and API routes."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.extractor import (
    ExtractedEdge,
    ExtractedNode,
    ExtractionResult,
    GraphExtractor,
)
from clinical_graph.config import get_settings
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.main import create_app
from clinical_graph.schemas import (
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    SourceType,
)
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    DeterministicHashEmbedder,
    KnowledgeManager,
    LocalVectorStore,
)


def _make_doc(
    *,
    title: str,
    content: str,
    doc_type: DocType = DocType.GUIDELINE,
    local_restriction: LocalRestriction = LocalRestriction.DE,
    conditions: list[str] | None = None,
    description: str | None = None,
) -> DocumentCreate:
    return DocumentCreate(
        content=content,
        metadata=DocumentMetadata(
            title=title,
            doc_type=doc_type,
            source_type=SourceType.MARKDOWN,
            local_restriction=local_restriction,
            conditions=conditions or [],
            description=description,
        ),
    )


def test_deterministic_hash_embedder_and_vector_store() -> None:
    embedder = DeterministicHashEmbedder(dim=128)
    v1 = embedder.embed("Methylphenidate Ritalin ADHD")
    v2 = embedder.embed("Methylphenidate Ritalin ADHD")
    assert v1.shape == (128,)
    assert v1.dtype == np.float32
    assert np.allclose(v1, v2)
    assert np.isclose(float(np.linalg.norm(v1)), 1.0, atol=1e-5)

    store = LocalVectorStore(embedder=embedder)
    store.upsert("drug_mph", "Methylphenidate | Aliases: Ritalin, Concerta | Stimulant for ADHD")
    store.upsert("drug_ins", "Insulin glargine | Aliases: Lantus | Long-acting basal insulin")
    assert store.count() == 2

    hits = store.search("Ritalin ADHD", top_k=2)
    assert hits[0][0] == "drug_mph"
    assert hits[0][1] > hits[1][1]

    filtered = store.search("Ritalin", top_k=2, allowed_node_ids={"drug_ins"})
    assert len(filtered) == 1
    assert filtered[0][0] == "drug_ins"

    assert store.delete_many(["drug_mph", "missing"]) == 1
    assert store.count() == 1


def test_synonym_seed_matching_and_k_hop_traversal() -> None:
    graph = ClinicalKnowledgeGraph()
    graph.upsert_node(
        node_id="drug_methylphenidate",
        canonical_name="Methylphenidate",
        node_type="drug",
        aliases=["Ritalin", "Medikinet"],
        description="First-line stimulant medication for ADHD.",
        doc_id="doc_adhd_de",
    )
    graph.upsert_node(
        node_id="cond_adhd",
        canonical_name="ADHD",
        node_type="condition",
        aliases=["Attention-Deficit/Hyperactivity Disorder"],
        description="Neurodevelopmental disorder.",
        doc_id="doc_adhd_de",
    )
    graph.upsert_node(
        node_id="ae_insomnia",
        canonical_name="Insomnia",
        node_type="adverse_effect",
        aliases=["Sleep disturbance"],
        description="Difficulty initiating or maintaining sleep.",
        doc_id="doc_adhd_us",
    )

    graph.upsert_edge(
        source="drug_methylphenidate",
        target="cond_adhd",
        relation="TREATS",
        doc_id="doc_adhd_de",
        excerpt="Methylphenidate is recommended as first-line pharmacotherapy for ADHD.",
    )
    graph.upsert_edge(
        source="drug_methylphenidate",
        target="ae_insomnia",
        relation="CAUSES",
        doc_id="doc_adhd_us",
        excerpt="Insomnia is a common adverse reaction of Ritalin.",
    )

    # Exact brand synonym match for Ritalin -> drug_methylphenidate with score 1.0
    seeds_exact = graph.find_seed_nodes("Ritalin", top_k=3)
    assert seeds_exact[0] == ("drug_methylphenidate", 1.0)

    # Multi-word query with 1-word token fallback still surfaces drug_methylphenidate
    seeds = graph.find_seed_nodes("Ritalin dosing", top_k=3)
    assert seeds[0][0] == "drug_methylphenidate"
    assert seeds[0][1] >= 0.50

    # Restricting to doc_adhd_de should exclude ae_insomnia (only in doc_adhd_us)
    edges_de, matched_docs_de = graph.traverse_k_hops(
        ["drug_methylphenidate"],
        k_hops=2,
        allowed_doc_ids={"doc_adhd_de"},
    )
    assert matched_docs_de == {"doc_adhd_de"}
    assert len(edges_de) == 1
    assert edges_de[0]["target"] == "cond_adhd"
    assert edges_de[0]["relation"] == "TREATS"

    # Unrestricted traversal reaches both edges and both documents
    edges_all, matched_docs_all = graph.traverse_k_hops(["drug_methylphenidate"], k_hops=2)
    assert matched_docs_all == {"doc_adhd_de", "doc_adhd_us"}
    assert len(edges_all) == 2


def test_tight_document_deletion_shared_vs_exclusive_nodes() -> None:
    manager = KnowledgeManager()

    doc1 = manager.ingest_document(
        _make_doc(
            title="DE ADHD Guideline",
            content="Methylphenidate (Ritalin) treats ADHD; monitor blood pressure.",
            conditions=["adhd"],
        )
    )
    doc2 = manager.ingest_document(
        _make_doc(
            title="US ADHD Label",
            content="Methylphenidate carries a warning for hypertension.",
            local_restriction=LocalRestriction.US,
            conditions=["adhd"],
        )
    )

    # Shared node across doc1 and doc2
    manager.graph.upsert_node(
        node_id="drug_methylphenidate",
        canonical_name="Methylphenidate",
        node_type="drug",
        aliases=["Ritalin"],
        description="CNS stimulant.",
        doc_id=doc1.doc_id,
    )
    manager.graph.upsert_node(
        node_id="drug_methylphenidate",
        canonical_name="Methylphenidate",
        node_type="drug",
        aliases=["Concerta"],
        description="CNS stimulant.",
        doc_id=doc2.doc_id,
    )

    # Exclusive node & edge for doc1
    manager.graph.upsert_node(
        node_id="ae_appetite_loss",
        canonical_name="Decreased Appetite",
        node_type="adverse_effect",
        aliases=["Anorexia"],
        description="Common stimulant side effect.",
        doc_id=doc1.doc_id,
    )
    manager.graph.upsert_edge(
        source="drug_methylphenidate",
        target="ae_appetite_loss",
        relation="CAUSES",
        doc_id=doc1.doc_id,
        excerpt="Monitor weight due to decreased appetite.",
    )

    # Exclusive node & edge for doc2
    manager.graph.upsert_node(
        node_id="ae_hypertension",
        canonical_name="Hypertension",
        node_type="adverse_effect",
        aliases=["High Blood Pressure"],
        description="Elevated arterial blood pressure.",
        doc_id=doc2.doc_id,
    )
    manager.graph.upsert_edge(
        source="drug_methylphenidate",
        target="ae_hypertension",
        relation="CAUSES",
        doc_id=doc2.doc_id,
        excerpt="May increase blood pressure.",
    )

    assert manager.db.count() == 2
    # 2 document nodes + 3 entity nodes = 5 nodes & 5 embeddings
    assert manager.graph.graph.number_of_nodes() == 5
    assert manager.graph.vector_store.count() == 5

    # Delete doc1: doc1 node + ae_appetite_loss removed; shared drug_methylphenidate kept
    report = manager.delete_document(doc1.doc_id)
    assert report.doc_id == doc1.doc_id
    assert report.deleted_from_db is True
    assert report.removed_nodes == 2
    assert report.removed_edges == 1
    assert report.removed_embeddings == 2

    assert manager.db.count() == 1
    assert manager.db.get_by_id(doc1.doc_id) is None
    assert manager.graph.get_node("ae_appetite_loss") is None
    assert manager.graph.get_node(doc1.doc_id) is None

    shared = manager.graph.get_node("drug_methylphenidate")
    assert shared is not None
    assert shared.doc_ids == {doc2.doc_id}
    assert manager.graph.graph.number_of_nodes() == 3
    assert manager.graph.vector_store.count() == 3

    # Delete doc2: remaining nodes, edges, embeddings, and DB record are all removed
    report2 = manager.delete_document(doc2.doc_id)
    assert report2.deleted_from_db is True
    assert report2.removed_nodes == 3
    assert report2.removed_edges == 1
    assert report2.removed_embeddings == 3
    assert manager.db.count() == 0
    assert manager.graph.graph.number_of_nodes() == 0
    assert manager.graph.vector_store.count() == 0


def test_api_document_lifecycle_and_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.knowledge_manager.graph.vector_store._embedder = DeterministicHashEmbedder()
            app.state.answer_agent = AnswerAgent()

            create_resp = client.post(
                "/v1/documents",
                json={
                    "content": "Ritalin (methylphenidate) is indicated for pediatric ADHD.",
                    "metadata": {
                        "title": "Pediatric ADHD Guideline",
                        "doc_type": "guideline",
                        "source_type": "markdown",
                        "local_restriction": "DE",
                        "conditions": ["adhd"],
                        "target_populations": ["pediatric"],
                        "description": "German pediatric ADHD guideline covering Ritalin.",
                    },
                },
            )
            assert create_resp.status_code == 201
            created = create_resp.json()
            doc_id = created["doc_id"]
            slug = created["slug"]

            list_resp = client.get("/v1/documents", params={"condition": "adhd"})
            assert list_resp.status_code == 200
            assert len(list_resp.json()) == 1

            by_slug = client.get(f"/v1/documents/{slug}")
            assert by_slug.status_code == 200
            assert by_slug.json()["doc_id"] == doc_id

            query_resp = client.post(
                "/v1/query",
                json={
                    "query": "ADHD Ritalin",
                    "session_id": "sess-1",
                    "include_trace": True,
                },
            )
            assert query_resp.status_code == 200
            qdata = query_resp.json()
            assert qdata["session_id"] == "sess-1"
            assert len(qdata["citations"]) == 1
            assert qdata["citations"][0]["doc_id"] == doc_id
            assert qdata["trace"] is not None
            assert doc_id in qdata["trace"]["matched_doc_ids"]

            del_resp = client.delete(f"/v1/documents/{doc_id}")
            assert del_resp.status_code == 200
            del_data = del_resp.json()
            assert del_data["deleted_from_db"] is True
            assert del_data["removed_nodes"] == 1
            assert del_data["removed_embeddings"] == 1

            missing_resp = client.get(f"/v1/documents/{doc_id}")
            assert missing_resp.status_code == 404

            missing_del = client.delete(f"/v1/documents/{doc_id}")
            assert missing_del.status_code == 404
    finally:
        get_settings.cache_clear()


def test_disk_document_persistence_and_deletion(tmp_path: Path) -> None:
    storage_dir = tmp_path / "documents"
    manager = KnowledgeManager(storage_dir=storage_dir)

    payload = _make_doc(
        title="Disk Persisted Guideline",
        content="Metformin is first-line therapy for type 2 diabetes mellitus.",
        conditions=["type 2 diabetes"],
    )
    record = manager.ingest_document(payload)

    expected_file = storage_dir / f"{record.doc_id}.txt"
    assert record.storage_path == str(expected_file)
    assert expected_file.is_file()
    assert expected_file.read_text(encoding="utf-8") == payload.content

    report = manager.delete_document(record.doc_id)
    assert report.deleted_from_db is True
    assert report.deleted_from_disk is True
    assert report.removed_nodes == 1
    assert report.removed_embeddings == 1
    assert not expected_file.exists()


def test_graph_extractor_schema_and_ingest(tmp_path: Path) -> None:
    # Validate extraction schemas
    node = ExtractedNode(
        canonical_name="Methylphenidate",
        node_type="drug",
        aliases=["Ritalin", "Concerta"],
        description="First-line CNS stimulant for ADHD.",
    )
    edge = ExtractedEdge(
        source="Methylphenidate",
        target="ADHD",
        relation="TREATS",
        quote="Methylphenidate is indicated for pediatric ADHD.",
    )
    result = ExtractionResult(nodes=[node], edges=[edge])
    assert result.nodes[0].canonical_name == "Methylphenidate"
    assert result.edges[0].quote == "Methylphenidate is indicated for pediatric ADHD."

    graph_json = (
        '{"nodes": ['
        '{"canonical_name": "Methylphenidate", "node_type": "drug", '
        '"aliases": ["Ritalin"], "description": "CNS stimulant."},'
        '{"canonical_name": "ADHD", "node_type": "condition", '
        '"aliases": ["Attention-Deficit/Hyperactivity Disorder"], '
        '"description": "Neurodevelopmental disorder."}'
        '], "edges": ['
        '{"source": "Methylphenidate", "target": "ADHD", "relation": "TREATS", '
        '"quote": "Ritalin (methylphenidate) is indicated for pediatric ADHD."}'
        "]}"
    )

    mock_client = LiteLLMClient()
    mock_client.complete = AsyncMock(return_value=graph_json)  # type: ignore[method-assign]

    extractor = GraphExtractor(
        llm_client=mock_client,
        node_model="gpt-4o-mini",
        edge_model="gpt-4o-mini",
    )
    manager = KnowledgeManager(storage_dir=tmp_path / "docs")
    payload = _make_doc(
        title="ADHD Extraction Test",
        content="Ritalin (methylphenidate) is indicated for pediatric ADHD.",
        conditions=["adhd"],
    )

    record, extraction = asyncio.run(extractor.ingest_with_extraction(manager, payload))
    assert len(extraction.nodes) == 2
    assert len(extraction.edges) == 1
    assert mock_client.complete.await_count == 1

    # 1 document node + 2 extracted entity nodes
    assert manager.graph.graph.number_of_nodes() == 3
    assert manager.graph.graph.number_of_edges() == 1
    seeds = manager.graph.find_seed_nodes("Ritalin", top_k=2)
    assert seeds[0] == ("drug-methylphenidate", 1.0)


def test_vector_store_reconciles_dimension_mismatch_on_upsert_and_search(tmp_path: Path) -> None:
    small_store = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=32))
    small_store.upsert("n1", "Metformin | Aliases: Glucophage | First-line therapy for type 2 diabetes")
    small_store.upsert("n2", "Insulin glargine | Aliases: Lantus | Basal insulin")
    small_store.save(tmp_path)

    # Load into a store configured with a different embedding dimension.
    large_store = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=128))
    large_store.load(tmp_path)

    # Upserting a new vector reconciles existing stored vectors and allows saving cleanly.
    large_store.upsert("n3", "Empagliflozin | Aliases: Jardiance | SGLT2 inhibitor")
    large_store.save(tmp_path)

    reloaded = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=128))
    reloaded.load(tmp_path)
    assert reloaded.count() == 3
    hits = reloaded.search("Metformin type 2 diabetes", top_k=2)
    assert hits[0][0] == "n1"

    # Searching directly after loading mismatched dimensions also reconciles without error.
    direct_search_store = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=256))
    direct_search_store.load(tmp_path)
    direct_hits = direct_search_store.search("Empagliflozin SGLT2", top_k=2)
    assert direct_hits[0][0] == "n3"


def test_find_seed_nodes_vector_first_and_phrase_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = ClinicalKnowledgeGraph()
    graph.upsert_node(
        node_id="cond_dn",
        canonical_name="Diabetic Neuropathy",
        node_type="condition",
        aliases=["Diabetic Neuropathy"],
        description="Peripheral nerve damage caused by diabetes.",
        doc_id="doc1",
    )
    graph.upsert_node(
        node_id="proc_screening",
        canonical_name="Screening",
        node_type="procedure",
        aliases=["Screening"],
        description="Generic diagnostic screening procedure.",
        doc_id="doc1",
    )
    graph.upsert_node(
        node_id="drug_pregabalin",
        canonical_name="Pregabalin",
        node_type="drug",
        aliases=["Lyrica"],
        description="First-line neuropathic pain medication.",
        doc_id="doc1",
    )
    graph.upsert_node(
        node_id="drug_methylphenidate",
        canonical_name="Methylphenidate",
        node_type="drug",
        aliases=["Ritalin"],
        description="Stimulant medication.",
        doc_id="doc1",
    )

    monkeypatch.setattr(
        graph.vector_store,
        "search",
        lambda query, top_k=10, allowed_node_ids=None, api_key=None: [
            ("drug_pregabalin", 0.65),
        ],
    )

    # With top_k=2, phrase match (0.86) and confident vector hit (0.65) satisfy top_k,
    # so generic 1-word token "Screening" is not even added.
    seeds_top2 = graph.find_seed_nodes("Screening for Diabetic Neuropathy", top_k=2)
    assert seeds_top2 == [
        ("cond_dn", pytest.approx(0.86)),
        ("drug_pregabalin", 0.65),
    ]

    # With top_k=3, 1-word token fallback runs (0.50) but ranks below phrase (0.86) and vector (0.65).
    seeds_top3 = graph.find_seed_nodes("Screening for Diabetic Neuropathy", top_k=3)
    assert seeds_top3 == [
        ("cond_dn", pytest.approx(0.86)),
        ("drug_pregabalin", 0.65),
        ("proc_screening", 0.50),
    ]

    # Exact full-query alias match ("Ritalin") still scores 1.0.
    exact_seeds = graph.find_seed_nodes("Ritalin", top_k=2)
    assert exact_seeds[0] == ("drug_methylphenidate", 1.0)


def test_vector_store_entry_type_filtering_and_delete_by_doc_id() -> None:
    store = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=64))
    store.upsert("drug_metformin", "Metformin renal dosing", entry_type="node", doc_id="doc_1")
    store.upsert("doc_1::chunk_0", "Metformin renal dosing chunk", entry_type="chunk", doc_id="doc_1")
    store.upsert("doc_1::table_0", "Metformin renal dosing table", entry_type="table", doc_id="doc_1")
    store.upsert("doc_2::chunk_0", "Metformin renal dosing other doc", entry_type="chunk", doc_id="doc_2")

    assert store.count() == 4

    node_hits = store.search("Metformin renal dosing", top_k=10, entry_types={"node"})
    assert [nid for nid, _ in node_hits] == ["drug_metformin"]

    chunk_hits = store.search(
        "Metformin renal dosing",
        top_k=10,
        entry_types={"chunk"},
        allowed_doc_ids={"doc_1"},
    )
    assert [nid for nid, _ in chunk_hits] == ["doc_1::chunk_0"]

    table_hits = store.search("Metformin renal dosing", top_k=10, entry_types={"table"})
    assert [nid for nid, _ in table_hits] == ["doc_1::table_0"]

    rag_hits = store.search("Metformin renal dosing", top_k=10, entry_types={"chunk", "table"})
    assert {nid for nid, _ in rag_hits} == {"doc_1::chunk_0", "doc_1::table_0", "doc_2::chunk_0"}

    # delete_by_doc_id removes only chunk and table entries for doc_1, preserving node and doc_2
    removed = store.delete_by_doc_id("doc_1")
    assert removed == 2
    assert store.count() == 2
    assert [nid for nid, _ in store.search("Metformin", top_k=10, entry_types={"node"})] == ["drug_metformin"]
    assert [nid for nid, _ in store.search("Metformin", top_k=10, entry_types={"chunk"})] == ["doc_2::chunk_0"]


def test_knowledge_graph_self_loop_rejection_and_max_weight_merging() -> None:
    graph = ClinicalKnowledgeGraph()
    graph.upsert_node("drug_metformin", canonical_name="Metformin", doc_id="doc_1")
    graph.upsert_node("cond_ckd", canonical_name="CKD", doc_id="doc_1")

    # Self-loop (case/whitespace insensitive) must return None and not add an edge
    assert graph.upsert_edge(
        source="drug_metformin",
        target="drug_metformin",
        relation="INTERACTS_WITH",
        doc_id="doc_1",
    ) is None
    assert graph.upsert_edge(
        source="drug_metformin",
        target=" DRUG_METFORMIN ",
        relation="INTERACTS_WITH",
        doc_id="doc_1",
    ) is None
    assert graph.graph.number_of_edges() == 0

    # Double-edge between same (u, v) merges with max(weight) and longest description
    e1 = graph.upsert_edge(
        source="drug_metformin",
        target="cond_ckd",
        relation="CONTRAINDICATED_IN",
        doc_id="doc_1",
        excerpt="Avoid when eGFR < 30.",
        weight=4.0,
        relationship_description="Short desc",
    )
    assert e1 is not None
    assert e1.weight == 4.0

    e2 = graph.upsert_edge(
        source="drug_metformin",
        target="cond_ckd",
        relation="CONTRAINDICATED_IN",
        doc_id="doc_2",
        excerpt="Contraindicated in severe renal impairment.",
        weight=9.0,
        relationship_description="Detailed contraindication in severe CKD.",
    )
    assert e2 is not None
    assert graph.graph.number_of_edges() == 1
    assert e2.weight == 9.0
    assert e2.doc_ids == {"doc_1", "doc_2"}
    assert len(e2.evidence) == 2
    assert e2.relationship_description == "Detailed contraindication in severe CKD."

    # Lower subsequent weight does not reduce the stored max weight
    e3 = graph.upsert_edge(
        source="drug_metformin",
        target="cond_ckd",
        relation="CONTRAINDICATED_IN",
        doc_id="doc_3",
        weight=2.0,
    )
    assert e3 is not None
    assert e3.weight == 9.0




