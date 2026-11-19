"""Unit tests for LLM concurrency/backoff, batched embeddings, metadata auto-inference, and thread-safe KnowledgeManager writes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import litellm
import numpy as np
import pytest

from clinical_graph.agents.classifier import DocumentClassification, DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.config import Settings
from clinical_graph.graph.knowledge_graph import ClinicalKnowledgeGraph
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.graph.vector_store import LiteLLMEmbedder, LocalVectorStore
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.main import create_app
from clinical_graph.schemas.document import (
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    SourceType,
)
from clinical_graph.schemas.metrics import LLMUsage


def _mock_completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.mark.anyio
async def test_litellm_client_enforces_semaphore_concurrency_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_seen = 0

    async def fake_acompletion(**kwargs: object) -> SimpleNamespace:
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.02)
        active -= 1
        return _mock_completion_response("ok")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)

    client = LiteLLMClient(max_concurrent_calls=2, max_retries=0)
    results = await asyncio.gather(
        *(
            client.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(6)
        )
    )

    assert results == ["ok"] * 6
    assert max_seen == 2


@pytest.mark.anyio
async def test_litellm_client_retries_429_with_backoff_and_shared_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    class FakeRateLimitError(Exception):
        status_code = 429

    async def fake_acompletion(**kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeRateLimitError("rate limited")
        return _mock_completion_response("recovered")

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = LiteLLMClient(max_concurrent_calls=2, max_retries=2, base_retry_delay=0.25)
    result = await client.complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result == "recovered"
    assert attempts == 2
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] >= 0.25
    assert client._cooldown_until > 0.0


@pytest.mark.anyio
async def test_litellm_client_fast_fails_on_insufficient_quota_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleep_calls: list[float] = []

    class FakeQuotaExceededError(Exception):
        status_code = 429
        code = "insufficient_quota"

    async def fake_acompletion(**kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        raise FakeQuotaExceededError(
            "You exceeded your current quota, please check your plan and billing details."
        )

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = LiteLLMClient(max_concurrent_calls=2, max_retries=3, base_retry_delay=0.25)
    with pytest.raises(FakeQuotaExceededError):
        await client.complete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert attempts == 1
    assert sleep_calls == []


@pytest.mark.anyio
async def test_litellm_client_passes_timeout_and_retries_on_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    seen_timeouts: list[object] = []
    sleep_calls: list[float] = []

    async def fake_acompletion(**kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        seen_timeouts.append(kwargs.get("timeout"))
        if attempts == 1:
            raise TimeoutError("request timed out")
        return _mock_completion_response("ok-after-timeout")

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = LiteLLMClient(max_concurrent_calls=2, max_retries=2, base_retry_delay=0.1, timeout=12.5)
    with caplog.at_level("WARNING"):
        result = await client.complete(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert result == "ok-after-timeout"
    assert attempts == 2
    assert seen_timeouts == [12.5, 12.5]
    assert len(sleep_calls) == 1
    assert any("TimeoutError" in rec.message and "attempt 1/2" in rec.message for rec in caplog.records)


def test_litellm_embedder_embed_many_and_vector_store_upsert_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_embedding(**kwargs: object) -> SimpleNamespace:
        batch = list(kwargs["input"])  # type: ignore[arg-type]
        calls.append(batch)
        data = [
            {"index": idx, "embedding": [3.0, 4.0, 0.0]}
            for idx in range(len(batch))
        ]
        return SimpleNamespace(data=data)

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    texts = [f"term {i}" for i in range(105)]
    vectors = embedder.embed_many(texts)

    assert len(calls) == 2
    assert len(calls[0]) == 100
    assert len(calls[1]) == 5
    assert len(vectors) == 105
    assert all(np.allclose(v, np.array([0.6, 0.8, 0.0], dtype=np.float32)) for v in vectors)

    store = LocalVectorStore(embedder=embedder)
    store.upsert_many([(f"node_{i}", f"text {i}") for i in range(10)])
    assert store.count() == 10
    assert len(calls) == 3

    graph = ClinicalKnowledgeGraph(vector_store=store)
    nodes = graph.upsert_nodes_batch(
        [
            {"node_id": "n1", "canonical_name": "Metformin", "node_type": "drug"},
            {"node_id": "n2", "canonical_name": "Type 2 Diabetes", "node_type": "condition"},
        ],
        doc_id="doc_1",
    )
    assert len(nodes) == 2
    # Batch 4: is_node embeddings; Batch 5: label_vectors embeddings
    assert len(calls) == 5
    assert store.count() == 12


@pytest.mark.anyio
async def test_classifier_inference_extractor_backfill_and_upload_csv_fields(
    tmp_path: Path,
) -> None:
    # 1. Classifier parses inferred metadata fields
    classification_json = DocumentClassification(
        is_allowed=True,
        predicted_doc_type=DocType.GUIDELINE,
        inferred_conditions=["Type 2 Diabetes"],
        inferred_target_populations=["Adults"],
        inferred_specialties=["Endocrinology"],
        inferred_local_restriction=LocalRestriction.US,
        inferred_description="First-line glycemic management guideline.",
    ).model_dump_json()

    classifier_llm = MagicMock()
    classifier_llm.complete_with_usage = AsyncMock(
        return_value=(classification_json, LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    )
    classifier = DocumentClassifier(llm_client=classifier_llm)
    parsed_cls, _ = await classifier.classify_async("Metformin is first-line for adults with T2D.")
    assert parsed_cls.inferred_conditions == ["Type 2 Diabetes"]
    assert parsed_cls.inferred_local_restriction == LocalRestriction.US

    # 2. Extractor backfills empty metadata.conditions and target_populations from extracted nodes
    node_payload = (
        '{"nodes": ['
        '{"canonical_name": "Type 2 Diabetes", "node_type": "condition", "aliases": ["T2D"], "description": "Metabolic disorder"},'
        '{"canonical_name": "Older Adults", "node_type": "population", "aliases": [], "description": "Patients >= 65"}'
        ']}'
    )
    edge_payload = (
        '{"edges": ['
        '{"source": "Type 2 Diabetes", "target": "Older Adults", "relation": "RISK_FACTOR_FOR", "quote": "Type 2 diabetes in older adults."}'
        ']}'
    )
    extractor_llm = MagicMock()
    extractor_llm.complete_with_usage = AsyncMock(
        side_effect=[
            (node_payload, LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)),
            (edge_payload, LLMUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)),
        ]
    )
    manager = KnowledgeManager(graph_dir=tmp_path / "graph")
    extractor = GraphExtractor(llm_client=extractor_llm)
    doc_payload = DocumentCreate(
        content="Type 2 diabetes in older adults.",
        metadata=DocumentMetadata(
            title="Diabetes Care",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.RAW_TEXT,
        ),
    )
    record, _ = await extractor.ingest_with_extraction(manager=manager, payload=doc_payload)
    assert record.metadata.conditions == ["type 2 diabetes"]
    assert record.metadata.target_populations == ["older adults"]

    # 3. /v1/documents/upload parses comma-separated form fields and backfills remaining empty fields when classify=True
    app = create_app()
    with TestClient(app) as client:
        app.state.settings = Settings(_env_file=None)
        app.state.knowledge_manager = KnowledgeManager()
        app.state.document_classifier = classifier

        response = client.post(
            "/v1/documents/upload",
            data={
                "title": "Uploaded Guideline",
                "conditions": "Hypertension, Heart Failure",
                "target_populations": "",
                "specialties": "Cardiology, Internal Medicine",
                "classify": "true",
            },
            files={"file": ("guideline.txt", b"Clinical guidance for hypertension and heart failure.", "text/plain")},
        )
        assert response.status_code == 201
        body = response.json()
        meta = body["metadata"]
        # Provided CSV fields are preserved (not overwritten by classifier)
        assert meta["conditions"] == ["hypertension", "heart failure"]
        assert meta["specialties"] == ["cardiology", "internal medicine"]
        # Empty fields are backfilled from classification.inferred_*
        assert meta["target_populations"] == ["adults"]
        assert meta["local_restriction"] == "US"
        assert meta["description"] == "First-line glycemic management guideline."


def test_knowledge_manager_thread_safe_concurrent_writes(tmp_path: Path) -> None:
    manager = KnowledgeManager(
        storage_dir=tmp_path / "docs",
        graph_dir=tmp_path / "graph",
    )

    def worker(idx: int) -> str:
        payload = DocumentCreate(
            content=f"Clinical protocol {idx} for condition {idx}.",
            metadata=DocumentMetadata(
                title=f"Protocol {idx}",
                doc_type=DocType.GUIDELINE,
                source_type=SourceType.RAW_TEXT,
            ),
        )
        rec = manager.ingest_document(payload)
        manager.record_processing_metadata(
            doc_id=rec.doc_id,
            chunk_ids=[f"{rec.doc_id}:chunk:0"],
            omissions=[],
        )
        manager.update_inferred_metadata(
            rec.doc_id,
            conditions=[f"condition {idx}"],
        )
        return rec.doc_id

    with ThreadPoolExecutor(max_workers=6) as pool:
        doc_ids = list(pool.map(worker, range(12)))

    assert len(set(doc_ids)) == 12
    assert manager.db.count() == 12
    for doc_id in doc_ids:
        stored = manager.db.get_by_id(doc_id)
        assert stored is not None
        assert len(stored.chunk_ids) == 1
        assert len(stored.metadata.conditions) == 1
