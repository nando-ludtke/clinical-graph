"""Unit tests for Settings, LiteLLMClient, AnswerAgent, LocalVectorStore pre-filtering, and FastAPI routes."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.classifier import (
    BlockedDocumentCategory,
    DocumentClassification,
    DocumentClassifier,
)
from clinical_graph.agents.query_rewriter import QueryRewriter, RewrittenQueries
from clinical_graph.config import Settings, get_settings
from clinical_graph.graph.vector_store import DeterministicHashEmbedder, LocalVectorStore
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.main import create_app
from clinical_graph.schemas.document import (
    DocType,
    DocumentMetadata,
    DocumentRecord,
    LocalRestriction,
    SourceType,
)
from clinical_graph.schemas.query import ChatMessage, Citation, PatientContext, QueryRequest


def test_settings_env_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLINICAL_GRAPH_OPENAI_API_KEY", raising=False)

    defaults = Settings(_env_file=None)
    assert defaults.classifier_model == "gpt-4o-mini"
    assert defaults.embedding_model == "text-embedding-3-small"
    assert defaults.query_rewrite_model == "gpt-4o-mini"
    assert defaults.node_extraction_model == "gpt-4o-mini"
    assert defaults.edge_extraction_model == "gpt-4o-mini"
    assert defaults.answer_generation_model == "gpt-4o"
    assert defaults.openai_api_key is None

    monkeypatch.setenv("CLASSIFIER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("QUERY_REWRITE_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("NODE_EXTRACTION_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("EDGE_EXTRACTION_MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("ANSWER_GENERATION_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")

    overridden = Settings(_env_file=None)
    assert overridden.classifier_model == "gpt-4.1-mini"
    assert overridden.embedding_model == "text-embedding-3-large"
    assert overridden.query_rewrite_model == "gpt-4.1-mini"
    assert overridden.node_extraction_model == "gpt-4.1-mini"
    assert overridden.edge_extraction_model == "gpt-4.1-nano"
    assert overridden.answer_generation_model == "gpt-4.1"
    assert overridden.openai_api_key == "sk-env-test-key"


@pytest.mark.anyio
async def test_litellm_client_complete_forwards_api_key() -> None:
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Metformin is first-line."))]
    )
    mock_acompletion = AsyncMock(return_value=mock_response)
    client = LiteLLMClient()

    with patch("clinical_graph.llm.client.litellm.acompletion", mock_acompletion):
        result = await client.complete(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is first-line for T2D?"}],
            temperature=0.0,
            api_key="sk-req-123",
        )

    assert result == "Metformin is first-line."
    mock_acompletion.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is first-line for T2D?"}],
        temperature=0.0,
        timeout=45.0,
        api_key="sk-req-123",
    )

    mock_acompletion.reset_mock()
    with patch("clinical_graph.llm.client.litellm.acompletion", mock_acompletion):
        await client.complete(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Ping"}],
        )

    mock_acompletion.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Ping"}],
        temperature=0.0,
        timeout=45.0,
    )


def _sample_doc() -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        doc_id="doc-1",
        slug="t2d-guideline",
        content="Metformin is first-line therapy for type 2 diabetes.",
        metadata=DocumentMetadata(
            title="Type 2 Diabetes Guideline",
            doc_type=DocType.GUIDELINE,
            source_type=SourceType.MARKDOWN,
            local_restriction=LocalRestriction.NONE,
        ),
        created_at=now,
        updated_at=now,
    )


def test_answer_agent_synthesize_offline() -> None:
    agent = AnswerAgent()
    req = QueryRequest(query="First-line T2D therapy?", session_id="test-session")

    empty = agent.synthesize(
        request=req,
        seed_nodes=[],
        traversed_edges=[],
        documents=[],
        citations=[],
    )
    assert "No matching clinical evidence" in empty.answer
    assert empty.citations == []

    doc = _sample_doc()
    citation = Citation(
        doc_id=doc.doc_id,
        slug=doc.slug,
        title=doc.metadata.title,
        doc_type=doc.metadata.doc_type,
        local_restriction=doc.metadata.local_restriction,
    )
    resp = agent.synthesize(
        request=req,
        seed_nodes=[("drug-metformin", 0.95)],
        traversed_edges=[
            {
                "source": "drug-metformin",
                "relation": "TREATS",
                "target": "condition-t2d",
                "evidence": [{"excerpt": "Metformin is first-line therapy."}],
            }
        ],
        documents=[doc],
        citations=[citation],
    )
    assert "Query: First-line T2D therapy?" in resp.answer
    assert "drug-metformin --[TREATS]--> condition-t2d" in resp.answer
    assert len(resp.citations) == 1


@pytest.mark.anyio
async def test_answer_agent_synthesize_async_forwards_api_key() -> None:
    mock_client = MagicMock(spec=LiteLLMClient)
    mock_client.complete = AsyncMock(return_value="Synthesized clinical answer.")

    agent = AnswerAgent(llm_client=mock_client, model="gpt-4o")
    req = QueryRequest(query="First-line T2D therapy?", session_id="test-session")
    doc = _sample_doc()
    citation = Citation(
        doc_id=doc.doc_id,
        slug=doc.slug,
        title=doc.metadata.title,
        doc_type=doc.metadata.doc_type,
        local_restriction=doc.metadata.local_restriction,
    )

    resp = await agent.synthesize_async(
        request=req,
        seed_nodes=[("drug-metformin", 0.95)],
        traversed_edges=[],
        documents=[doc],
        citations=[citation],
        api_key="sk-answer-key",
    )

    assert resp.answer == "Synthesized clinical answer."
    mock_client.complete.assert_awaited_once()
    call_kwargs = mock_client.complete.await_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["api_key"] == "sk-answer-key"


def test_local_vector_store_prefilters_before_top_k() -> None:
    store = LocalVectorStore(embedder=DeterministicHashEmbedder(dim=128))
    store.upsert("n-exact-1", "metformin diabetes mellitus")
    store.upsert("n-exact-2", "metformin glycemic control")
    store.upsert("n-allowed", "metformin renal dosing adjustment")

    # Even with top_k=1, pre-filtering to {"n-allowed"} must return "n-allowed"
    # rather than selecting "n-exact-1" first and filtering it out afterward.
    hits = store.search("metformin diabetes", top_k=1, allowed_node_ids={"n-allowed"})
    assert len(hits) == 1
    assert hits[0][0] == "n-allowed"
    assert isinstance(hits[0][1], float)


def test_api_health_documents_and_query_forward_openai_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    get_settings.cache_clear()

    import litellm

    mock_embedding = MagicMock(
        return_value=SimpleNamespace(
            data=[{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32).tolist()}]
        )
    )
    mock_acompletion = AsyncMock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"search_queries": ["Metformin Renal Guideline"]}'
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Grounded answer from mocked LiteLLM.")
                    )
                ]
            ),
        ]
    )
    monkeypatch.setattr(litellm, "embedding", mock_embedding)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    try:
        app = create_app()
        with TestClient(app) as client:
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "ok"

            doc_resp = client.post(
                "/v1/documents",
                headers={"X-OpenAI-API-Key": "sk-header-key"},
                json={
                    "content": "Metformin is contraindicated in severe renal impairment.",
                    "metadata": {
                        "title": "Metformin Renal Guideline",
                        "doc_type": "guideline",
                        "source_type": "markdown",
                        "conditions": ["type 2 diabetes"],
                    },
                },
            )
            assert doc_resp.status_code == 201
            doc_id = doc_resp.json()["doc_id"]

            assert mock_embedding.call_count == 1
            assert mock_embedding.call_args.kwargs["api_key"] == "sk-header-key"

            query_resp = client.post(
                "/v1/query",
                headers={"X-OpenAI-API-Key": "sk-header-key"},
                json={
                    "query": "Metformin Renal Guideline",
                    "session_id": "test-session",
                    "top_k_seeds": 3,
                    "k_hops": 1,
                    "include_trace": True,
                },
            )
            assert query_resp.status_code == 200
            body = query_resp.json()
            assert body["answer"] == "Grounded answer from mocked LiteLLM."
            assert body["trace"]["search_queries"] == ["Metformin Renal Guideline"]
            assert body["trace"]["matched_doc_ids"] == [doc_id]
            assert len(body["citations"]) == 1
            assert body["citations"][0]["doc_id"] == doc_id

            assert mock_embedding.call_count >= 2
            assert mock_embedding.call_args.kwargs["api_key"] == "sk-header-key"
            assert mock_acompletion.await_count == 2
            for call in mock_acompletion.await_args_list:
                assert call.kwargs["api_key"] == "sk-header-key"
    finally:
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_query_rewriter_offline_and_mocked_history() -> None:
    req = QueryRequest(
        query="What about renal dosing for it?",
        session_id="rewrite-sess",
        messages=[
            ChatMessage(role="user", content="Tell me about Metformin in type 2 diabetes."),
            ChatMessage(role="assistant", content="Metformin is first-line therapy."),
        ],
        patient_context=PatientContext(age_group="geriatric", renal_impairment=True),
    )

    offline = QueryRewriter()
    assert await offline.rewrite_async(req) == ["What about renal dosing for it?"]

    mock_client = MagicMock(spec=LiteLLMClient)
    mock_client.complete = AsyncMock(
        return_value=(
            '{"search_queries": ["Metformin renal dosing", "metformin renal dosing", '
            '"Metformin contraindication eGFR"]}'
        )
    )
    rewriter = QueryRewriter(llm_client=mock_client, model="gpt-4o-mini")
    queries = await rewriter.rewrite_async(req, api_key="sk-rewrite-key")

    assert queries == ["Metformin renal dosing", "Metformin contraindication eGFR"]
    mock_client.complete.assert_awaited_once()
    call_kwargs = mock_client.complete.await_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["response_format"] is RewrittenQueries
    assert call_kwargs["api_key"] == "sk-rewrite-key"
    user_content = call_kwargs["messages"][1]["content"]
    assert "Tell me about Metformin in type 2 diabetes." in user_content
    assert "renal_impairment=True" in user_content
    assert "What about renal dosing for it?" in user_content


@pytest.mark.anyio
async def test_query_rewriter_falls_back_on_error() -> None:
    req = QueryRequest(query="Ritalin pediatric dose", session_id="err-sess")
    mock_client = MagicMock(spec=LiteLLMClient)
    mock_client.complete = AsyncMock(return_value="not-valid-json")

    rewriter = QueryRewriter(llm_client=mock_client)
    assert await rewriter.rewrite_async(req) == ["Ritalin pediatric dose"]


def test_multi_query_seed_merging_by_max_score_and_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            manager = app.state.knowledge_manager
            manager.graph.vector_store._embedder = DeterministicHashEmbedder()
            app.state.answer_agent = AnswerAgent()

            mock_rewriter_llm = MagicMock(spec=LiteLLMClient)
            mock_rewriter_llm.complete = AsyncMock(
                return_value='{"search_queries": ["query-one", "query-two"]}'
            )
            app.state.query_rewriter = QueryRewriter(
                llm_client=mock_rewriter_llm,
                model="gpt-4o-mini",
            )

            doc_resp = client.post(
                "/v1/documents",
                json={
                    "content": "Metformin and SGLT2 inhibitors for diabetes.",
                    "metadata": {
                        "title": "Diabetes Pharmacotherapy",
                        "doc_type": "guideline",
                        "source_type": "markdown",
                    },
                },
            )
            assert doc_resp.status_code == 201
            doc_id = doc_resp.json()["doc_id"]

            def fake_find_seed_nodes(
                query: str,
                top_k: int = 5,
                allowed_doc_ids: set[str] | None = None,
                api_key: str | None = None,
            ) -> list[tuple[str, float]]:
                if query == "query-one":
                    return [("node-a", 0.40), ("node-b", 0.70)]
                if query == "query-two":
                    return [("node-a", 0.95), ("node-c", 0.80)]
                return []

            with patch.object(
                manager.graph,
                "find_seed_nodes",
                side_effect=fake_find_seed_nodes,
            ) as mock_find:
                resp = client.post(
                    "/v1/query",
                    json={
                        "query": "What about it?",
                        "session_id": "merge-sess",
                        "top_k_seeds": 2,
                        "include_trace": True,
                    },
                )

            assert resp.status_code == 200
            assert mock_find.call_count == 2
            trace = resp.json()["trace"]
            assert trace["search_queries"] == ["query-one", "query-two"]
            # node-a max(0.40, 0.95) = 0.95, node-c = 0.80, node-b = 0.70 -> top 2 are [node-a, node-c]
            assert trace["seed_nodes"] == ["node-a", "node-c"]
            assert doc_id is not None
    finally:
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_litellm_client_complete_with_usage_extracts_tokens_and_cost() -> None:
    attr_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer text"))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
    )
    dict_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Dict usage text"))],
        usage={"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    )
    client = LiteLLMClient()

    with (
        patch(
            "clinical_graph.llm.client.litellm.acompletion",
            AsyncMock(return_value=attr_response),
        ),
        patch(
            "clinical_graph.llm.client.litellm.completion_cost",
            return_value=0.0025,
        ) as mock_cost,
    ):
        text, usage = await client.complete_with_usage(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert text == "Answer text"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 150
    assert usage.estimated_cost_usd == pytest.approx(0.0025)
    mock_cost.assert_called_once_with(completion_response=attr_response)

    with (
        patch(
            "clinical_graph.llm.client.litellm.acompletion",
            AsyncMock(return_value=dict_response),
        ),
        patch(
            "clinical_graph.llm.client.litellm.completion_cost",
            side_effect=RuntimeError("model price unknown"),
        ),
    ):
        text2, usage2 = await client.complete_with_usage(
            model="custom-model",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert text2 == "Dict usage text"
    assert usage2.prompt_tokens == 40
    assert usage2.completion_tokens == 10
    assert usage2.total_tokens == 50
    assert usage2.estimated_cost_usd == 0.0


def test_query_response_metrics_latencies_and_aggregated_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    get_settings.cache_clear()

    import litellm

    mock_embedding = MagicMock(
        return_value=SimpleNamespace(
            data=[{"embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32).tolist()}]
        )
    )
    rewrite_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"search_queries": ["Metformin"]}')
            )
        ],
        usage=SimpleNamespace(prompt_tokens=25, completion_tokens=10, total_tokens=35),
    )
    answer_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Metformin is first-line for T2D.")
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )
    mock_acompletion = AsyncMock(side_effect=[rewrite_resp, answer_resp])
    mock_completion_cost = MagicMock(side_effect=[0.0005, 0.0015])
    monkeypatch.setattr(litellm, "embedding", mock_embedding)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", mock_completion_cost)

    try:
        app = create_app()
        with TestClient(app) as client:
            doc_resp = client.post(
                "/v1/documents",
                headers={"X-OpenAI-API-Key": "sk-metrics-key"},
                json={
                    "content": "Metformin is first-line for type 2 diabetes.",
                    "metadata": {
                        "title": "Metformin",
                        "doc_type": "guideline",
                        "source_type": "markdown",
                    },
                },
            )
            assert doc_resp.status_code == 201

            query_resp = client.post(
                "/v1/query",
                headers={"X-OpenAI-API-Key": "sk-metrics-key"},
                json={
                    "query": "What is Metformin used for?",
                    "session_id": "metrics-session",
                },
            )
            assert query_resp.status_code == 200
            metrics = query_resp.json()["metrics"]
            assert metrics is not None

            latencies = metrics["latencies_ms"]
            for step in (
                "filter_documents",
                "query_rewrite",
                "seed_search",
                "k_hop_traversal",
                "answer_synthesis",
                "total",
            ):
                assert step in latencies
                assert latencies[step] >= 0.0

            usage = metrics["usage"]
            assert usage["prompt_tokens"] == 125
            assert usage["completion_tokens"] == 30
            assert usage["total_tokens"] == 155
            assert usage["estimated_cost_usd"] == pytest.approx(0.0020)
    finally:
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_document_classifier_deterministic_pii_regex_blocks() -> None:
    mock_client = MagicMock(spec=LiteLLMClient)
    mock_client.complete_with_usage = AsyncMock()
    classifier = DocumentClassifier(llm_client=mock_client, model="gpt-4o-mini")

    for text in (
        "Patient SSN: 123-45-6789 requires follow-up.",
        "MRN: A992817 admitted for chest pain.",
        "Patient Name: Jane Doe\nDOB: 1980-04-12\nAssessment: Stable.",
    ):
        result, usage = await classifier.classify_async(text=text, title="Clinical Note")
        assert result.is_allowed is False
        assert result.blocked_category == BlockedDocumentCategory.PII_PHI_DETECTED
        assert result.contains_pii_or_phi is True
        assert result.confidence == 1.0
        assert usage.total_tokens == 0

    mock_client.complete_with_usage.assert_not_awaited()


@pytest.mark.anyio
async def test_document_classifier_mocked_llm_blocks_and_allows() -> None:
    mock_client = MagicMock(spec=LiteLLMClient)
    blocked_payload = DocumentClassification(
        is_allowed=False,
        blocked_category=BlockedDocumentCategory.DISCHARGE_SUMMARY,
        contains_pii_or_phi=False,
        confidence=0.95,
        reason="Hospital discharge summary for an individual encounter.",
    ).model_dump_json()
    allowed_payload = DocumentClassification(
        is_allowed=True,
        predicted_doc_type=DocType.GUIDELINE,
        blocked_category=None,
        contains_pii_or_phi=False,
        confidence=0.98,
        reason="Published clinical practice guideline.",
    ).model_dump_json()

    mock_client.complete = AsyncMock(side_effect=[blocked_payload, allowed_payload])
    classifier = DocumentClassifier(llm_client=mock_client, model="gpt-4o-mini")

    blocked, _ = await classifier.classify_async(
        text="Hospital Course: Discharged home on day 3 in stable condition.",
        title="Discharge Summary",
        api_key="sk-cls-key",
    )
    assert blocked.is_allowed is False
    assert blocked.blocked_category == BlockedDocumentCategory.DISCHARGE_SUMMARY

    allowed, _ = await classifier.classify_async(
        text="Metformin is recommended as first-line therapy for type 2 diabetes.",
        title="ADA T2D Guideline",
        api_key="sk-cls-key",
    )
    assert allowed.is_allowed is True
    assert allowed.predicted_doc_type == DocType.GUIDELINE
    assert allowed.blocked_category is None

    assert mock_client.complete.await_count == 2
    assert mock_client.complete.await_args.kwargs["response_format"] is DocumentClassification
    assert mock_client.complete.await_args.kwargs["api_key"] == "sk-cls-key"


def test_post_documents_classify_true_rejects_422_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_dir = tmp_path / "documents"
    graph_dir = tmp_path / "graph"
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(doc_dir))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(graph_dir))
    get_settings.cache_clear()

    mock_client = MagicMock(spec=LiteLLMClient)
    blocked_record_json = DocumentClassification(
        is_allowed=False,
        blocked_category=BlockedDocumentCategory.PATIENT_RECORD,
        contains_pii_or_phi=False,
        confidence=0.94,
        reason="Individual patient record.",
    ).model_dump_json()
    mock_client.complete = AsyncMock(return_value=blocked_record_json)

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.document_classifier = DocumentClassifier(
                llm_client=mock_client,
                model="gpt-4o-mini",
            )
            manager = app.state.knowledge_manager

            # 1. Deterministic PII regex rejection
            pii_resp = client.post(
                "/v1/documents?classify=true",
                json={
                    "content": "Patient SSN 123-45-6789 presented with hypertension.",
                    "metadata": {
                        "title": "Blocked PII Doc",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                    },
                },
            )
            assert pii_resp.status_code == 422
            assert "pii_phi_detected" in pii_resp.json()["detail"]

            # 2. Mocked LLM patient_record rejection
            llm_resp = client.post(
                "/v1/documents?classify=true",
                json={
                    "content": "64yo male seen in clinic today for routine follow-up.",
                    "metadata": {
                        "title": "Clinic Encounter",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                    },
                },
            )
            assert llm_resp.status_code == 422
            assert "patient_record" in llm_resp.json()["detail"]

            # Verify nothing was persisted to DB, graph, or disk
            assert client.get("/v1/documents").json() == []
            assert manager.graph.graph.number_of_nodes() == 0
            assert not doc_dir.exists() or list(doc_dir.iterdir()) == []
    finally:
        get_settings.cache_clear()


def test_upload_document_classify_auto_populates_predicted_doc_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_dir = tmp_path / "documents"
    graph_dir = tmp_path / "graph"
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(doc_dir))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(graph_dir))
    get_settings.cache_clear()

    mock_client = MagicMock(spec=LiteLLMClient)
    allowed_drug_label_json = DocumentClassification(
        is_allowed=True,
        predicted_doc_type=DocType.DRUG_LABEL,
        contains_pii_or_phi=False,
        confidence=0.97,
        reason="FDA-style prescribing information.",
    ).model_dump_json()
    mock_client.complete = AsyncMock(return_value=allowed_drug_label_json)

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.document_classifier = DocumentClassifier(
                llm_client=mock_client,
                model="gpt-4o-mini",
            )
            manager = app.state.knowledge_manager
            manager.graph.vector_store._embedder = DeterministicHashEmbedder()

            resp = client.post(
                "/v1/documents/upload",
                files={"file": ("metformin.txt", b"Metformin hydrochloride prescribing information.", "text/plain")},
                data={"title": "Metformin Label", "classify": "true"},
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["metadata"]["doc_type"] == DocType.DRUG_LABEL.value

            stored = manager.db.get_by_id(body["doc_id"])
            assert stored is not None
            assert stored.metadata.doc_type == DocType.DRUG_LABEL
    finally:
        get_settings.cache_clear()
