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
from clinical_graph.api.rate_limit import SlidingWindowRateLimiter
from clinical_graph.config import Settings, get_settings
from clinical_graph.graph.vector_store import DeterministicHashEmbedder, LocalVectorStore
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.main import create_app
from clinical_graph.schemas.document import (
    DocType,
    DocumentMetadata,
    DocumentRecord,
    JobStatus,
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
            app.state.settings = app.state.settings.model_copy(
                update={"openai_api_key": None}
            )
            app.state.query_rewriter = QueryRewriter(llm_client=None)
            app.state.answer_agent = AnswerAgent(llm_client=None)
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


def test_async_background_ingestion_and_job_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_dir = tmp_path / "documents"
    graph_dir = tmp_path / "graph"
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(doc_dir))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(graph_dir))
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.knowledge_manager.graph.vector_store._embedder = DeterministicHashEmbedder()

            # 1. JSON ingestion with async_extract=true returns 202 + IngestionJob
            resp = client.post(
                "/v1/documents?async_extract=true",
                json={
                    "content": "Metformin is recommended as first-line therapy for type 2 diabetes.",
                    "metadata": {
                        "title": "Async T2D Guideline",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                    },
                },
            )
            assert resp.status_code == 202
            job_payload = resp.json()
            assert job_payload["job_id"]
            assert job_payload["title"] == "Async T2D Guideline"
            assert job_payload["status"] in {
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.COMPLETED.value,
            }

            # TestClient runs BackgroundTasks before returning, so polling returns completed
            poll_resp = client.get(f"/v1/documents/jobs/{job_payload['job_id']}")
            assert poll_resp.status_code == 200
            polled = poll_resp.json()
            assert polled["status"] == JobStatus.COMPLETED.value
            assert polled["doc_id"] is not None
            assert polled["result"]["doc_id"] == polled["doc_id"]
            assert polled["error"] is None

            # Verify document is retrievable via standard GET
            doc_resp = client.get(f"/v1/documents/{polled['doc_id']}")
            assert doc_resp.status_code == 200
            assert doc_resp.json()["metadata"]["title"] == "Async T2D Guideline"

            # Unknown job returns 404 (and is not routed as a document slug)
            missing_resp = client.get("/v1/documents/jobs/nonexistent-job-id")
            assert missing_resp.status_code == 404
    finally:
        get_settings.cache_clear()


def test_sliding_window_rate_limiter_and_http_429_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unit test SlidingWindowRateLimiter directly
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("client-a") == (True, 0)
    assert limiter.allow("client-a") == (True, 0)
    allowed, retry_after = limiter.allow("client-a")
    assert allowed is False
    assert retry_after >= 1
    assert limiter.allow("client-b") == (True, 0)

    # HTTP 429 + Retry-After header enforcement on FastAPI routes
    doc_dir = tmp_path / "documents"
    graph_dir = tmp_path / "graph"
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(doc_dir))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(graph_dir))
    monkeypatch.setenv("CLINICAL_GRAPH_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("CLINICAL_GRAPH_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("CLINICAL_GRAPH_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.knowledge_manager.graph.vector_store._embedder = DeterministicHashEmbedder()

            payload = {
                "content": "Aspirin is used for secondary prevention of cardiovascular events.",
                "metadata": {
                    "title": "Cardio Guideline",
                    "doc_type": "guideline",
                    "source_type": "raw_text",
                },
            }
            headers_a = {"X-Forwarded-For": "203.0.113.10"}
            query_payload = {"query": "Aspirin?", "session_id": "sess-1"}
            assert client.post("/v1/documents", json=payload, headers=headers_a).status_code == 201
            assert client.post("/v1/documents", json=payload, headers=headers_a).status_code == 201

            # 3rd request from same client is blocked before reaching the query pipeline
            blocked = client.post("/v1/query", json=query_payload, headers=headers_a)
            assert blocked.status_code == 429
            assert blocked.json()["detail"] == "Rate limit exceeded"
            assert int(blocked.headers["Retry-After"]) >= 1

            # Different client IP has an independent quota
            headers_b = {"X-Forwarded-For": "203.0.113.20"}
            assert client.post("/v1/documents", json=payload, headers=headers_b).status_code == 201
    finally:
        get_settings.cache_clear()


def test_system_status_and_graph_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLINICAL_GRAPH_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.settings = app.state.settings.model_copy(
                update={"openai_api_key": None}
            )
            status_resp = client.get("/v1/system/status")
            assert status_resp.status_code == 200
            assert status_resp.json() == {"has_openai_api_key": False}

            app.state.settings = app.state.settings.model_copy(
                update={"openai_api_key": "sk-configured"}
            )
            status_resp_key = client.get("/v1/system/status")
            assert status_resp_key.status_code == 200
            assert status_resp_key.json() == {"has_openai_api_key": True}

            graph = app.state.knowledge_manager.graph
            graph.vector_store._embedder = DeterministicHashEmbedder()
            graph.upsert_node(
                "metformin",
                canonical_name="Metformin",
                node_type="medication",
                aliases=["glucophage"],
                description="First-line biguanide for type 2 diabetes.",
                doc_id="doc-1",
            )
            graph.upsert_node(
                "t2d",
                canonical_name="Type 2 Diabetes",
                node_type="condition",
                doc_id="doc-1",
            )
            graph.upsert_node(
                "doc-1",
                canonical_name="Diabetes Guideline",
                node_type="document",
                doc_id="doc-1",
            )
            graph.upsert_node(
                "insulin",
                canonical_name="Insulin",
                node_type="medication",
                doc_id="doc-1",
            )
            graph.upsert_edge(
                source="metformin",
                target="t2d",
                relation="TREATS",
                doc_id="doc-1",
                excerpt="Metformin is first-line therapy for T2D.",
                weight=8.5,
                relationship_description="First-line pharmacotherapy.",
            )

            graph_resp = client.get("/v1/graph")
            assert graph_resp.status_code == 200
            payload = graph_resp.json()
            nodes_by_id = {n["node_id"]: n for n in payload["nodes"]}
            assert "metformin" in nodes_by_id
            assert "t2d" in nodes_by_id
            assert "doc-1" not in nodes_by_id
            assert "insulin" not in nodes_by_id
            assert nodes_by_id["metformin"]["canonical_name"] == "Metformin"
            assert nodes_by_id["metformin"]["aliases"] == ["glucophage"]
            assert nodes_by_id["metformin"]["doc_ids"] == ["doc-1"]

            isolated_resp = client.get("/v1/graph?include_isolated=true")
            assert isolated_resp.status_code == 200
            isolated_nodes_by_id = {n["node_id"]: n for n in isolated_resp.json()["nodes"]}
            assert "insulin" in isolated_nodes_by_id
            assert "doc-1" not in isolated_nodes_by_id

            assert len(payload["edges"]) == 1
            edge = payload["edges"][0]
            assert edge["id"] == "metformin->t2d"
            assert edge["source"] == "metformin"
            assert edge["target"] == "t2d"
            assert edge["relation"] == "TREATS"
            assert edge["weight"] == 8.5
            assert edge["doc_ids"] == ["doc-1"]
            assert len(edge["evidence"]) == 1
    finally:
        get_settings.cache_clear()


def test_delete_frozen_document_returns_403(
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
            resp = client.post(
                "/v1/documents",
                json={
                    "content": "Frozen example clinical guideline content.",
                    "metadata": {
                        "title": "Frozen Example Guideline",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                        "is_frozen": True,
                    },
                },
            )
            assert resp.status_code == 201
            doc_id = resp.json()["doc_id"]
            assert resp.json()["metadata"]["is_frozen"] is True

            del_resp = client.delete(f"/v1/documents/{doc_id}")
            assert del_resp.status_code == 403
            assert del_resp.json()["detail"] == "Frozen example document cannot be removed."
            assert client.get(f"/v1/documents/{doc_id}").status_code == 200
    finally:
        get_settings.cache_clear()


def test_job_store_update_step() -> None:
    from clinical_graph.pipelines.jobs import JobStore

    store = JobStore()
    job = store.create_job(title="Example Doc")
    assert job.current_step is None

    store.update_step(job.job_id, "Extracting nodes...")
    updated = store.get_job(job.job_id)
    assert updated is not None
    assert updated.current_step == "Extracting nodes..."
    assert updated.updated_at >= job.updated_at

    # Unknown job ID should be a safe no-op
    store.update_step("missing-job-id", "Ignored")


def test_query_stream_sse_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLINICAL_GRAPH_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        app = create_app()
        with TestClient(app) as client:
            app.state.settings = app.state.settings.model_copy(
                update={"openai_api_key": None}
            )
            app.state.query_rewriter = QueryRewriter(llm_client=None)
            app.state.answer_agent = AnswerAgent(llm_client=None)
            manager = app.state.knowledge_manager
            manager.graph.vector_store._embedder = DeterministicHashEmbedder()

            doc_resp = client.post(
                "/v1/documents",
                json={
                    "content": "Metformin is recommended as first-line therapy for type 2 diabetes.",
                    "metadata": {
                        "title": "T2D Care Standard",
                        "doc_type": "guideline",
                        "source_type": "raw_text",
                    },
                },
            )
            assert doc_resp.status_code == 201
            doc_id = doc_resp.json()["doc_id"]

            manager.graph.upsert_node(
                "metformin",
                canonical_name="Metformin",
                node_type="medication",
                doc_id=doc_id,
            )
            manager.graph.upsert_node(
                "type 2 diabetes",
                canonical_name="Type 2 Diabetes",
                node_type="condition",
                doc_id=doc_id,
            )
            manager.graph.upsert_edge(
                source="metformin",
                target="type 2 diabetes",
                relation="TREATS",
                doc_id=doc_id,
                excerpt="Metformin is recommended as first-line therapy for type 2 diabetes.",
            )

            with client.stream(
                "POST",
                "/v1/query/stream",
                json={
                    "query": "What is first-line therapy for type 2 diabetes?",
                    "session_id": "sse-test-session",
                },
            ) as stream_resp:
                assert stream_resp.status_code == 200
                assert stream_resp.headers["content-type"].startswith("text/event-stream")
                raw_body = "".join(stream_resp.iter_text())

            events: list[tuple[str, dict]] = []
            for block in raw_body.strip().split("\n\n"):
                lines = block.splitlines()
                event_name = ""
                data_raw = ""
                for line in lines:
                    if line.startswith("event: "):
                        event_name = line[len("event: ") :]
                    elif line.startswith("data: "):
                        data_raw = line[len("data: ") :]
                if event_name:
                    events.append((event_name, json.loads(data_raw)))

            names = [name for name, _ in events]
            assert names == ["rewrite", "retrieval", "synthesizing", "done"]

            events_by_name = dict(events)
            assert "search_queries" in events_by_name["rewrite"]
            assert events_by_name["rewrite"]["chosen_mode"] in ("rag", "hybrid")
            assert "chunk_count" in events_by_name["retrieval"]
            assert "seed_nodes" in events_by_name["retrieval"]
            assert "highlighted_edges" in events_by_name["retrieval"]
            assert "evidence_items" in events_by_name["retrieval"]
            assert len(events_by_name["retrieval"]["evidence_items"]) <= 10
            for item in events_by_name["retrieval"]["evidence_items"]:
                assert set(item.keys()) == {"id", "label", "source", "node_ids"}
                assert item["source"] in ("rag", "graph", "table")
            assert events_by_name["synthesizing"] == {"status": "generating_answer"}
            assert "answer" in events_by_name["done"]
            assert "citations" in events_by_name["done"]
    finally:
        get_settings.cache_clear()


def test_get_benchmarks_returns_compact_latest_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setenv("CLINICAL_GRAPH_DOCUMENT_STORAGE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CLINICAL_GRAPH_GRAPH_STORAGE_DIR", str(tmp_path / "graph"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    results_dir = tmp_path / "eval" / "results"
    results_dir.mkdir(parents=True)

    older = results_dir / "eval_20261116_010000.json"
    older.write_text(json.dumps({"timestamp": "old", "summary": {}, "results": []}), encoding="utf-8")

    newer = results_dir / "eval_20261117_020000.json"
    newer.write_text(
        json.dumps(
            {
                "timestamp": "2026-11-17T02:00:00Z",
                "queries_file": "eval/queries.json",
                "summary": {"total_queries": 1, "auto_matches": 1},
                "results": [
                    {
                        "id": "q1",
                        "query": "First-line T2D therapy?",
                        "expected_mode": "rag",
                        "rationale": "Single-fact lookup",
                        "auto_matches_expected": True,
                        "runs": {
                            "rag": {
                                "requested_mode": "rag",
                                "chosen_mode": "rag",
                                "citation_count": 2,
                                "citations": [{"doc_id": "d1"}],
                                "seed_nodes": ["n1", "n2"],
                                "traversed_edges": [{"source": "n1", "target": "n2"}],
                                "search_queries": ["metformin T2D"],
                                "assembled_context": "very long context string",
                                "latency_ms": 123.4,
                                "total_tokens": 500,
                                "cost_usd": 0.002,
                                "answer": "Metformin is first-line.",
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/v1/benchmarks")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["available"] is True
            assert payload["filename"] == "eval_20261117_020000.json"
            assert payload["timestamp"] == "2026-11-17T02:00:00Z"
            assert payload["queries_file"] == "eval/queries.json"
            assert payload["summary"] == {"total_queries": 1, "auto_matches": 1}
            assert len(payload["results"]) == 1

            rag_run = payload["results"][0]["runs"]["rag"]
            assert rag_run["seed_node_count"] == 2
            assert rag_run["traversed_edge_count"] == 1
            assert rag_run["citation_count"] == 2
            assert rag_run["search_queries"] == ["metformin T2D"]
            assert rag_run["answer"] == "Metformin is first-line."
            assert "assembled_context" not in rag_run
            assert "citations" not in rag_run
    finally:
        get_settings.cache_clear()
