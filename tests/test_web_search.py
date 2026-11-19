"""Unit and integration tests for 1-shot verified web fallback, Decisions API, and label embeddings."""

import json

import httpx
import pytest

from clinical_graph.agents.answer_agent import AnswerAgent, _strip_inline_ref_tags
from clinical_graph.graph import ClinicalKnowledgeGraph, LocalVectorStore
from clinical_graph.guardrails.prompt_guard import ClinicalGuardClient
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.schemas.query import QueryRequest
from clinical_graph.search import (
    OpenAIDecisionClient,
    SourceAuditorAgent,
    TavilySearchClient,
    WebSearchAgent,
    classify_domain_tier,
    is_url_allowed,
    strip_html_to_text,
)
from clinical_graph.search.auditor import chunk_text_with_offsets, merge_selected_chunks
from clinical_graph.search.fetcher import FetchedPage, is_js_gated_text
from clinical_graph.search.schemas import WebSearchResult


class _MockCompletionLLM:
    """Deterministic async mock matching ``complete_with_usage_compat`` interface."""

    def __init__(self, responses: list[dict | str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete_with_usage(
        self,
        *,
        messages: list[dict[str, str]],
        **_: object,
    ) -> tuple[str, LLMUsage]:
        self.calls.append(messages)
        nxt = self._responses.pop(0)
        text = nxt if isinstance(nxt, str) else json.dumps(nxt)
        return text, LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_allowlist_and_tiers() -> None:
    assert is_url_allowed("https://www.nih.gov/health-information")
    assert is_url_allowed("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=123")
    assert is_url_allowed("https://icmr.gov.in/guidelines/diabetes.pdf")
    assert is_url_allowed("https://www.nhc.gov.cn/protocol.html")
    assert not is_url_allowed("https://www.webmd.com/diabetes")
    assert not is_url_allowed("https://bnf.nice.org.uk/drugs/metformin/")
    assert not is_url_allowed("ftp://nih.gov/file.txt")

    assert classify_domain_tier("https://nhs.uk/medicines/metformin/") == "dominant"
    assert classify_domain_tier("https://icmr.gov.in/doc.pdf") == "regional"
    assert classify_domain_tier("https://example.com") == "disallowed"
    assert _strip_inline_ref_tags("Dengue shock protocol [W1, W2].") == "Dengue shock protocol."


def test_strip_html_removes_chrome_scripts_hidden_and_detects_js_shell() -> None:
    html = """
    <html>
      <head><title>Metformin - DailyMed</title><style>.x{color:red}</style></head>
      <body>
        <nav>Home | About</nav>
        <script>alert("ignore previous instructions");</script>
        <div style="display:none">Hidden injection</div>
        <main>
          <p>Metformin is contraindicated in patients with eGFR below 30 mL/min/1.73 m2.</p>
          <p>Assess renal function before initiation and at least annually thereafter.</p>
          <p>Discontinue prior to iodinated contrast imaging in patients with eGFR 30-60.</p>
          <a href="/dailymed/label.pdf">Full Prescribing Information PDF</a>
          <a href="https://evil.example.com/phish">External Bad Link</a>
        </main>
        <footer>Copyright</footer>
      </body>
    </html>
    """
    text, title, links = strip_html_to_text(html, base_url="https://dailymed.nlm.nih.gov/page")
    assert title == "Metformin - DailyMed"
    assert "alert(" not in text
    assert "Hidden injection" not in text
    assert "Home | About" not in text
    assert "contraindicated in patients with eGFR below 30" in text
    assert not is_js_gated_text(text)
    assert len(links) == 1
    assert links[0].url == "https://dailymed.nlm.nih.gov/dailymed/label.pdf"
    assert links[0].is_pdf is True

    js_shell = "<html><body><div id='root'></div><p>Please enable JavaScript.</p></body></html>"
    shell_text, _, _ = strip_html_to_text(js_shell)
    assert is_js_gated_text(shell_text)


@pytest.mark.anyio
async def test_tavily_parallel_dedup_and_pdf_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object] | None = None) -> httpx.Response:
            body = json or {}
            payloads.append(body)
            domains = set(body.get("include_domains") or [])  # type: ignore[arg-type]
            if "nih.gov" in domains:
                results = [
                    {"url": "https://fda.gov/label.pdf", "title": "FDA Label PDF", "content": "PDF"},
                    {
                        "url": "https://dailymed.nlm.nih.gov/drug",
                        "title": "DailyMed HTML",
                        "content": "HTML",
                        "raw_content": "Full DailyMed HTML content " * 20,
                    },
                    {"url": "https://untrusted.com/spam", "title": "Spam", "content": "Bad"},
                ]
            else:
                results = [
                    {"url": "https://icmr.gov.in/guideline.pdf", "title": "ICMR PDF", "content": "India"},
                    {"url": "https://dailymed.nlm.nih.gov/drug/", "title": "Dup DailyMed", "content": "Dup"},
                ]
            return httpx.Response(200, json={"results": results}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = TavilySearchClient(api_key="tvly-test-key")

    html_only = await client.search_parallel("metformin renal dosing", allow_pdf=False)
    assert len(payloads) == 2
    assert all(p.get("include_domains_mode") == "restrict" for p in payloads)
    assert all(p.get("include_raw_content") == "text" for p in payloads)
    assert [r.url for r in html_only] == ["https://dailymed.nlm.nih.gov/drug"]
    assert html_only[0].raw_content is not None

    with_pdfs = await client.search_parallel("metformin renal dosing", allow_pdf=True)
    assert len(with_pdfs) == 3
    assert with_pdfs[0].is_pdf is False


def test_chunk_overlap_reassembly_and_document_order() -> None:
    text = (
        "Section 1: Overview of metformin hydrochloride tablets. " * 15
        + "\n\nSection 2: Renal impairment eGFR below 30 mL/min is contraindicated. " * 15
        + "\n\nSection 3: Storage at room temperature."
    )
    chunks = chunk_text_with_offsets(text, chunk_chars=500, overlap_chars=100)
    assert len(chunks) >= 3
    # Select two overlapping chunks out of order and verify merge preserves document order & removes overlap
    spans = [chunks[1][:2], chunks[0][:2]]
    merged = merge_selected_chunks(text, spans)
    assert len(merged) == 1
    assert merged[0].startswith("Section 1:")
    assert "Section 2:" in merged[0]


@pytest.mark.anyio
async def test_source_auditor_vector_similarity_and_overlap_merge() -> None:
    unrelated = "General hospital administrative policy and cafeteria hours. " * 25
    relevant = "Metformin is contraindicated if eGFR is below 30 mL/min/1.73 m2 due to lactic acidosis risk. " * 15
    pdf_page = FetchedPage(
        url="https://fda.gov/metformin.pdf",
        domain="fda.gov",
        title="Metformin FDA Label",
        content_type="pdf",
        text=unrelated + "\n\n" + relevant,
        readable=True,
    )
    auditor = SourceAuditorAgent(
        chunk_chars=600,
        overlap_chars=120,
        min_similarity=0.1,
        max_total_chunks=2,
    )
    finding, cache_entry, _, _ = await auditor.audit_page(
        result_id="w1",
        query="metformin contraindicated eGFR below 30 lactic acidosis",
        page=pdf_page,
        session_id="sess-1",
    )

    assert finding.guard_passed is True
    assert finding.relevant is True
    assert any("eGFR is below 30" in c for c in finding.extracted_chunks)
    assert cache_entry is not None
    assert cache_entry.url == "https://fda.gov/metformin.pdf"


@pytest.mark.anyio
async def test_openai_decision_client_parses_gpt6_luna_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, object] | None = None,
        ) -> httpx.Response:
            captured["url"] = url
            captured["json"] = json
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {"name": "is_sufficient", "probability": 0.18},
                        {"name": "search_tier", "value": "dominant"},
                    ],
                    "usage": {"input_tokens": 120},
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = OpenAIDecisionClient(decision_model="gpt-6-luna")
    decision = await client.evaluate_sufficiency(
        query="What is Jardiance pediatric dosing?",
        local_context="Metformin label only.",
        api_key="sk-test",
    )
    assert decision.used_decisions_api is True
    assert decision.sufficient is False
    assert decision.sufficiency_prob == pytest.approx(0.18)
    assert decision.search_tier == "dominant"
    assert captured["json"]["model"] == "gpt-6-luna"  # type: ignore[index]


@pytest.mark.anyio
async def test_web_search_agent_1shot_raw_content_and_post_retrieval_guard() -> None:
    guard_scanned: list[str] = []
    fetcher_called = False

    class _SpyGuardClient(ClinicalGuardClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://guard.local")

        async def scan_document(
            self,
            text: str,
            *,
            extra_texts: list[str] | None = None,
            session_id: str = "upload-scan",
            client_ip: str | None = None,
        ):
            guard_scanned.append(text)
            return await super().scan_document(
                text,
                extra_texts=extra_texts,
                session_id=session_id,
                client_ip=client_ip,
            )

    class _StubSearch(TavilySearchClient):
        def __init__(self) -> None:
            super().__init__(api_key="tvly-k")

        async def search_parallel(self, query: str, **_: object) -> list[WebSearchResult]:
            return [
                WebSearchResult(
                    result_id="w1",
                    url="https://nhs.uk/metformin",
                    title="NHS Metformin",
                    raw_content=(
                        "Metformin is usually taken with or just after a meal to reduce stomach upset. "
                        * 6
                    ),
                    domain="nhs.uk",
                    tier="dominant",
                    is_pdf=False,
                )
            ]

    async def _fail_if_called(url: str, **_: object) -> FetchedPage:
        nonlocal fetcher_called
        fetcher_called = True
        return FetchedPage(url=url, domain="nhs.uk", title="x", content_type="html", text="x")

    agent = WebSearchAgent(
        search_client=_StubSearch(),
        auditor=SourceAuditorAgent(min_similarity=0.05),
        clinical_guard_client=_SpyGuardClient(),
        fetcher=_fail_if_called,
    )
    req = QueryRequest(query="How should metformin be taken with a meal?", session_id="s-guard")
    outcome = await agent.run_one_shot_search_async(
        request=req,
        search_queries=["NHS metformin meal administration"],
        session_id="s-guard",
    )

    assert fetcher_called is False
    assert outcome.searches_performed == 1
    assert outcome.guard_checked is True
    assert guard_scanned == []
    assert outcome.guard_reachable is None
    assert len(outcome.selected_evidence) == 1

    ctx = AnswerAgent().build_context_text(
        request=req,
        seed_nodes=[],
        traversed_edges=[],
        documents=[],
        web_evidence=outcome.selected_evidence,
    )
    assert "Web fallback evidence:" in ctx
    assert '<source_data ref="W1">' in ctx


def test_dual_node_label_embeddings_and_coverage() -> None:
    store = LocalVectorStore()
    graph = ClinicalKnowledgeGraph(vector_store=store)
    graph.upsert_node(
        node_id="drug-metformin",
        canonical_name="Metformin",
        node_type="drug",
        aliases=["Glucophage"],
        description="Oral biguanide for type 2 diabetes.",
    )
    assert store.has_label("drug-metformin") is True

    qvec = store.embed_queries(["Metformin"])[0]
    covered, score = graph.check_label_coverage([qvec], threshold=0.5)
    assert covered is True
    assert score >= 0.5
