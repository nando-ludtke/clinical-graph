# clinical-graph

A FastAPI microservice for hybrid knowledge-graph and vector-backed clinical retrieval, document ingestion, and strictly grounded answer synthesis.

## Core Features (Implemented)

- **Document Parsing & Sub-Chunking**: Parses `.pdf` (PyMuPDF `fitz` text + `ParsedTable` with surrounding page context), `.md`, and `.txt`; strips references/boilerplate; splits text into `~1,300`-char `is_chunk` retrieval chunks grouped into `~5,000`-char extraction windows; summarizes tables via `table_summary.yml`.
- **Single-Pass GraphRAG Extraction**: `GraphExtractor` (`graph_extraction.yml`) wraps extraction windows in `<source_data>` delimiters, extracts canonical clinical entities and directed relations (`relationship_description`, `relationship_strength`, verbatim `quote`, `chunk_id` mapping), rejects self-loops, and merges parallel edges with `max(weight)`.
- **Typed Vector Store, Knowledge Graph & Showcase Lifecycle**: `NetworkX DiGraph` + typed `LocalVectorStore` (`is_node`, `is_chunk`, `is_table`), SQLite `DocumentDatabase`, tight cross-store deletion (`delete_by_doc_id` + graph provenance pruning), frozen showcase documents (`is_frozen`), and `AUTO_PRUNE_NON_FROZEN_HOURS` (24h showcase auto-cleanup of non-frozen uploads).
- **Hybrid Retrieval (`rag` / `hybrid` / `auto`)**: `QueryRewriter` (`v4`) expands queries, routes `auto` requests, and applies `is_medical` query gating; embeds rewritten `search_queries` in a single batch (`embed_queries`) reused across `is_chunk`, `is_table`, and `is_node` searches; fuses `is_chunk`/`is_table` RRF results with $k$-hop `DiGraph` traversal using hop-decayed IDF-BM25 + bridge-bonus + hub-damped edge ranking and elastic two-channel chunk assembly (`chunk_sources`: `rag`, `graph`, `both`).
- **Grounded Answer Synthesis & Highlighted Citations**: `AnswerAgent` (`v4`) wraps context passages in `<source_data>` delimiters and enforces a strict no-pretraining rule, returning `SynthesizedAnswer` + `CitationQuote(ref_id, quote)` resolved into `CitationPassage` with `chunk_id` and `<mark>` highlights.
- **`clinical-guard` Prompt-Injection Screening**: `ClinicalGuardClient` (`guardrails/`) screens queries and document uploads regex-first (`scan_text_for_injection`) and `clinical-guard` second (`POST /v1/guard/prompt`), with `GET /health` periodic polling, `X-Forwarded-For` forwarding, `Semaphore(2)` concurrent window scanning, and `429`/`503` `Retry-After` backoff.
- **Classification, Telemetry, Benchmarks & Web UI**: `DocumentClassifier` (`classify=true`) for PII/PHI regex screening and `DocType` allowlist validation; step-level latency, token usage, and cost telemetry (`PipelineMetrics`); `GET /v1/benchmarks` telemetry endpoint; and a 2-view (`Overview` + `Workspace`) static frontend.
- **Verified Internet Search (`search/`)**: 1-shot Tavily allowlisted web fallback (`enable_web_fallback`, with explicit UI confirmation modal) triggered via `node_label` similarity or OpenAI Decisions (`gpt-6-luna`), with temporary vector-chunk RAG and `< 1 ms` local regex + `<source_data>` delimiting (see [docs/web_search.md](docs/web_search.md)).

## Planned (Phase 2)

- **Independent Audit Agent**: Secondary verification pass over generated answers.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Run the API Server

```bash
uvicorn clinical_graph.main:app --reload
```

Or run via Docker Compose:

```bash
docker compose up --build
```

### Run Live E2E Script & Retrieval Evaluation

```bash
# Ingest a document and/or run a live query
python scripts/run_live_e2e.py --ingest path/to/guideline.pdf --query "First-line therapy for T2D?"

# Benchmark rag vs. hybrid vs. auto retrieval modes
python eval/run_eval.py
```

### Run Tests

```bash
.venv/bin/pytest -q
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Document Parsing & Sub-Chunking](docs/document_parsing.md)
- [GraphRAG & Document Handling](docs/graphrag_and_documents.md)
- [Query & Retrieval Pipeline](docs/query_and_retrieval.md)

## License

Licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license.
