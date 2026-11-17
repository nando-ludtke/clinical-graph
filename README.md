# clinical-graph

A FastAPI microservice for hybrid knowledge-graph and vector-backed clinical retrieval, document ingestion, and strictly grounded answer synthesis.

## Core Features (Implemented)

- **Document Parsing & Sub-Chunking**: Parses `.pdf` (PyMuPDF `fitz` text + `ParsedTable` with surrounding page context), `.md`, and `.txt`; strips references/boilerplate; splits text into `~1,300`-char `is_chunk` retrieval chunks grouped into `~5,000`-char extraction windows; summarizes tables via `table_summary.yml`.
- **Single-Pass GraphRAG Extraction**: `GraphExtractor` (`graph_extraction.yml`) extracts canonical clinical entities and directed relations (`relationship_description`, `relationship_strength`, verbatim `quote`, `chunk_id` mapping), rejects self-loops, and merges parallel edges with `max(weight)`.
- **Typed Vector Store & Knowledge Graph**: `NetworkX DiGraph` + typed `LocalVectorStore` (`is_node`, `is_chunk`, `is_table`), SQLite `DocumentDatabase`, and tight cross-store deletion (`delete_by_doc_id` + graph provenance pruning).
- **Hybrid Retrieval (`rag` / `hybrid` / `auto`)**: `QueryRewriter` (`v3`) expands queries and routes `auto` requests; fuses `is_chunk`/`is_table` vector search with `is_node` seed lookup + $k$-hop `DiGraph` traversal using Reciprocal Rank Fusion (RRF) and IDF-BM25 + bridge-bonus + hub-damped edge ranking.
- **Grounded Answer Synthesis & Highlighted Citations**: `AnswerAgent` (`v3`) enforces a strict no-pretraining rule, returning `SynthesizedAnswer` + `CitationQuote(ref_id, quote)` resolved into `CitationPassage` with `chunk_id` and `<mark>` highlights.
- **Optional Classification & Telemetry**: `DocumentClassifier` (`classify=true`) for PII/PHI regex screening and `DocType` allowlist validation; step-level latency, token usage, and cost telemetry (`PipelineMetrics`).

## Planned (Phase 2)

- **`clinical-guard` Middleware (Gates 1–3)**: Document ingestion screening, query entry screening, and search query formatting (`enable_guardrails`).
- **Verified Internet Search + Source Auditor**: External fallback retrieval with source credibility auditing (`enable_web_fallback`).
- **Independent Audit Agent**: Secondary verification pass over generated answers (`enable_audit_agent`).

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
