"""Framework-agnostic document ingestion pipeline."""

import asyncio
from collections.abc import Callable
import time
from typing import Any

from pydantic import BaseModel

from clinical_graph.agents.classifier import DocumentClassification, DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.parser import DocumentParser, ParsedTable
from clinical_graph.schemas.document import (
    DeletionReport,
    DocType,
    DocumentCreate,
    DocumentMetadata,
    DocumentRecord,
    LocalRestriction,
)
from clinical_graph.schemas.metrics import LLMUsage, PipelineMetrics


class IngestionResult(BaseModel):
    """Result payload from document ingestion including telemetry and extraction counts."""

    record: DocumentRecord
    metrics: PipelineMetrics
    extracted_node_count: int = 0
    extracted_edge_count: int = 0


class DocumentRejectedError(Exception):
    """Raised when ``DocumentClassifier`` rejects a document during ingestion."""

    def __init__(self, classification: DocumentClassification, detail: str) -> None:
        super().__init__(detail)
        self.classification = classification
        self.detail = detail


def _backfill_metadata_from_classification(
    payload: DocumentCreate,
    classification: DocumentClassification,
) -> DocumentCreate:
    """Populate empty metadata fields from classifier inferences."""
    meta = payload.metadata
    updates: dict[str, Any] = {}
    if not meta.conditions and classification.inferred_conditions:
        normalized = [
            item.strip().lower() for item in classification.inferred_conditions if item.strip()
        ]
        if normalized:
            updates["conditions"] = list(dict.fromkeys(normalized))
    if not meta.target_populations and classification.inferred_target_populations:
        normalized = [
            item.strip().lower()
            for item in classification.inferred_target_populations
            if item.strip()
        ]
        if normalized:
            updates["target_populations"] = list(dict.fromkeys(normalized))
    if not meta.specialties and classification.inferred_specialties:
        normalized = [
            item.strip().lower() for item in classification.inferred_specialties if item.strip()
        ]
        if normalized:
            updates["specialties"] = list(dict.fromkeys(normalized))
    if not meta.description and classification.inferred_description:
        cleaned = classification.inferred_description.strip()
        if cleaned:
            updates["description"] = cleaned
    if (
        meta.local_restriction == LocalRestriction.NONE
        and classification.inferred_local_restriction is not None
        and classification.inferred_local_restriction != LocalRestriction.NONE
    ):
        updates["local_restriction"] = classification.inferred_local_restriction
    if not updates:
        return payload
    return payload.model_copy(update={"metadata": meta.model_copy(update=updates)})


def _emit(on_step: Callable[[str], None] | None, message: str) -> None:
    if on_step is not None:
        on_step(message)


class IngestionPipeline:
    """Orchestrates parsing, optional classification, graph extraction, and deletion."""

    def __init__(
        self,
        manager: KnowledgeManager,
        parser: DocumentParser | None = None,
        classifier: DocumentClassifier | None = None,
        extractor: GraphExtractor | None = None,
    ) -> None:
        self.manager = manager
        self.parser = parser or DocumentParser()
        self.classifier = classifier or DocumentClassifier()
        self.extractor = extractor or GraphExtractor()

    async def ingest_document_with_report(
        self,
        payload: DocumentCreate,
        *,
        classify: bool = False,
        override_doc_type: bool = False,
        extract: bool = False,
        parsed_tables: list[ParsedTable] | None = None,
        api_key: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> IngestionResult:
        """Optionally classify, then ingest (and optionally extract) a document with metrics."""
        total_t0 = time.perf_counter()
        latencies_ms: dict[str, float] = {}
        usage = LLMUsage()

        if classify:
            _emit(on_step, "[ingest] Classifying document...")
            t0 = time.perf_counter()
            classification, classify_usage = await self.classifier.classify_async(
                text=payload.content,
                title=payload.metadata.title,
                api_key=api_key,
            )
            latencies_ms["classify_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            usage = usage.add(classify_usage)
            if not classification.is_allowed:
                category = (
                    classification.blocked_category.value
                    if classification.blocked_category is not None
                    else "disallowed_document"
                )
                detail = (
                    f"Document rejected by classifier ({category}): {classification.reason}"
                    if classification.reason
                    else f"Document rejected by classifier ({category})."
                )
                raise DocumentRejectedError(classification=classification, detail=detail)

            if override_doc_type and classification.predicted_doc_type is not None:
                payload = payload.model_copy(
                    update={
                        "metadata": payload.metadata.model_copy(
                            update={"doc_type": classification.predicted_doc_type}
                        )
                    }
                )
            payload = _backfill_metadata_from_classification(payload, classification)
            doc_type_label = (
                classification.predicted_doc_type or payload.metadata.doc_type
            ).value
            _emit(
                on_step,
                f"[ingest] Classified ({doc_type_label}) in {latencies_ms['classify_ms']:.2f} ms",
            )

        if not extract:
            t0 = time.perf_counter()
            record = await asyncio.to_thread(
                self.manager.ingest_document, payload, api_key=api_key
            )
            latencies_ms["ingest_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            latencies_ms["total_ms"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
            _emit(
                on_step,
                f"[ingest] Persisted document {record.doc_id} in {latencies_ms['total_ms']:.2f} ms total",
            )
            return IngestionResult(
                record=record,
                metrics=PipelineMetrics(latencies_ms=latencies_ms, usage=usage),
            )

        _emit(on_step, "[ingest] Extracting nodes/edges & embedding...")
        t0 = time.perf_counter()
        record, extraction_result = await self.extractor.ingest_with_extraction(
            manager=self.manager,
            payload=payload,
            parsed_tables=parsed_tables,
            api_key=api_key,
            on_step=on_step,
        )
        latencies_ms["extract_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        usage = usage.add(extraction_result.metrics.usage)
        node_count = len(extraction_result.nodes)
        edge_count = len(extraction_result.edges)
        _emit(
            on_step,
            f"[ingest] Extracted {node_count} nodes, {edge_count} edges in {latencies_ms['extract_ms']:.2f} ms",
        )

        t0 = time.perf_counter()
        await asyncio.to_thread(self.manager.save)
        latencies_ms["ingest_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        latencies_ms["total_ms"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
        _emit(
            on_step,
            f"[ingest] Persisted document {record.doc_id} in {latencies_ms['total_ms']:.2f} ms total",
        )

        return IngestionResult(
            record=record,
            metrics=PipelineMetrics(latencies_ms=latencies_ms, usage=usage),
            extracted_node_count=node_count,
            extracted_edge_count=edge_count,
        )

    async def ingest_document(
        self,
        payload: DocumentCreate,
        *,
        classify: bool = False,
        override_doc_type: bool = False,
        extract: bool = False,
        api_key: str | None = None,
    ) -> DocumentRecord:
        """Optionally classify, then ingest (and optionally extract) a document."""
        result = await self.ingest_document_with_report(
            payload,
            classify=classify,
            override_doc_type=override_doc_type,
            extract=extract,
            api_key=api_key,
        )
        return result.record

    async def ingest_file_bytes_with_report(
        self,
        data: bytes,
        filename: str,
        *,
        title: str,
        doc_type: DocType | None = None,
        local_restriction: LocalRestriction = LocalRestriction.NONE,
        description: str | None = None,
        conditions: list[str] | None = None,
        target_populations: list[str] | None = None,
        specialties: list[str] | None = None,
        is_frozen: bool = False,
        classify: bool = False,
        extract: bool = False,
        api_key: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> IngestionResult:
        """Parse raw file bytes into a ``DocumentCreate``, ingest it, and return telemetry."""
        total_t0 = time.perf_counter()
        _emit(on_step, f"[ingest] Parsing file {filename}...")
        t0 = time.perf_counter()
        parsed = self.parser.parse_bytes(data, filename)
        parse_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        parts = [part for part in (parsed.text.strip(), *parsed.tables) if part]
        content = "\n\n".join(parts)
        if not content:
            raise ValueError("Uploaded file contains no extractable text or tables.")
        _emit(on_step, f"[ingest] Parsed {filename} ({len(content)} chars) in {parse_ms:.2f} ms")

        effective_doc_type = doc_type if doc_type is not None else DocType.GUIDELINE
        payload = DocumentCreate(
            content=content,
            metadata=DocumentMetadata(
                title=title,
                doc_type=effective_doc_type,
                source_type=parsed.source_type,
                local_restriction=local_restriction,
                description=description,
                conditions=conditions or [],
                target_populations=target_populations or [],
                specialties=specialties or [],
                is_frozen=is_frozen,
            ),
        )
        result = await self.ingest_document_with_report(
            payload,
            classify=classify,
            override_doc_type=doc_type is None,
            extract=extract,
            parsed_tables=parsed.table_entries,
            api_key=api_key,
            on_step=on_step,
        )
        latencies_ms = {"parse_ms": parse_ms, **result.metrics.latencies_ms}
        latencies_ms["total_ms"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
        return result.model_copy(
            update={
                "metrics": result.metrics.model_copy(update={"latencies_ms": latencies_ms})
            }
        )

    async def ingest_file_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        title: str,
        doc_type: DocType | None = None,
        local_restriction: LocalRestriction = LocalRestriction.NONE,
        description: str | None = None,
        conditions: list[str] | None = None,
        target_populations: list[str] | None = None,
        specialties: list[str] | None = None,
        is_frozen: bool = False,
        classify: bool = False,
        extract: bool = False,
        api_key: str | None = None,
    ) -> DocumentRecord:
        """Parse raw file bytes into a ``DocumentCreate`` and ingest it."""
        result = await self.ingest_file_bytes_with_report(
            data,
            filename,
            title=title,
            doc_type=doc_type,
            local_restriction=local_restriction,
            description=description,
            conditions=conditions,
            target_populations=target_populations,
            specialties=specialties,
            is_frozen=is_frozen,
            classify=classify,
            extract=extract,
            api_key=api_key,
        )
        return result.record

    async def delete_document(self, doc_id: str) -> DeletionReport:
        """Tightly delete a document from the knowledge graph, vector store, and database."""
        return await asyncio.to_thread(self.manager.delete_document, doc_id)
