"""Coordinated KnowledgeManager synchronizing DocumentDatabase, disk storage, and KnowledgeGraph."""

from pathlib import Path
import threading

from clinical_graph.schemas.document import (
    ChunkSpan,
    DeletionReport,
    DocumentCreate,
    DocumentRecord,
    LocalRestriction,
    OmissionRecord,
    TableRecord,
)
from clinical_graph.graph.document_db import DocumentDatabase
from clinical_graph.graph.knowledge_graph import ClinicalKnowledgeGraph


class KnowledgeManager:
    """Coordinates document persistence, optional disk storage, graph provenance, and vector cleanup."""

    def __init__(
        self,
        db: DocumentDatabase | None = None,
        graph: ClinicalKnowledgeGraph | None = None,
        storage_dir: str | Path | None = None,
        graph_dir: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.storage_dir: Path | None = Path(storage_dir) if storage_dir is not None else None
        self.graph_dir: Path | None = Path(graph_dir) if graph_dir is not None else None
        if db is not None:
            self.db: DocumentDatabase = db
        elif self.graph_dir is not None:
            self.db = DocumentDatabase(db_path=self.graph_dir / "documents.db")
        else:
            self.db = DocumentDatabase()
        self.graph: ClinicalKnowledgeGraph = graph or ClinicalKnowledgeGraph()

    def save(self) -> None:
        """Persist document database and knowledge graph to ``graph_dir`` if configured."""
        with self._lock:
            if self.graph_dir is None:
                return
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            self.db.save(self.graph_dir / "documents.db")
            self.graph.save(self.graph_dir)

    def load(self) -> None:
        """Load document database and knowledge graph from ``graph_dir`` if configured."""
        with self._lock:
            if self.graph_dir is None or not self.graph_dir.exists():
                return
            self.db.load(self.graph_dir / "documents.db")
            self.graph.load(self.graph_dir)

    def ingest_document(
        self,
        payload: DocumentCreate,
        api_key: str | None = None,
    ) -> DocumentRecord:
        """Persist a document in the database, optionally on disk, and register its graph node."""
        with self._lock:
            record = self.db.add_document(payload)
            if self.storage_dir is not None:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                file_path = self.storage_dir / f"{record.doc_id}.txt"
                file_path.write_text(record.content, encoding="utf-8")
                updated = self.db.update_storage_path(record.doc_id, str(file_path))
                if updated is not None:
                    record = updated

            self.graph.upsert_node(
                node_id=record.doc_id,
                canonical_name=record.metadata.title,
                node_type="document",
                aliases=[record.slug, *record.metadata.conditions],
                description=record.metadata.description or record.content[:300],
                doc_id=record.doc_id,
                api_key=api_key,
            )
            self.save()
            return record

    def record_processing_metadata(
        self,
        doc_id: str,
        chunk_ids: list[str],
        omissions: list[OmissionRecord],
        chunk_spans: list[ChunkSpan] | None = None,
        tables: list[TableRecord] | None = None,
    ) -> DocumentRecord | None:
        """Persist chunk IDs, chunk spans, omission records, and tables on ``doc_id`` and save if ``graph_dir`` is set."""
        with self._lock:
            record = self.db.update_processing_metadata(
                doc_id=doc_id,
                chunk_ids=chunk_ids,
                omissions=omissions,
                chunk_spans=chunk_spans,
                tables=tables,
            )
            if record is not None and self.graph_dir is not None:
                self.save()
            return record

    def update_inferred_metadata(
        self,
        doc_id: str,
        *,
        conditions: list[str] | None = None,
        target_populations: list[str] | None = None,
        specialties: list[str] | None = None,
        description: str | None = None,
        local_restriction: LocalRestriction | None = None,
    ) -> DocumentRecord | None:
        """Backfill empty metadata fields on ``doc_id`` and save if ``graph_dir`` is set."""
        with self._lock:
            record = self.db.update_inferred_metadata(
                doc_id=doc_id,
                conditions=conditions,
                target_populations=target_populations,
                specialties=specialties,
                description=description,
                local_restriction=local_restriction,
            )
            if record is not None and self.graph_dir is not None:
                self.save()
            return record

    def delete_document(self, doc_id: str) -> DeletionReport:
        """Tightly delete a document from the graph, vector store, disk, and document database."""
        with self._lock:
            existing_record = self.db.get_by_id(doc_id)
            removed_nodes, removed_edges, removed_embeddings = (
                self.graph.remove_document_provenance(doc_id)
            )
            deleted_from_disk = self._delete_disk_file(doc_id, existing_record)
            deleted_from_db = self.db.delete_document(doc_id)
            self.save()
            return DeletionReport(
                doc_id=doc_id,
                deleted_from_db=deleted_from_db,
                deleted_from_disk=deleted_from_disk,
                removed_nodes=removed_nodes,
                removed_edges=removed_edges,
                removed_embeddings=removed_embeddings,
            )

    def _delete_disk_file(self, doc_id: str, record: DocumentRecord | None) -> bool:
        candidate: Path | None = None
        if record is not None and record.storage_path is not None:
            candidate = Path(record.storage_path)
        elif self.storage_dir is not None:
            candidate = self.storage_dir / f"{doc_id}.txt"

        if candidate is not None and candidate.is_file():
            candidate.unlink()
            return True
        return False
