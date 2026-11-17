"""SQLite-backed document database for clinical document IDs, slugs, and metadata."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from clinical_graph.schemas.document import (
    ChunkSpan,
    DocumentCreate,
    DocumentFilter,
    DocumentMetadata,
    DocumentRecord,
    LocalRestriction,
    OmissionRecord,
    TableRecord,
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    source_type TEXT NOT NULL,
    local_restriction TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    chunk_ids_json TEXT NOT NULL,
    chunk_spans_json TEXT NOT NULL DEFAULT '[]',
    omissions_json TEXT NOT NULL,
    tables_json TEXT NOT NULL DEFAULT '[]',
    storage_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents (doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents (source_type);
CREATE INDEX IF NOT EXISTS idx_documents_local_restriction ON documents (local_restriction);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
"""


def slugify(text: str) -> str:
    """Convert arbitrary text into a lowercase, URL- and citation-safe slug."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALNUM_RE.sub("-", normalized.lower()).strip("-")
    return slug or "doc"


def _resolve_documents_path(path: Path) -> Path:
    return path if path.suffix == ".db" else path / "documents.db"


def _matches_any_normalized(requested: list[str] | None, actual: list[str]) -> bool:
    """Return True if no filter is set or at least one requested value matches."""
    if requested is None:
        return True
    req_set = {item.strip().lower() for item in requested if item.strip()}
    return bool(req_set & set(actual))


class DocumentDatabase:
    """SQLite-backed document store indexed by primary ``doc_id`` and unique ``slug``."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None or str(db_path) == ":memory:":
            self._db_path: Path | None = None
        else:
            self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = self._open_connection(self._db_path)
        self._init_schema()

    def _open_connection(self, db_path: Path | None) -> sqlite3.Connection:
        if db_path is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DocumentDatabase connection is closed")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._require_conn()
        conn.executescript(_SCHEMA_SQL)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "chunk_spans_json" not in columns:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN chunk_spans_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "tables_json" not in columns:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN tables_json TEXT NOT NULL DEFAULT '[]'"
            )
        conn.commit()

    def add_document(
        self,
        payload: DocumentCreate,
        storage_path: str | None = None,
    ) -> DocumentRecord:
        """Persist a new document with generated ``doc_id``, unique ``slug``, hash, and timestamps."""
        conn = self._require_conn()
        content_hash = hashlib.sha256(payload.content.encode("utf-8")).hexdigest()
        doc_id = self._unique_doc_id(f"doc_{content_hash[:12]}")

        base_slug_input = payload.slug
        if not base_slug_input:
            if payload.metadata.effective_year is not None:
                base_slug_input = f"{payload.metadata.title} {payload.metadata.effective_year}"
            else:
                base_slug_input = payload.metadata.title
        slug = self._unique_slug(slugify(base_slug_input))

        now = datetime.now(UTC)
        metadata = payload.metadata.model_copy(update={"content_hash": content_hash})
        record = DocumentRecord(
            doc_id=doc_id,
            slug=slug,
            content=payload.content,
            metadata=metadata,
            storage_path=storage_path,
            created_at=now,
            updated_at=now,
        )
        conn.execute(
            """
            INSERT INTO documents (
                doc_id, slug, content, doc_type, source_type, local_restriction,
                status, metadata_json, chunk_ids_json, chunk_spans_json, omissions_json,
                tables_json, storage_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.doc_id,
                record.slug,
                record.content,
                metadata.doc_type.value,
                metadata.source_type.value,
                metadata.local_restriction.value,
                metadata.status.value,
                metadata.model_dump_json(),
                json.dumps(record.chunk_ids),
                json.dumps([s.model_dump(mode="json") for s in record.chunk_spans]),
                json.dumps([o.model_dump(mode="json") for o in record.omissions]),
                json.dumps([t.model_dump(mode="json") for t in record.tables]),
                record.storage_path,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )
        conn.commit()
        return record

    def get_by_id(self, doc_id: str) -> DocumentRecord | None:
        """Return a document record by its ``doc_id``, or ``None`` if not found."""
        conn = self._require_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_by_slug(self, slug: str) -> DocumentRecord | None:
        """Return a document record by its unique ``slug``, or ``None`` if not found."""
        conn = self._require_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE slug = ?", (slug,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def update_storage_path(
        self,
        doc_id: str,
        storage_path: str | None,
    ) -> DocumentRecord | None:
        """Persist ``storage_path`` on an existing document record."""
        conn = self._require_conn()
        cursor = conn.execute(
            "UPDATE documents SET storage_path = ? WHERE doc_id = ?",
            (storage_path, doc_id),
        )
        if cursor.rowcount == 0:
            return None
        conn.commit()
        return self.get_by_id(doc_id)

    def update_processing_metadata(
        self,
        doc_id: str,
        chunk_ids: list[str],
        omissions: list[OmissionRecord],
        chunk_spans: list[ChunkSpan] | None = None,
        tables: list[TableRecord] | None = None,
    ) -> DocumentRecord | None:
        """Attach chunk IDs, chunk spans, omission records, and tables to an existing document record."""
        existing = self.get_by_id(doc_id)
        if existing is None:
            return None
        conn = self._require_conn()
        now = datetime.now(UTC)
        new_chunk_ids = list(chunk_ids)
        new_chunk_spans = (
            list(chunk_spans) if chunk_spans is not None else list(existing.chunk_spans)
        )
        new_omissions = list(omissions)
        new_tables = list(tables) if tables is not None else list(existing.tables)
        conn.execute(
            """
            UPDATE documents
            SET chunk_ids_json = ?, chunk_spans_json = ?, omissions_json = ?, tables_json = ?, updated_at = ?
            WHERE doc_id = ?
            """,
            (
                json.dumps(new_chunk_ids),
                json.dumps([s.model_dump(mode="json") for s in new_chunk_spans]),
                json.dumps([o.model_dump(mode="json") for o in new_omissions]),
                json.dumps([t.model_dump(mode="json") for t in new_tables]),
                now.isoformat(),
                doc_id,
            ),
        )
        conn.commit()
        return existing.model_copy(
            update={
                "chunk_ids": new_chunk_ids,
                "chunk_spans": new_chunk_spans,
                "omissions": new_omissions,
                "tables": new_tables,
                "updated_at": now,
            }
        )

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
        """Backfill empty metadata fields on an existing document record."""
        existing = self.get_by_id(doc_id)
        if existing is None:
            return None

        meta = existing.metadata
        updates: dict[str, object] = {}

        if conditions and not meta.conditions:
            normalized = [item.strip().lower() for item in conditions if item.strip()]
            if normalized:
                updates["conditions"] = list(dict.fromkeys(normalized))
        if target_populations and not meta.target_populations:
            normalized = [item.strip().lower() for item in target_populations if item.strip()]
            if normalized:
                updates["target_populations"] = list(dict.fromkeys(normalized))
        if specialties and not meta.specialties:
            normalized = [item.strip().lower() for item in specialties if item.strip()]
            if normalized:
                updates["specialties"] = list(dict.fromkeys(normalized))
        if description and not meta.description:
            cleaned_desc = description.strip()
            if cleaned_desc:
                updates["description"] = cleaned_desc
        if (
            local_restriction is not None
            and meta.local_restriction == LocalRestriction.NONE
            and local_restriction != LocalRestriction.NONE
        ):
            updates["local_restriction"] = local_restriction

        if not updates:
            return existing

        new_meta = meta.model_copy(update=updates)
        now = datetime.now(UTC)
        conn = self._require_conn()
        conn.execute(
            """
            UPDATE documents
            SET local_restriction = ?, metadata_json = ?, updated_at = ?
            WHERE doc_id = ?
            """,
            (
                new_meta.local_restriction.value,
                new_meta.model_dump_json(),
                now.isoformat(),
                doc_id,
            ),
        )
        conn.commit()
        return existing.model_copy(update={"metadata": new_meta, "updated_at": now})

    def filter_documents(self, filters: DocumentFilter | None = None) -> list[DocumentRecord]:
        """Return documents matching all specified facets (AND semantics)."""
        conn = self._require_conn()
        if filters is None:
            rows = conn.execute("SELECT * FROM documents ORDER BY rowid").fetchall()
            return [self._row_to_record(row) for row in rows]

        clauses: list[str] = []
        params: list[str] = []

        for col, values in (
            ("doc_type", filters.doc_types),
            ("source_type", filters.source_types),
            ("local_restriction", filters.local_restrictions),
        ):
            if values is not None:
                if not values:
                    return []
                placeholders = ", ".join("?" for _ in values)
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(item.value for item in values)

        if filters.status is not None:
            clauses.append("status = ?")
            params.append(filters.status.value)

        sql = "SELECT * FROM documents"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"

        rows = conn.execute(sql, params).fetchall()
        results: list[DocumentRecord] = []
        for row in rows:
            record = self._row_to_record(row)
            meta = record.metadata
            if not _matches_any_normalized(filters.conditions, meta.conditions):
                continue
            if not _matches_any_normalized(filters.target_populations, meta.target_populations):
                continue
            if not _matches_any_normalized(filters.specialties, meta.specialties):
                continue
            results.append(record)
        return results

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ``doc_id``. Returns ``True`` if deleted, ``False`` otherwise."""
        conn = self._require_conn()
        cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of stored documents."""
        conn = self._require_conn()
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row is not None else 0

    def save(self, path: Path) -> None:
        """Commit changes and persist the SQLite database to ``path`` if different."""
        conn = self._require_conn()
        target = _resolve_documents_path(Path(path))
        conn.commit()
        if self._db_path is not None and self._db_path.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        dst = sqlite3.connect(str(target))
        try:
            conn.backup(dst)
            dst.commit()
        finally:
            dst.close()

    def load(self, path: Path) -> None:
        """Load or reopen the SQLite database from ``path`` if it exists."""
        target = _resolve_documents_path(Path(path))
        if not target.is_file():
            return
        self.close()
        self._db_path = target
        self._conn = self._open_connection(target)
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection cleanly."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _doc_id_exists(self, doc_id: str) -> bool:
        conn = self._require_conn()
        row = conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row is not None

    def _slug_exists(self, slug: str) -> bool:
        conn = self._require_conn()
        row = conn.execute(
            "SELECT 1 FROM documents WHERE slug = ? LIMIT 1", (slug,)
        ).fetchone()
        return row is not None

    def _unique_doc_id(self, base_id: str) -> str:
        if not self._doc_id_exists(base_id):
            return base_id
        suffix = 2
        while self._doc_id_exists(f"{base_id}-{suffix}"):
            suffix += 1
        return f"{base_id}-{suffix}"

    def _unique_slug(self, base_slug: str) -> str:
        if not self._slug_exists(base_slug):
            return base_slug
        suffix = 2
        while self._slug_exists(f"{base_slug}-{suffix}"):
            suffix += 1
        return f"{base_slug}-{suffix}"

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        metadata = DocumentMetadata.model_validate_json(row["metadata_json"])
        chunk_ids = list(json.loads(row["chunk_ids_json"]))
        chunk_spans = [
            ChunkSpan.model_validate(item)
            for item in json.loads(row["chunk_spans_json"] or "[]")
        ]
        omissions = [
            OmissionRecord.model_validate(item)
            for item in json.loads(row["omissions_json"])
        ]
        tables = [
            TableRecord.model_validate(item)
            for item in json.loads(row["tables_json"] or "[]")
        ]
        return DocumentRecord(
            doc_id=row["doc_id"],
            slug=row["slug"],
            content=row["content"],
            metadata=metadata,
            storage_path=row["storage_path"],
            chunk_ids=chunk_ids,
            chunk_spans=chunk_spans,
            omissions=omissions,
            tables=tables,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
