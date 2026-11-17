"""Unit tests for DocumentDatabase CRUD, slug generation, and faceted filtering."""

import hashlib
from pathlib import Path
import sqlite3

from clinical_graph.schemas import (
    ChunkSpan,
    DocType,
    DocumentCreate,
    DocumentFilter,
    DocumentMetadata,
    DocumentStatus,
    LocalRestriction,
    OmissionRecord,
    SourceType,
)
from clinical_graph.graph import DocumentDatabase, slugify


def _make_doc(
    *,
    title: str,
    content: str,
    doc_type: DocType = DocType.GUIDELINE,
    source_type: SourceType = SourceType.MARKDOWN,
    local_restriction: LocalRestriction = LocalRestriction.NONE,
    effective_year: int | None = 2024,
    conditions: list[str] | None = None,
    target_populations: list[str] | None = None,
    specialties: list[str] | None = None,
    status: DocumentStatus = DocumentStatus.ACTIVE,
    slug: str | None = None,
) -> DocumentCreate:
    return DocumentCreate(
        content=content,
        slug=slug,
        metadata=DocumentMetadata(
            title=title,
            doc_type=doc_type,
            source_type=source_type,
            local_restriction=local_restriction,
            effective_year=effective_year,
            conditions=conditions or [],
            target_populations=target_populations or [],
            specialties=specialties or [],
            status=status,
        ),
    )


def test_slugify_behavior() -> None:
    assert slugify("S3-Leitlinie Diabetes mellitus (DE) 2024!") == "s3-leitlinie-diabetes-mellitus-de-2024"
    assert slugify("  Überwachung & Therapie  ") == "uberwachung-therapie"
    assert slugify("!!!") == "doc"


def test_document_db_crud_and_collisions() -> None:
    db = DocumentDatabase()
    assert db.count() == 0

    content = "Insulin therapy guidance for pediatric type 1 diabetes."
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    rec1 = db.add_document(
        _make_doc(
            title="DDG Pediatric Diabetes Guideline",
            content=content,
            effective_year=2024,
            local_restriction=LocalRestriction.DE,
            conditions=["Diabetes"],
            target_populations=["Pediatric"],
        )
    )
    assert db.count() == 1
    assert rec1.doc_id == f"doc_{expected_hash[:12]}"
    assert rec1.slug == "ddg-pediatric-diabetes-guideline-2024"
    assert rec1.metadata.content_hash == expected_hash
    assert rec1.created_at.tzinfo is not None
    assert rec1.updated_at == rec1.created_at

    assert db.get_by_id(rec1.doc_id) == rec1
    assert db.get_by_slug(rec1.slug) == rec1
    assert db.get_by_id("doc_missing") is None
    assert db.get_by_slug("missing-slug") is None

    # Adding identical title/year and content disambiguates both doc_id and slug
    rec2 = db.add_document(
        _make_doc(
            title="DDG Pediatric Diabetes Guideline",
            content=content,
            effective_year=2024,
        )
    )
    assert rec2.doc_id == f"doc_{expected_hash[:12]}-2"
    assert rec2.slug == "ddg-pediatric-diabetes-guideline-2024-2"
    assert db.count() == 2

    # Custom slug is respected
    rec3 = db.add_document(
        _make_doc(
            title="Custom",
            content="Custom content",
            slug="Custom Slug 01",
        )
    )
    assert rec3.slug == "custom-slug-01"

    # Delete removes from both primary map and slug index
    assert db.delete_document(rec1.doc_id) is True
    assert db.delete_document(rec1.doc_id) is False
    assert db.get_by_id(rec1.doc_id) is None
    assert db.get_by_slug(rec1.slug) is None
    assert db.count() == 2


def test_hierarchical_faceted_filtering() -> None:
    db = DocumentDatabase()

    doc_de_ped = db.add_document(
        _make_doc(
            title="DE Pediatric Diabetes",
            content="DE pediatric diabetes protocol",
            doc_type=DocType.GUIDELINE,
            local_restriction=LocalRestriction.DE,
            conditions=["Type 1 Diabetes", "Diabetes"],
            target_populations=["Pediatric"],
            specialties=["Endocrinology"],
        )
    )
    doc_us_ped = db.add_document(
        _make_doc(
            title="US Pediatric Diabetes",
            content="US pediatric diabetes protocol",
            doc_type=DocType.GUIDELINE,
            local_restriction=LocalRestriction.US,
            conditions=["Diabetes"],
            target_populations=["Pediatric"],
            specialties=["Endocrinology"],
        )
    )
    doc_de_adult = db.add_document(
        _make_doc(
            title="DE Adult Diabetes",
            content="DE adult diabetes protocol",
            doc_type=DocType.DRUG_LABEL,
            local_restriction=LocalRestriction.DE,
            conditions=["Diabetes"],
            target_populations=["Adult"],
            specialties=["Internal Medicine"],
        )
    )
    doc_de_htn = db.add_document(
        _make_doc(
            title="DE Pediatric Hypertension",
            content="DE pediatric hypertension protocol",
            doc_type=DocType.GUIDELINE,
            local_restriction=LocalRestriction.DE,
            conditions=["Hypertension"],
            target_populations=["Pediatric"],
            specialties=["Cardiology"],
        )
    )
    doc_archived = db.add_document(
        _make_doc(
            title="Archived DE Pediatric Diabetes",
            content="Old DE pediatric diabetes protocol",
            doc_type=DocType.GUIDELINE,
            local_restriction=LocalRestriction.DE,
            conditions=["Diabetes"],
            target_populations=["Pediatric"],
            status=DocumentStatus.ARCHIVED,
        )
    )

    # Level 1: condition = diabetes (active by default)
    lvl1 = db.filter_documents(DocumentFilter(conditions=["  DIABETES "]))
    assert {d.doc_id for d in lvl1} == {doc_de_ped.doc_id, doc_us_ped.doc_id, doc_de_adult.doc_id}

    # Level 2: condition = diabetes + target_population = pediatric
    lvl2 = db.filter_documents(
        DocumentFilter(conditions=["diabetes"], target_populations=["Pediatric"])
    )
    assert {d.doc_id for d in lvl2} == {doc_de_ped.doc_id, doc_us_ped.doc_id}

    # Level 3: condition = diabetes + target_population = pediatric + local_restriction = DE
    lvl3 = db.filter_documents(
        DocumentFilter(
            conditions=["diabetes"],
            target_populations=["pediatric"],
            local_restrictions=[LocalRestriction.DE],
        )
    )
    assert [d.doc_id for d in lvl3] == [doc_de_ped.doc_id]

    # Specialty and doc_type facets
    assert [
        d.doc_id
        for d in db.filter_documents(
            DocumentFilter(doc_types=[DocType.DRUG_LABEL], specialties=["internal medicine"])
        )
    ] == [doc_de_adult.doc_id]

    # Status=None includes archived
    all_ped_de = db.filter_documents(
        DocumentFilter(
            target_populations=["pediatric"],
            local_restrictions=[LocalRestriction.DE],
            status=None,
        )
    )
    assert {d.doc_id for d in all_ped_de} == {
        doc_de_ped.doc_id,
        doc_de_htn.doc_id,
        doc_archived.doc_id,
    }

    # filters=None returns all documents
    assert len(db.filter_documents(None)) == 5


def test_sqlite_file_persistence_save_load_and_close(tmp_path: Path) -> None:
    db_file = tmp_path / "nested" / "documents.db"
    db = DocumentDatabase(db_path=db_file)

    omission = OmissionRecord(
        omission_id="omit-0",
        reason="references",
        heading="## References",
        char_count=42,
        preview="[1] Example citation",
    )
    rec = db.add_document(
        _make_doc(
            title="SQLite Persistence Guideline",
            content="Metformin is first-line therapy for T2D.",
            conditions=["Type 2 Diabetes"],
        ),
        storage_path=str(tmp_path / "docs" / "example.txt"),
    )
    span = ChunkSpan(chunk_id=f"{rec.doc_id}:chunk:0", start_char=0, end_char=len(rec.content))
    updated = db.update_processing_metadata(
        rec.doc_id,
        chunk_ids=[f"{rec.doc_id}:chunk:0"],
        omissions=[omission],
        chunk_spans=[span],
    )
    assert updated is not None
    assert updated.chunk_spans == [span]
    db.save(db_file)
    db.close()

    assert db_file.is_file()

    reopened = DocumentDatabase(db_path=db_file)
    restored = reopened.get_by_id(rec.doc_id)
    assert restored is not None
    assert restored == updated
    assert reopened.get_by_slug(rec.slug) == updated
    reopened.close()

    # In-memory DB can also save to and load from a directory or .db path
    mem_db = DocumentDatabase()
    mem_rec = mem_db.add_document(
        _make_doc(title="In-Memory Doc", content="In-memory content.")
    )
    backup_dir = tmp_path / "backup"
    mem_db.save(backup_dir)
    mem_db.close()

    assert (backup_dir / "documents.db").is_file()
    loaded_db = DocumentDatabase()
    loaded_db.load(backup_dir)
    assert loaded_db.count() == 1
    assert loaded_db.get_by_id(mem_rec.doc_id) == mem_rec
    loaded_db.close()


def test_sqlite_migration_adds_chunk_spans_json_column(tmp_path: Path) -> None:
    legacy_db_path = tmp_path / "legacy_documents.db"
    conn = sqlite3.connect(str(legacy_db_path))
    conn.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            source_type TEXT NOT NULL,
            local_restriction TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            chunk_ids_json TEXT NOT NULL,
            omissions_json TEXT NOT NULL,
            storage_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    meta = DocumentMetadata(
        title="Legacy Guideline",
        doc_type=DocType.GUIDELINE,
        source_type=SourceType.RAW_TEXT,
    )
    conn.execute(
        """
        INSERT INTO documents (
            doc_id, slug, content, doc_type, source_type, local_restriction,
            status, metadata_json, chunk_ids_json, omissions_json,
            storage_path, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "doc_legacy1",
            "legacy-guideline",
            "Legacy clinical content.",
            "guideline",
            "raw_text",
            "none",
            "active",
            meta.model_dump_json(),
            '["doc_legacy1:chunk:0"]',
            "[]",
            None,
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    migrated_db = DocumentDatabase(db_path=legacy_db_path)
    restored = migrated_db.get_by_id("doc_legacy1")
    assert restored is not None
    assert restored.chunk_ids == ["doc_legacy1:chunk:0"]
    assert restored.chunk_spans == []

    new_span = ChunkSpan(chunk_id="doc_legacy1:chunk:0", start_char=0, end_char=24)
    updated = migrated_db.update_processing_metadata(
        "doc_legacy1",
        chunk_ids=["doc_legacy1:chunk:0"],
        omissions=[],
        chunk_spans=[new_span],
    )
    assert updated is not None
    assert updated.chunk_spans == [new_span]
    migrated_db.close()
