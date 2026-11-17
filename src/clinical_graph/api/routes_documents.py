"""Document ingestion, listing, retrieval, and coordinated deletion endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from clinical_graph.api.deps import (
    get_ingestion_pipeline,
    get_knowledge_manager,
    get_openai_api_key,
)
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.pipelines.ingestion import DocumentRejectedError, IngestionPipeline
from clinical_graph.schemas.document import (
    DeletionReport,
    DocType,
    DocumentCreate,
    DocumentFilter,
    DocumentRecord,
    DocumentStatus,
    LocalRestriction,
    SourceType,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate,
    extract: bool = Query(default=False),
    classify: bool = Query(default=False),
    override_doc_type: bool = Query(default=False),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    api_key: str | None = Depends(get_openai_api_key),
) -> DocumentRecord:
    """Ingest a clinical document into the database and knowledge graph."""
    try:
        return await pipeline.ingest_document(
            payload,
            classify=classify,
            override_doc_type=override_doc_type,
            extract=extract,
            api_key=api_key,
        )
    except DocumentRejectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.detail,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/upload", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: DocType | None = Form(default=None),
    local_restriction: LocalRestriction = Form(LocalRestriction.NONE),
    description: str | None = Form(None),
    conditions: str | None = Form(default=None),
    target_populations: str | None = Form(default=None),
    specialties: str | None = Form(default=None),
    extract: bool = Form(default=False),
    classify: bool = Form(default=False),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    api_key: str | None = Depends(get_openai_api_key),
) -> DocumentRecord:
    """Upload and parse a ``.pdf``, ``.md``, or ``.txt`` file, then ingest it."""
    data = await file.read()
    filename = file.filename or "upload.txt"
    try:
        return await pipeline.ingest_file_bytes(
            data,
            filename,
            title=title,
            doc_type=doc_type,
            local_restriction=local_restriction,
            description=description,
            conditions=_parse_csv_list(conditions),
            target_populations=_parse_csv_list(target_populations),
            specialties=_parse_csv_list(specialties),
            classify=classify,
            extract=extract,
            api_key=api_key,
        )
    except DocumentRejectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.detail,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[DocumentRecord])
async def list_documents(
    doc_type: list[DocType] | None = Query(default=None),
    source_type: list[SourceType] | None = Query(default=None),
    local_restriction: list[LocalRestriction] | None = Query(default=None),
    condition: list[str] | None = Query(default=None),
    target_population: list[str] | None = Query(default=None),
    specialty: list[str] | None = Query(default=None),
    doc_status: DocumentStatus | None = Query(default=DocumentStatus.ACTIVE, alias="status"),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> list[DocumentRecord]:
    """List stored documents matching optional faceted AND filters."""
    filters = DocumentFilter(
        doc_types=doc_type,
        source_types=source_type,
        local_restrictions=local_restriction,
        conditions=condition,
        target_populations=target_population,
        specialties=specialty,
        status=doc_status,
    )
    return manager.db.filter_documents(filters)


@router.get("/{identifier}", response_model=DocumentRecord)
async def get_document(
    identifier: str,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> DocumentRecord:
    """Retrieve a single document by ``doc_id`` or unique ``slug``."""
    record = manager.db.get_by_id(identifier) or manager.db.get_by_slug(identifier)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{identifier}' not found",
        )
    return record


@router.delete("/{doc_id}", response_model=DeletionReport)
async def delete_document(
    doc_id: str,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DeletionReport:
    """Tightly delete a document from the knowledge graph, vector store, and database."""
    if pipeline.manager.db.get_by_id(doc_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return await pipeline.delete_document(doc_id)
