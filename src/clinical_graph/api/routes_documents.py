"""Document ingestion, listing, retrieval, and coordinated deletion endpoints."""

from collections.abc import Awaitable, Callable
import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from clinical_graph.api.deps import (
    get_ingestion_pipeline,
    get_job_store,
    get_knowledge_manager,
    get_openai_api_key,
)
from clinical_graph.api.rate_limit import check_rate_limit
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.pipelines.ingestion import (
    DocumentRejectedError,
    IngestionPipeline,
    IngestionResult,
)
from clinical_graph.pipelines.jobs import JobStore
from clinical_graph.schemas.document import (
    DeletionReport,
    DocType,
    DocumentCreate,
    DocumentFilter,
    DocumentRecord,
    DocumentStatus,
    IngestionJob,
    LocalRestriction,
    SourceType,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


async def _execute_ingestion_job(
    job_id: str,
    job_store: JobStore,
    ingest_coro_factory: Callable[[], Awaitable[DocumentRecord | IngestionResult]],
) -> None:
    """Run an ingestion coroutine in the background and update ``job_store``."""
    job_store.mark_running(job_id)
    try:
        outcome = await ingest_coro_factory()
        record = outcome.record if isinstance(outcome, IngestionResult) else outcome
    except DocumentRejectedError as exc:
        logger.warning("Background ingestion job %s rejected: %s", job_id, exc.detail)
        job_store.mark_failed(job_id, exc.detail)
    except Exception as exc:
        logger.exception("Background ingestion job %s failed", job_id)
        job_store.mark_failed(job_id, str(exc))
    else:
        job_store.mark_completed(
            job_id,
            doc_id=record.doc_id,
            result=record.model_dump(mode="json"),
        )


@router.post(
    "",
    response_model=DocumentRecord,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_202_ACCEPTED: {"model": IngestionJob}},
    dependencies=[Depends(check_rate_limit)],
)
async def create_document(
    payload: DocumentCreate,
    background_tasks: BackgroundTasks,
    extract: bool = Query(default=False),
    classify: bool = Query(default=False),
    override_doc_type: bool = Query(default=False),
    async_extract: bool = Query(default=False),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    job_store: JobStore = Depends(get_job_store),
    api_key: str | None = Depends(get_openai_api_key),
) -> DocumentRecord | JSONResponse:
    """Ingest a clinical document into the database and knowledge graph."""
    if async_extract:
        job = job_store.create_job(title=payload.metadata.title)
        background_tasks.add_task(
            _execute_ingestion_job,
            job.job_id,
            job_store,
            lambda: pipeline.ingest_document_with_report(
                payload,
                classify=classify,
                override_doc_type=override_doc_type,
                extract=extract,
                api_key=api_key,
                on_step=lambda msg: job_store.update_step(job.job_id, msg),
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=job.model_dump(mode="json"),
        )

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


@router.post(
    "/upload",
    response_model=DocumentRecord,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_202_ACCEPTED: {"model": IngestionJob}},
    dependencies=[Depends(check_rate_limit)],
)
async def upload_document(
    background_tasks: BackgroundTasks,
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
    async_extract: bool = Form(default=False),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    job_store: JobStore = Depends(get_job_store),
    api_key: str | None = Depends(get_openai_api_key),
) -> DocumentRecord | JSONResponse:
    """Upload and parse a ``.pdf``, ``.md``, or ``.txt`` file, then ingest it."""
    data = await file.read()
    filename = file.filename or "upload.txt"
    parsed_conditions = _parse_csv_list(conditions)
    parsed_populations = _parse_csv_list(target_populations)
    parsed_specialties = _parse_csv_list(specialties)

    if async_extract:
        job = job_store.create_job(filename=filename, title=title)
        background_tasks.add_task(
            _execute_ingestion_job,
            job.job_id,
            job_store,
            lambda: pipeline.ingest_file_bytes_with_report(
                data,
                filename,
                title=title,
                doc_type=doc_type,
                local_restriction=local_restriction,
                description=description,
                conditions=parsed_conditions,
                target_populations=parsed_populations,
                specialties=parsed_specialties,
                classify=classify,
                extract=extract,
                api_key=api_key,
                on_step=lambda msg: job_store.update_step(job.job_id, msg),
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=job.model_dump(mode="json"),
        )

    try:
        return await pipeline.ingest_file_bytes(
            data,
            filename,
            title=title,
            doc_type=doc_type,
            local_restriction=local_restriction,
            description=description,
            conditions=parsed_conditions,
            target_populations=parsed_populations,
            specialties=parsed_specialties,
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


@router.get("/jobs/{job_id}", response_model=IngestionJob)
async def get_ingestion_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> IngestionJob:
    """Retrieve the status and result of a background ingestion job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found",
        )
    return job


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


@router.delete(
    "/{doc_id}",
    response_model=DeletionReport,
    dependencies=[Depends(check_rate_limit)],
)
async def delete_document(
    doc_id: str,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DeletionReport:
    """Tightly delete a document from the knowledge graph, vector store, and database."""
    existing = pipeline.manager.db.get_by_id(doc_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    if existing.metadata.is_frozen:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Frozen example document cannot be removed.",
        )
    return await pipeline.delete_document(doc_id)
