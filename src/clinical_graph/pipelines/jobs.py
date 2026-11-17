"""Thread-safe in-memory job store for asynchronous background ingestion."""

from datetime import UTC, datetime
from threading import RLock
from typing import Any
import uuid

from clinical_graph.schemas.document import IngestionJob, JobStatus


class JobStore:
    """Bounded, thread-safe in-memory registry of background ingestion jobs."""

    def __init__(self, max_jobs: int = 500) -> None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be >= 1")
        self._max_jobs = max_jobs
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = RLock()

    def create_job(
        self,
        *,
        filename: str | None = None,
        title: str | None = None,
    ) -> IngestionJob:
        """Create and store a new ``queued`` ingestion job."""
        now = datetime.now(UTC)
        job = IngestionJob(
            job_id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            filename=filename,
            title=title,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            while len(self._jobs) >= self._max_jobs:
                oldest_key = next(iter(self._jobs))
                del self._jobs[oldest_key]
            self._jobs[job.job_id] = job
        return job

    def mark_running(self, job_id: str) -> IngestionJob | None:
        """Transition an existing job to ``running``."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated

    def mark_completed(
        self,
        job_id: str,
        doc_id: str,
        result: dict[str, Any] | None = None,
    ) -> IngestionJob | None:
        """Transition an existing job to ``completed`` with its persisted ``doc_id``."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(
                update={
                    "status": JobStatus.COMPLETED,
                    "doc_id": doc_id,
                    "error": None,
                    "result": result,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated

    def mark_failed(self, job_id: str, error: str) -> IngestionJob | None:
        """Transition an existing job to ``failed`` with an error message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "error": error,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated

    def update_step(self, job_id: str, step: str) -> None:
        """Update the live ``current_step`` message for an existing job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            self._jobs[job_id] = job.model_copy(
                update={
                    "current_step": step,
                    "updated_at": datetime.now(UTC),
                }
            )

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Return the job for ``job_id``, or ``None`` if unknown or evicted."""
        with self._lock:
            return self._jobs.get(job_id)


__all__ = ["JobStore"]
