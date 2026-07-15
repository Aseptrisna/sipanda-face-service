"""Training endpoints — stateless w.r.t. the NestJS backend's MongoDB.

This service only ever receives an existing `student_id` (NestJS always
creates the Siswa document itself before calling here) — there is no
"create a new student" path on this side.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import settings
from app.jobs.training_job import get_job_status, is_training_in_progress, run_training_job
from app.schemas.training import (
    TrainingStatusResponse,
    TriggerTrainingRequest,
    TriggerTrainingResponse,
    UploadTrainingRequest,
    UploadTrainingResponse,
)
from app.storage.local_storage import delete_student_data, latest_version, save_photo
from app.storage.student_status import delete_status, load_status, save_status
from app.utils.logger import get_logger

router = APIRouter(prefix="/training", tags=["training"])
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadTrainingResponse, status_code=202)
def upload_training_photos(payload: UploadTrainingRequest) -> UploadTrainingResponse:
    if len(payload.foto_urls) < settings.min_training_photos:
        raise HTTPException(400, f"Minimal {settings.min_training_photos} foto diperlukan")

    # Download all photos BEFORE writing anything to disk under the new
    # version — a partial failure must not leave a half-populated version.
    downloaded: list[tuple[str, bytes]] = []
    for url in payload.foto_urls:
        try:
            response = httpx.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Gagal mengambil foto dari %s: %s", url, exc)
            raise HTTPException(400, f"Gagal mengambil foto dari {url}: {exc}")

        suffix = Path(url).suffix or ".jpg"
        downloaded.append((suffix, response.content))

    version = latest_version(payload.student_id) + 1
    for index, (suffix, content) in enumerate(downloaded):
        save_photo(payload.student_id, version, f"photo_{index}{suffix}", content)

    save_status(payload.student_id, "pending", version, None)

    logger.info(
        "Uploaded %d photos for student_id=%s version=%d (nama_display=%s)",
        len(payload.foto_urls), payload.student_id, version, payload.nama_display,
    )

    return UploadTrainingResponse(student_id=payload.student_id, version=version, status="pending")


@router.post("/trigger", response_model=TriggerTrainingResponse, status_code=202)
def trigger_training(
    payload: TriggerTrainingRequest, background_tasks: BackgroundTasks
) -> TriggerTrainingResponse:
    # Best-effort fast rejection — only one full retrain may run at a time
    # (see app/jobs/training_job.py _TRAINING_LOCK). This check has an
    # inherent race with a job starting right after it, but that race is
    # closed by the lock itself inside run_training_job; this just avoids
    # queueing an obviously-doomed job when we can already tell it's busy.
    if is_training_in_progress():
        raise HTTPException(409, "Training lain sedang berjalan, coba lagi setelah itu selesai")

    job_id = f"job_{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"

    for student_id in payload.student_ids:
        save_status(student_id, "processing", latest_version(student_id), None)

    background_tasks.add_task(run_training_job, job_id, payload.student_ids)

    logger.info("Training job %s queued for %d students", job_id, len(payload.student_ids))
    return TriggerTrainingResponse(job_id=job_id)


@router.get("/status/{student_id}", response_model=TrainingStatusResponse)
def get_training_status(student_id: str) -> TrainingStatusResponse:
    status = load_status(student_id)
    if status is None:
        raise HTTPException(404, f"Tidak ada data training untuk student_id={student_id}")

    return TrainingStatusResponse(**status)


@router.delete("/{student_id}", status_code=204)
def delete_training_data(student_id: str) -> None:
    delete_student_data(student_id)
    delete_status(student_id)
    logger.info("Deleted all training data for student_id=%s", student_id)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan")
    return status
