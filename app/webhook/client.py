"""Webhook client: reports training completion back to the NestJS backend.

Retries with backoff — the contract (sipanda-face-training-service.md)
requires this so a transient backend outage doesn't silently lose the
training result.
"""

from __future__ import annotations

import time

import httpx

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1, 3, 9)


def send_training_complete(
    student_id: str,
    status: str,
    version: int,
    model_version: str | None = None,
    error_message: str | None = None,
) -> bool:
    payload = {
        "student_id": student_id,
        "status": status,
        "version": version,
        "model_version": model_version,
        "error_message": error_message,
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                settings.backend_webhook_url,
                json=payload,
                headers={"x-webhook-secret": settings.face_service_webhook_secret},
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Webhook sent for student_id=%s status=%s", student_id, status)
            return True
        except httpx.HTTPError as exc:
            logger.warning(
                "Webhook attempt %d/%d failed for student_id=%s: %s",
                attempt, MAX_ATTEMPTS, student_id, exc,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS[attempt - 1])

    logger.error("Webhook permanently failed for student_id=%s after %d attempts", student_id, MAX_ATTEMPTS)
    return False
