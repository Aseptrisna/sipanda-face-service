"""Per-student status tracking, local-disk only (no DB) — this service is
stateless w.r.t. the NestJS backend, so it needs its own minimal bookkeeping
for `GET /training/status/:student_id` to answer without a live job lookup.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _status_path(student_id: str) -> Path:
    return Path(settings.model_dir) / "student_status" / f"{student_id}.json"


def save_status(student_id: str, status: str, version: int, model_version: str | None = None) -> None:
    path = _status_path(student_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "student_id": student_id,
                "status": status,
                "current_version": version,
                "model_version": model_version,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )


def load_status(student_id: str) -> dict | None:
    path = _status_path(student_id)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def delete_status(student_id: str) -> None:
    path = _status_path(student_id)
    if path.is_file():
        path.unlink()
