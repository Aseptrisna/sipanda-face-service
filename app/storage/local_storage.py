"""Local-disk photo storage, one folder per student_id/version.

Also the source of truth for "current version" per student (no separate
DB — version is just "how many v* folders exist for this student_id").

Kept behind this thin module (not scattered `open()` calls) so swapping to
S3-compatible storage later only means changing this file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.config import settings


def student_root_dir(student_id: str) -> Path:
    return Path(settings.storage_root) / student_id


def student_version_dir(student_id: str, version: int) -> Path:
    path = student_root_dir(student_id) / f"v{version}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_version(student_id: str) -> int:
    root = student_root_dir(student_id)
    if not root.is_dir():
        return 0
    versions = [int(p.name[1:]) for p in root.iterdir() if p.is_dir() and p.name.startswith("v")]
    return max(versions, default=0)


def save_photo(student_id: str, version: int, filename: str, content: bytes) -> Path:
    dest = student_version_dir(student_id, version) / filename
    dest.write_bytes(content)
    return dest


def list_photos(student_id: str, version: int) -> list[Path]:
    directory = student_version_dir(student_id, version)
    return sorted(p for p in directory.iterdir() if p.is_file())


def list_registered_students() -> list[str]:
    """All student_ids that have at least one uploaded photo version."""
    root = Path(settings.storage_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and latest_version(p.name) > 0
    )


def delete_student_data(student_id: str) -> None:
    student_path = student_root_dir(student_id)
    if student_path.exists():
        shutil.rmtree(student_path)
