from __future__ import annotations

from pydantic import BaseModel, Field


class UploadTrainingRequest(BaseModel):
    student_id: str
    foto_urls: list[str] = Field(default_factory=list)
    nama_display: str | None = None  # optional, logging only — never used as a key


class UploadTrainingResponse(BaseModel):
    student_id: str
    version: int
    status: str


class TriggerTrainingRequest(BaseModel):
    student_ids: list[str]


class TriggerTrainingResponse(BaseModel):
    job_id: str


class TrainingStatusResponse(BaseModel):
    student_id: str
    status: str
    current_version: int
    model_version: str | None = None
    updated_at: str
