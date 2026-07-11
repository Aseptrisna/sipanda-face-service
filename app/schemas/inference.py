from __future__ import annotations

from pydantic import BaseModel


class MatchRequest(BaseModel):
    image_base64: str


class MatchResponse(BaseModel):
    student_id: str | None
    confidence: float
