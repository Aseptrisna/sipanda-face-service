from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core import classifier as classifier_module
from app.schemas.inference import MatchRequest, MatchResponse
from app.utils.logger import get_logger

router = APIRouter(prefix="/inference", tags=["inference"])
logger = get_logger(__name__)


@router.post("/match", response_model=MatchResponse)
def match_face(payload: MatchRequest) -> MatchResponse:
    if classifier_module.CURRENT is None:
        raise HTTPException(503, "Model belum pernah dilatih — jalankan /training/trigger dahulu")

    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(400, "image_base64 tidak valid")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Gagal decode gambar")

    student_id, confidence = classifier_module.CURRENT.predict(image)

    if confidence < settings.face_match_threshold:
        logger.info("Below threshold: predicted=%s confidence=%.4f", student_id, confidence)
        return MatchResponse(student_id=None, confidence=confidence)

    return MatchResponse(student_id=student_id, confidence=confidence)
