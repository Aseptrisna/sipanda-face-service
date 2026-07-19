from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core import classifier as classifier_module
from app.schemas.inference import MatchRequest, MatchResponse
from app.storage.local_storage import list_registered_students
from app.utils.logger import get_logger

router = APIRouter(prefix="/inference", tags=["inference"])
logger = get_logger(__name__)


@router.post("/match", response_model=MatchResponse)
def match_face(payload: MatchRequest) -> MatchResponse:
    if classifier_module.CURRENT is None:
        raise HTTPException(503, "Model belum pernah dilatih — jalankan /training/trigger dahulu")

    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception as exc:
        logger.error("image_base64 tidak valid: %s", exc)
        raise HTTPException(400, "image_base64 tidak valid")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Gagal decode gambar (ukuran payload=%d bytes)", len(image_bytes))
        raise HTTPException(400, "Gagal decode gambar")

    # cv2.imdecode always returns BGR, but training decodes photos via
    # tf.keras.utils.image_dataset_from_directory, which yields RGB. Without
    # this conversion the model sees red/blue channels swapped at inference
    # time only — a silent train/inference mismatch that shows up as
    # misclassification between known students, not as a decode error.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        student_id, confidence, margin = classifier_module.CURRENT.predict(image)
    except Exception:
        logger.exception("Gagal melakukan prediksi wajah")
        raise HTTPException(500, "Gagal melakukan prediksi wajah")

    if confidence < settings.face_match_threshold:
        logger.info("Below threshold: predicted=%s confidence=%.4f", student_id, confidence)
        return MatchResponse(student_id=None, confidence=confidence)

    if margin < settings.face_match_margin:
        logger.info(
            "Ambiguous match rejected: predicted=%s confidence=%.4f margin=%.4f (min=%.4f)",
            student_id, confidence, margin, settings.face_match_margin,
        )
        return MatchResponse(student_id=None, confidence=confidence)

    # The classifier is a closed-set softmax fixed at the last training run —
    # deleting a student's photos (DELETE /training/:student_id) does NOT
    # retrain the model, so a deleted student_id can remain a live class the
    # model still confidently predicts until the next full retrain. Treat a
    # prediction for a student who is no longer registered as "unrecognized"
    # instead of trusting a stale class — cheap to check, no retrain needed.
    if student_id not in list_registered_students():
        logger.warning(
            "Predicted student_id=%s tidak lagi terdaftar (data sudah dihapus) — "
            "diperlakukan sebagai tidak dikenali, confidence=%.4f",
            student_id, confidence,
        )
        return MatchResponse(student_id=None, confidence=confidence)

    return MatchResponse(student_id=student_id, confidence=confidence)
