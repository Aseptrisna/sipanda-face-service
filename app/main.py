from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes import health, inference, training
from app.utils.logger import get_logger

app = FastAPI(title="SIPANDA Face Recognition Service")
logger = get_logger(__name__)

# Routes are mounted under /api to match the backend's FACE_SERVICE_BASE_URL
# (http://localhost:4001/api) — avoids needing to change the already-running
# backend's .env.
app.include_router(health.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")


@app.on_event("startup")
def validate_environment() -> None:
    """Fail loudly at boot instead of only surfacing these as a buried
    exception inside the background training job later. CNN_PROJECT_DIR in
    particular is an external sibling folder — easy to forget when copying
    the deployment to a new machine."""
    cnn_dir = (Path(__file__).resolve().parents[1] / settings.cnn_project_dir).resolve()
    if not cnn_dir.is_dir():
        logger.error(
            "CNN_PROJECT_DIR tidak ditemukan di %s — /training/trigger akan gagal "
            "sampai folder 'training metode cnn' di-deploy ke lokasi ini (cek "
            "CNN_PROJECT_DIR di .env).",
            cnn_dir,
        )

    from app.core.classifier import CURRENT

    if CURRENT is None:
        logger.warning(
            "Belum ada model terlatih di %s — /inference/match akan menolak "
            "request sampai /training/trigger berhasil dijalankan minimal sekali.",
            settings.model_dir,
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Terjadi kesalahan pada server"})
