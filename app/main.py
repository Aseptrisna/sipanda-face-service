from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Terjadi kesalahan pada server"})
