from __future__ import annotations

from fastapi import FastAPI

from app.routes import health, inference, training

app = FastAPI(title="SIPANDA Face Recognition Service")

# Routes are mounted under /api to match the backend's FACE_SERVICE_BASE_URL
# (http://localhost:4001/api) — avoids needing to change the already-running
# backend's .env.
app.include_router(health.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
