"""Environment-driven configuration. All tunables come from here — nothing hardcoded elsewhere.

This service is stateless with respect to the NestJS backend's MongoDB — it
never connects to Mongo. All state (uploaded photos, trained model,
job status) lives on local disk under `storage_root`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    storage_root: str = "storage/dataset"
    model_dir: str = "storage/model"
    min_training_photos: int = 3
    face_match_threshold: float = 0.5

    # Where the trained CNN classifier + label map are read from — reusing
    # the "training metode cnn" project's pipeline directly (per skripsi
    # requirement: metode CNN dilatih dari nol, bukan model pretrained).
    cnn_project_dir: str = "../../training metode cnn"
    image_size: tuple[int, int] = (128, 128)
    train_epochs: int = 50
    train_batch_size: int = 8

    # Webhook back to the NestJS backend after training completes.
    backend_webhook_url: str = "http://localhost:3000/face-recognition/training-complete"
    face_service_webhook_secret: str = "change-me-webhook-secret"

    host: str = "0.0.0.0"
    port: int = 4001


settings = Settings()
