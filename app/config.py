"""Environment-driven configuration. All tunables come from here — nothing hardcoded elsewhere.

This service is stateless with respect to the NestJS backend's MongoDB — it
never connects to Mongo. All state (uploaded photos, trained model,
job status) lives on local disk under `storage_root`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    storage_root: str = "storage/dataset"
    model_dir: str = "storage/model"
    # Raised from 3 — real retrains repeatedly showed ~20-photo students
    # collapsing into other classes or getting swallowed by class
    # imbalance, while 40-60 photo students classified reliably. Must stay
    # in sync with the backend's UploadTrainingDto @ArrayMinSize.
    min_training_photos: int = 20
    # Calibrated against a real training_log.csv/verify_model.py run (3
    # classes, 20 photos each, post-overfitting-fix model): correct
    # predictions on training photos had confidence as low as 0.4056 — the
    # heavier dropout that fixed the earlier overfitting collapse also
    # lowered the model's overall softmax confidence, so 0.5 rejected
    # genuine matches. 0.4 leaves a small buffer below that observed floor.
    face_match_threshold: float = 0.4
    # Minimum gap required between the top-1 and runner-up class
    # probabilities. Closed-set softmax always outputs some "winner" even
    # when two students look similar to the model — a confident-looking
    # top-1 score doesn't by itself mean the model wasn't nearly as
    # confident about the wrong student too. Below this margin, treat the
    # match as ambiguous ("tidak dikenali") instead of trusting a coin-flip.
    #
    # Raised 0.05->0.15 (2026-07-20): this is a LIVE production system —
    # the student roster keeps growing as staff register real students, so
    # the model's accuracy (per scripts/verify_model.py) will keep moving
    # and can't be "fixed" by threshold tuning alone. A real verify_model.py
    # run against the live 10-student model found WRONG predictions with
    # confidence/margin as high as 0.40/0.46 alongside CORRECT predictions
    # as low as conf 0.24/margin 0.0006 — the two distributions genuinely
    # overlap, so no margin value perfectly separates them (raising this
    # will also reject some correct matches as "tidak dikenali", not just
    # wrong ones). Given this is attendance data, a false REJECT (asks a
    # guru to manually verify) is far cheaper than a false ACCEPT (silently
    # marks the wrong student present), so we deliberately bias toward
    # rejecting more — 0.15 cuts most of the weak false-accepts seen in that
    # run while accepting more manual-verification load as the trade-off.
    face_match_margin: float = 0.15

    # Where the trained CNN classifier + label map are read from — reusing
    # the "training metode cnn" project's pipeline directly (per skripsi
    # requirement: metode CNN dilatih dari nol, bukan model pretrained).
    cnn_project_dir: str = "../../training metode cnn"
    image_size: tuple[int, int] = (128, 128)
    train_epochs: int = 50
    # Kept small on purpose: this service's default deployment target is a
    # 2GB-RAM VPS running MongoDB + the Node backend alongside it (see
    # deploy/face-service.service) — a bigger batch size raises peak
    # training memory and risks the process getting OOM-killed/restarted
    # mid-job (status stuck at "processing" forever). Raise this only on a
    # host with headroom to spare.
    train_batch_size: int = 4

    # EarlyStopping / ReduceLROnPlateau patience, applied by overriding the
    # cnn project's TrainingConfig at train time (see jobs/training_job.py) so
    # this service controls them without the cnn repo needing to be re-pulled.
    # Kept at the cnn project's original defaults: raising them was tried
    # alongside a (bad) augmentation change and the pair regressed the model,
    # so we reverted to the known-good values. The real lever for a better
    # model is more training photos per student, not these.
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4

    # Webhook back to the NestJS backend after training completes.
    backend_webhook_url: str = "http://localhost:3000/face-recognition/training-complete"
    face_service_webhook_secret: str = "change-me-webhook-secret"

    host: str = "0.0.0.0"
    port: int = 4001


settings = Settings()
