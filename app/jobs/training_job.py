"""Full retrain job: reuses the "training metode cnn" project's own
split_dataset/dataloader/model/trainer modules directly — this service does
not reimplement CNN training, it drives the existing pipeline.

Design: the classifier is a SINGLE shared multi-class model covering every
student_id that currently has at least one uploaded photo. There is no
per-student incremental training — a closed-set softmax classifier cannot
do that. Every call to `run_training_job` retrains the whole model from
scratch on all currently registered students, then reports per-student
webhook results for the student_ids that were part of this job's request.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import settings
from app.core.face_crop import crop_to_face
from app.storage.local_storage import latest_version, list_registered_students, student_root_dir
from app.storage.student_status import save_status
from app.utils.logger import get_logger
from app.webhook.client import send_training_complete

logger = get_logger(__name__)

_FACE_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_CNN_PROJECT_DIR = (_FACE_SERVICE_ROOT / settings.cnn_project_dir).resolve()

sys.path.insert(0, str(_CNN_PROJECT_DIR / "src"))
sys.path.insert(0, str(_CNN_PROJECT_DIR / "configs"))

# Job status is in-memory only (no external queue/DB per this service's
# "minimize moving parts" design) — lost on restart, acceptable at this scale.
_JOB_STATUS: dict[str, dict] = {}

# Only one full retrain may run at a time: temp_dataset_dir/temp_split_dir and
# the checkpoint file below are fixed paths, not per-job — two overlapping
# retrains (e.g. a double-triggered /training/trigger) would read/write/
# rmtree each other's files, causing intermittent failures ("kadang bisa
# kadang tidak") instead of a clean, explicit rejection.
_TRAINING_LOCK = threading.Lock()


def is_training_in_progress() -> bool:
    return _TRAINING_LOCK.locked()


def get_job_status(job_id: str) -> dict | None:
    return _JOB_STATUS.get(job_id)


def _latest_photo_dir(student_id: str) -> Path | None:
    versions = sorted(
        (p for p in student_root_dir(student_id).iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda p: int(p.name[1:]),
    )
    return versions[-1] if versions else None


def _build_flat_dataset(temp_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Copy each registered student's LATEST version's photos into
    temp_dir/<student_id>/... so it matches the layout split_dataset.py
    expects. Returns (student_ids included, photo count per student_id) —
    the counts feed class weighting (see trainer.compute_class_weight): a
    real retrain with one student at 60 photos vs ~20 for everyone else
    collapsed the model toward that majority class, so training must be
    told how imbalanced the classes are.

    Each photo is face-cropped on the way in (crop_to_face) — the SAME crop
    inference applies — so the model trains on face-filled frames and stays
    consistent with inference, and already-uploaded full-frame photos get
    fixed here without needing a re-upload. A photo with no detectable face
    falls back to being copied whole (never silently dropped)."""
    included: list[str] = []
    photo_counts: dict[str, int] = {}
    cropped_count = 0
    fallback_count = 0

    for student_id in list_registered_students():
        latest_dir = _latest_photo_dir(student_id)
        if latest_dir is None:
            continue

        dest = temp_dir / student_id
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for photo in latest_dir.iterdir():
            if not photo.is_file():
                continue

            image = cv2.imread(str(photo))
            if image is None:
                # Not decodable by OpenCV — keep the original bytes as-is.
                shutil.copy2(photo, dest / photo.name)
                fallback_count += 1
                count += 1
                continue

            cropped = crop_to_face(image)
            if cropped is image:
                fallback_count += 1
            else:
                cropped_count += 1
            # Write as .jpg (OpenCV expects/writes BGR; TF later reads it back
            # as RGB, matching the existing pipeline).
            cv2.imwrite(str(dest / f"{photo.stem}.jpg"), cropped)
            count += 1

        included.append(student_id)
        photo_counts[student_id] = count

    logger.info(
        "Built training set: %d photos face-cropped, %d kept whole (no face found)",
        cropped_count, fallback_count,
    )
    return included, photo_counts


def run_training_job(job_id: str, requested_student_ids: list[str]) -> None:
    if not _TRAINING_LOCK.acquire(blocking=False):
        logger.warning(
            "Training job %s ditolak: retrain lain sedang berjalan", job_id,
        )
        _JOB_STATUS[job_id] = {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_message": "Retrain lain sedang berjalan, coba lagi setelah itu selesai",
        }
        for student_id in requested_student_ids:
            version = latest_version(student_id)
            save_status(student_id, "failed", version, None)
            send_training_complete(
                student_id=student_id,
                status="failed",
                version=version,
                error_message="Retrain lain sedang berjalan, coba lagi setelah itu selesai",
            )
        return

    from config import TrainingConfig  # training metode cnn's own config
    from data.dataloader import get_class_names
    from models.model import build_model  # architecture only — the actual "metode CNN"

    # Dataset pipeline, fit loop, and class weighting are all owned by THIS
    # service (not the cnn project) — so a deploy is pull+restart face-service
    # only, and none of the result-critical knobs depend on the separately
    # deployed cnn repo being re-pulled.
    from app.core.dataset import build_train_dataset, build_validation_dataset
    from app.core.trainer import compute_class_weight, train_model

    _JOB_STATUS[job_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    model_version = f"cnn-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    temp_dataset_dir = Path(settings.storage_root).parent / "_retrain_tmp" / "dataset"
    temp_split_dir = Path(settings.storage_root).parent / "_retrain_tmp" / "dataset_split"

    try:
        shutil.rmtree(temp_dataset_dir.parent, ignore_errors=True)
        temp_dataset_dir.mkdir(parents=True, exist_ok=True)

        included_students, photo_counts = _build_flat_dataset(temp_dataset_dir)
        if not included_students:
            raise ValueError("No students with uploaded photos found — nothing to train on")

        # Only used for its dropout defaults now (architecture regularization,
        # unrelated to the dataset/fit-loop knobs this service owns above).
        cnn_defaults = TrainingConfig()

        from data.split_dataset import split_dataset as split_dataset_fn

        split_dataset_fn(str(temp_dataset_dir), str(temp_split_dir))

        class_names = get_class_names(str(temp_split_dir))  # sorted student_ids, = class index order

        train_dataset = build_train_dataset(
            str(temp_split_dir), image_size=settings.image_size, batch_size=settings.train_batch_size
        )
        validation_dataset = build_validation_dataset(
            str(temp_split_dir), image_size=settings.image_size, batch_size=settings.train_batch_size
        )

        model = build_model(
            num_classes=len(class_names),
            input_shape=(*settings.image_size, 3),
            dropout_head=cnn_defaults.dropout_head,
            dropout_dense=cnn_defaults.dropout_dense,
        )

        class_weight = compute_class_weight(class_names, photo_counts)
        logger.info("Class weights (by photo count %s): %s", photo_counts, class_weight)

        train_model(
            model,
            train_dataset,
            validation_dataset,
            epochs=settings.train_epochs,
            monitor="val_loss",
            early_stopping_patience=settings.early_stopping_patience,
            reduce_lr_patience=settings.reduce_lr_patience,
            checkpoint_path=str(Path(settings.model_dir) / "best_model.h5"),
            tensorboard_dir=str(Path(settings.model_dir) / "tensorboard"),
            output_dir=str(Path(settings.model_dir) / "logs"),
            class_weight=class_weight,
        )

        Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
        label_map = {str(index): student_id for index, student_id in enumerate(class_names)}
        with open(Path(settings.model_dir) / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        from app.core import classifier as classifier_module

        classifier_module.CURRENT = classifier_module.Classifier.load_if_available()

        _JOB_STATUS[job_id] = {
            "status": "success",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "included_students": included_students,
            "model_version": model_version,
        }

        for student_id in requested_student_ids:
            version = latest_version(student_id)
            if student_id in included_students:
                save_status(student_id, "trained", version, model_version)
                send_training_complete(
                    student_id=student_id,
                    status="trained",
                    version=version,
                    model_version=model_version,
                )
            else:
                save_status(student_id, "failed", version, None)
                send_training_complete(
                    student_id=student_id,
                    status="failed",
                    version=version,
                    error_message="Tidak ada foto ter-upload untuk student_id ini",
                )

        logger.info(
            "Training job %s succeeded: %d students included (%s)",
            job_id, len(included_students), model_version,
        )

    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        _JOB_STATUS[job_id] = {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_message": str(exc),
        }
        for student_id in requested_student_ids:
            save_status(student_id, "failed", latest_version(student_id), None)
            send_training_complete(
                student_id=student_id,
                status="failed",
                version=latest_version(student_id),
                error_message=str(exc),
            )
    finally:
        shutil.rmtree(temp_dataset_dir.parent, ignore_errors=True)
        _TRAINING_LOCK.release()
