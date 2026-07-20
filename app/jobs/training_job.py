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

import dataclasses
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


def _build_flat_dataset(temp_dir: Path) -> list[str]:
    """Copy each registered student's LATEST version's photos into
    temp_dir/<student_id>/... so it matches the layout split_dataset.py
    expects. Returns the list of student_ids included.

    Each photo is face-cropped on the way in (crop_to_face) — the SAME crop
    inference applies — so the model trains on face-filled frames and stays
    consistent with inference, and already-uploaded full-frame photos get
    fixed here without needing a re-upload. A photo with no detectable face
    falls back to being copied whole (never silently dropped)."""
    included: list[str] = []
    cropped_count = 0
    fallback_count = 0

    for student_id in list_registered_students():
        latest_dir = _latest_photo_dir(student_id)
        if latest_dir is None:
            continue

        dest = temp_dir / student_id
        dest.mkdir(parents=True, exist_ok=True)
        for photo in latest_dir.iterdir():
            if not photo.is_file():
                continue

            image = cv2.imread(str(photo))
            if image is None:
                # Not decodable by OpenCV — keep the original bytes as-is.
                shutil.copy2(photo, dest / photo.name)
                fallback_count += 1
                continue

            cropped = crop_to_face(image)
            if cropped is image:
                fallback_count += 1
            else:
                cropped_count += 1
            # Write as .jpg (OpenCV expects/writes BGR; TF later reads it back
            # as RGB, matching the existing pipeline).
            cv2.imwrite(str(dest / f"{photo.stem}.jpg"), cropped)

        included.append(student_id)

    logger.info(
        "Built training set: %d photos face-cropped, %d kept whole (no face found)",
        cropped_count, fallback_count,
    )
    return included


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
    from data.dataloader import get_class_names, get_train_dataset, get_validation_dataset
    from models.model import build_model
    from training.trainer import train as run_keras_training

    _JOB_STATUS[job_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    model_version = f"cnn-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    temp_dataset_dir = Path(settings.storage_root).parent / "_retrain_tmp" / "dataset"
    temp_split_dir = Path(settings.storage_root).parent / "_retrain_tmp" / "dataset_split"

    try:
        shutil.rmtree(temp_dataset_dir.parent, ignore_errors=True)
        temp_dataset_dir.mkdir(parents=True, exist_ok=True)

        included_students = _build_flat_dataset(temp_dataset_dir)
        if not included_students:
            raise ValueError("No students with uploaded photos found — nothing to train on")

        config = TrainingConfig(
            split_dir=str(temp_split_dir),
            epochs=settings.train_epochs,
            batch_size=settings.train_batch_size,
            image_size=settings.image_size,
            num_classes=len(included_students),
            # Previously forced to "loss" (training loss) here because with
            # min_training_photos=3 the per-student validation split could be
            # as small as ~1 image — too noisy for ModelCheckpoint/
            # EarlyStopping to trust (see BAB_IV_IMPLEMENTASI.md §4.5.2). That
            # override was confirmed to actively pick the most-overfit
            # checkpoint once students actually upload ~20 photos each:
            # training accuracy reached >90% while val_accuracy stayed pinned
            # at chance level and val_loss diverged. Now that uploads are
            # closer to ~20 photos/student in practice (~4-5/class in
            # validation), fall through to TrainingConfig's own "val_loss"
            # default instead of forcing "loss".
            checkpoint_dir=settings.model_dir,
            checkpoint_filename="best_model.h5",
            tensorboard_dir=str(Path(settings.model_dir) / "tensorboard"),
            output_dir=str(Path(settings.model_dir) / "logs"),
        )

        from data.split_dataset import split_dataset as split_dataset_fn

        split_dataset_fn(str(temp_dataset_dir), str(temp_split_dir))

        class_names = get_class_names(config.split_dir)  # sorted student_ids, = class index order
        config = dataclasses.replace(config, num_classes=len(class_names))

        train_dataset = get_train_dataset(config.split_dir, image_size=config.image_size, batch_size=config.batch_size)
        validation_dataset = get_validation_dataset(config.split_dir, image_size=config.image_size, batch_size=config.batch_size)

        model = build_model(
            num_classes=config.num_classes,
            input_shape=(*config.image_size, 3),
            dropout_head=config.dropout_head,
            dropout_dense=config.dropout_dense,
        )

        run_keras_training(model, train_dataset, validation_dataset, config)

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
