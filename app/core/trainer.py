"""Model compile + fit loop — owned by this service.

Model ARCHITECTURE still comes from the cnn project (models.model.build_model)
— that's the actual "metode CNN" under study and isn't touched here. But the
fit loop mechanics (optimizer, callbacks, and now class weighting) live here,
same reasoning as app/core/dataset.py: a knob that decides whether the model
actually learns shouldn't live in a separately-deployed repo that has to be
re-pulled to take effect.

class_weight was added after a real VPS retrain (7 students, one with 60
photos vs ~20 for the rest) showed the model collapsing toward whichever
class had the most training photos — 30/40 of one under-represented
student's photos were predicted as the 60-photo student, even though that
student had face-crop working correctly. categorical_crossentropy with no
class weighting lets the majority class dominate; weighting each class
inversely to its photo count counteracts that directly.
"""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

LEARNING_RATE = 1e-3
REDUCE_LR_FACTOR = 0.5


def _build_callbacks(
    checkpoint_path: str,
    monitor: str,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    tensorboard_dir: str,
    output_dir: str,
) -> list[callbacks.Callback]:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return [
        callbacks.EarlyStopping(
            monitor=monitor, patience=early_stopping_patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=REDUCE_LR_FACTOR, patience=reduce_lr_patience, min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=monitor, save_best_only=True),
        callbacks.TensorBoard(log_dir=tensorboard_dir),
        callbacks.CSVLogger(filename=os.path.join(output_dir, "training_log.csv"), append=False),
    ]


def compute_class_weight(class_names: list[str], photo_counts: dict[str, int]) -> dict[int, float]:
    """Inverse-frequency weight per class index, so a class with fewer
    training photos counts for more in the loss — otherwise the class with
    the most photos dominates the softmax regardless of face-crop quality.

    Standard "balanced" formula: weight_i = n_samples / (n_classes * n_i).
    A class with the average photo count gets weight ~1.0; a class with half
    the average gets ~2.0; the majority class gets pulled below 1.0.
    """
    total = sum(photo_counts[name] for name in class_names)
    n_classes = len(class_names)
    return {
        index: total / (n_classes * photo_counts[name])
        for index, name in enumerate(class_names)
    }


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    *,
    epochs: int,
    monitor: str,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    checkpoint_path: str,
    tensorboard_dir: str,
    output_dir: str,
    class_weight: dict[int, float] | None,
) -> tf.keras.callbacks.History:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=_build_callbacks(
            checkpoint_path,
            monitor,
            early_stopping_patience,
            reduce_lr_patience,
            tensorboard_dir,
            output_dir,
        ),
    )
