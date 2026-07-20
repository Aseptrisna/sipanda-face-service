"""Training/validation tf.data pipelines — owned by THIS service.

Historically face-service reused the cnn project's own dataloader (and its
augmentation). That put a result-critical knob (augmentation strength) in a
separate repo that had to be re-pulled on the VPS to take effect — easy to
forget, and it did get forgotten. Since preprocessing is now genuinely a
face-service concern (it already crops every photo before training, see
app/core/face_crop), the dataset pipeline lives here too so that deploying
= pull + restart THIS service only.

The pipeline mirrors the cnn project's original order (decode → cache →
augment → normalize → shuffle → prefetch for train; decode → cache →
normalize → prefetch for val) so the trained model still expects the same
[0,1] normalized RGB input the inference path produces. Only the
augmentation ranges differ: they're moderated for the face-CROPPED inputs
this service feeds in (the wide original ranges were tuned for full-frame
photos with a tiny face and distorted/clipped a face-filling crop).
"""

from __future__ import annotations

import os

import tensorflow as tf

_SHUFFLE_BUFFER = 1000


def _build_augmentation_pipeline() -> tf.keras.Sequential:
    """Moderate augmentation for face-cropped inputs.

    Ranges reduced from the cnn project's original aggressive values
    (rotation 0.10, zoom ±0.3, translation 0.2, contrast/brightness 0.4)
    because those were sized for a tiny face in a full frame; on a
    face-filling crop they distort or clip the face, which capped training
    accuracy at ~62% (model couldn't fit its own training set) and left
    half the classes unseparated. Still enough variation to regularize the
    small per-class dataset.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),  # ~18 degrees
            tf.keras.layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ],
        name="train_augmentation",
    )


_augmentation_pipeline = _build_augmentation_pipeline()


def _normalize(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    normalized = tf.cast(image, tf.float32) / 255.0
    # RandomContrast/RandomBrightness can push pixels slightly outside
    # [0, 255]; clip after normalizing so the model never sees out-of-range
    # values (matches the cnn project's original behavior).
    return tf.clip_by_value(normalized, 0.0, 1.0), label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return _augmentation_pipeline(image, training=True), label


def build_train_dataset(
    split_dir: str, image_size: tuple[int, int], batch_size: int
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(split_dir, "train"),
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    return (
        dataset.cache()
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(_SHUFFLE_BUFFER)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_validation_dataset(
    split_dir: str, image_size: tuple[int, int], batch_size: int
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(split_dir, "validation"),
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,  # validation must reflect real, unaltered input
    )
    return (
        dataset.cache()
        .map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
