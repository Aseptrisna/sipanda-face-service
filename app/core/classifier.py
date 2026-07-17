"""Load the custom CNN classifier (trained from scratch, no pretrained backbone)
and run inference: raw photo -> student_id + confidence.

No face detection/alignment step here — the "training metode cnn" pipeline
this model was trained with never used one either (Phase 5 became data
augmentation, not MTCNN/RetinaFace). Adding face detection only at
inference time would create a train/inference preprocessing mismatch, so
we deliberately keep both sides consistent: resize the raw photo straight
to `image_size`.

Note: the caller (app/routes/inference.py) is responsible for handing this
class an RGB image. cv2.imdecode() (used by that route) returns BGR, so it
converts BGR->RGB before calling predict() — training reads photos as RGB
via tf.keras.utils.image_dataset_from_directory.
"""

from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf

from app.config import settings


class Classifier:
    def __init__(self, model_path: str, label_map_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(label_map_path) as f:
            raw_map = json.load(f)
        # raw_map: {"0": "<student_id>", "1": "<student_id>", ...}
        self.index_to_student_id = {int(k): v for k, v in raw_map.items()}

    @classmethod
    def load_if_available(cls) -> "Classifier | None":
        model_path = os.path.join(settings.model_dir, "best_model.h5")
        label_map_path = os.path.join(settings.model_dir, "label_map.json")
        if not (os.path.isfile(model_path) and os.path.isfile(label_map_path)):
            return None
        return cls(model_path, label_map_path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        resized = tf.image.resize(image, settings.image_size)
        normalized = tf.cast(resized, tf.float32) / 255.0
        batch = tf.expand_dims(normalized, axis=0)

        probabilities = self.model.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        return self.index_to_student_id[best_index], confidence


CURRENT: Classifier | None = Classifier.load_if_available()
