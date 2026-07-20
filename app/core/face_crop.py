"""Server-side face cropping — the authoritative preprocessing step.

The CNN resizes whatever it receives straight to `image_size` with no
detection of its own, so a full-frame photo trains/infers on background
instead of the face. Cropping here, on BOTH the training photos (see
jobs/training_job._build_flat_dataset) and the inference image (see
routes/inference), guarantees the two paths are framed identically AND
fixes already-uploaded full-frame photos at train time — no re-upload
needed.

Detector is OpenCV's bundled Haar cascade (classical Viola-Jones, ships
with opencv-python-headless — no extra dependency, no model download, and
not a deep detector like MTCNN/RetinaFace). It only re-frames the input;
the classifier is still the from-scratch CNN.

Because the crop re-anchors on the *detected face* (not on the frame), the
output framing is ~the same whether the input was a raw full frame or an
already-cropped face — so it composes safely with the frontend's own crop.

Falls back to returning the image unchanged when no face is found, so a
missed detection never drops a training photo or breaks an inference call.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def crop_to_face(image: np.ndarray, margin: float = 0.4) -> np.ndarray:
    """Return a square crop centred on the largest detected face.

    `margin` is the fraction of the face box added on every side. Color
    channels pass through untouched (detection runs on a grayscale copy),
    so this is safe for both BGR and RGB inputs. Returns the original image
    unchanged if no face is detected or detection errors out.
    """
    if image is None or image.ndim != 3:
        return image

    height, width = image.shape[:2]
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
    except cv2.error:
        return image

    if len(faces) == 0:
        return image

    # Largest detected face by area — the closest/most prominent person.
    x, y, face_w, face_h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    center_x = x + face_w / 2
    center_y = y + face_h / 2

    # Square side (face + margin), clamped so it can't exceed the frame.
    side = max(face_w, face_h) * (1 + 2 * margin)
    side = min(side, width, height)

    start_x = int(round(min(max(0.0, center_x - side / 2), width - side)))
    start_y = int(round(min(max(0.0, center_y - side / 2), height - side)))
    side_int = int(round(side))

    cropped = image[start_y : start_y + side_int, start_x : start_x + side_int]
    return cropped if cropped.size > 0 else image
