"""Sanity-check the currently loaded model against its own training photos.

Not a proper held-out evaluation (these images were part of the training
set, so this can't catch overfitting) — it only catches the failure mode
we saw before the last retrain: a stale/broken model confidently
mispredicting even the photos it was trained on. Run after every retrain
before testing from the frontend.

Usage (from face-service/, with venv activated):
    python scripts/verify_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from app.core.classifier import Classifier
from app.core.face_crop import crop_to_face
from app.storage.local_storage import list_photos, list_registered_students, latest_version


def main() -> None:
    model_path = Path("storage/model/best_model.h5")
    label_map_path = Path("storage/model/label_map.json")
    if not (model_path.is_file() and label_map_path.is_file()):
        print("No trained model found under storage/model/")
        return

    classifier = Classifier(str(model_path), str(label_map_path))

    total = 0
    correct = 0
    confusions: dict[str, dict[str, int]] = {}

    for student_id in list_registered_students():
        version = latest_version(student_id)
        photos = list_photos(student_id, version)
        student_correct = 0

        for photo_path in photos:
            image = cv2.imread(str(photo_path))
            if image is None:
                continue
            # Mirror the inference path exactly: face-crop first, then BGR->RGB.
            image = crop_to_face(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_id, confidence, margin = classifier.predict(image)

            total += 1
            if pred_id == student_id:
                correct += 1
                student_correct += 1
            else:
                confusions.setdefault(student_id, {}).setdefault(pred_id, 0)
                confusions[student_id][pred_id] += 1

            print(
                f"{student_id[:8]} {photo_path.name:16s} -> pred={pred_id[:8]} "
                f"correct={pred_id == student_id} conf={confidence:.4f} margin={margin:.4f}"
            )

        print(f"  == {student_id[:8]} (v{version}): {student_correct}/{len(photos)} correct ==\n")

    print(f"\nOverall: {correct}/{total} correct ({100 * correct / total:.1f}%)" if total else "No photos found")

    if confusions:
        print("\nConfusions (true -> predicted: count):")
        for true_id, preds in confusions.items():
            for pred_id, count in preds.items():
                print(f"  {true_id[:8]} -> {pred_id[:8]}: {count}")


if __name__ == "__main__":
    main()
