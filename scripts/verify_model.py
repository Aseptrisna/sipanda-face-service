"""Sanity-check the currently loaded model against its own training photos.

Not a proper held-out evaluation (these images were part of the training
set, so this can't catch overfitting) — it only catches the failure mode
we saw before the last retrain: a stale/broken model confidently
mispredicting even the photos it was trained on. Run after every retrain
before testing from the frontend.

This is the manual, verbose (per-photo) counterpart to the automatic
check training_job.py now runs after every retrain (see
app/core/model_health.py, which this script shares its evaluation logic
with) — run this when you want the full picture, not just the summary.

Usage (from face-service/, with venv activated):
    python scripts/verify_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.classifier import Classifier
from app.core.model_health import evaluate_model


def main() -> None:
    model_path = Path("storage/model/best_model.h5")
    label_map_path = Path("storage/model/label_map.json")
    if not (model_path.is_file() and label_map_path.is_file()):
        print("No trained model found under storage/model/")
        return

    classifier = Classifier(str(model_path), str(label_map_path))
    report = evaluate_model(classifier)

    for student_id, results in report.per_student.items():
        for r in sorted(results, key=lambda r: r.photo_name):
            print(
                f"{student_id[:8]} {r.photo_name:16s} -> pred={r.predicted_id[:8]} "
                f"correct={r.is_correct(student_id)} conf={r.confidence:.4f} margin={r.margin:.4f}"
            )
        print(
            f"  == {student_id[:8]}: {sum(1 for r in results if r.is_correct(student_id))}"
            f"/{len(results)} correct ==\n"
        )

    if report.total:
        print(f"\nOverall: {report.correct}/{report.total} correct ({100 * report.accuracy:.1f}%)")
    else:
        print("No photos found")

    confusions = report.confusions()
    if confusions:
        print("\nConfusions (true -> predicted: count):")
        for true_id, preds in confusions.items():
            for pred_id, count in preds.items():
                print(f"  {true_id[:8]} -> {pred_id[:8]}: {count}")


if __name__ == "__main__":
    main()
