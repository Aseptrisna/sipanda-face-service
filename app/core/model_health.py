"""Post-retrain accuracy sanity check.

Not a proper held-out evaluation (these images were part of the training
set, so this can't catch overfitting) — it only catches the failure modes
actually seen in production: a collapsed model where every class gets
swallowed by whichever class has the most training photos, or a stale/
broken checkpoint that confidently mispredicts even its own training
photos. Shared by scripts/verify_model.py (full per-photo detail, run
manually) and training_job.py (aggregate-only, run automatically after
every retrain so a bad result gets logged without anyone having to
remember to check manually).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from app.core.classifier import Classifier
from app.core.face_crop import crop_to_face
from app.storage.local_storage import list_photos, list_registered_students, latest_version

# Below this overall accuracy, the automatic post-retrain check logs a
# warning. Not a hard science — a healthy few-class model scores >90% on
# its own training photos; real production runs that later turned out to
# have genuine problems (class collapse, imbalance) scored 17-60%.
HEALTH_WARNING_THRESHOLD = 0.75


@dataclass
class PhotoResult:
    photo_name: str
    predicted_id: str
    confidence: float
    margin: float

    def is_correct(self, true_id: str) -> bool:
        return self.predicted_id == true_id


@dataclass
class ModelHealthReport:
    per_student: dict[str, list[PhotoResult]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(results) for results in self.per_student.values())

    @property
    def correct(self) -> int:
        return sum(
            sum(1 for r in results if r.is_correct(student_id))
            for student_id, results in self.per_student.items()
        )

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def student_accuracy(self, student_id: str) -> float:
        results = self.per_student.get(student_id, [])
        if not results:
            return 0.0
        return sum(1 for r in results if r.is_correct(student_id)) / len(results)

    @property
    def is_healthy(self) -> bool:
        return self.total > 0 and self.accuracy >= HEALTH_WARNING_THRESHOLD

    def confusions(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for student_id, results in self.per_student.items():
            for r in results:
                if r.predicted_id != student_id:
                    out.setdefault(student_id, {}).setdefault(r.predicted_id, 0)
                    out[student_id][r.predicted_id] += 1
        return out


def evaluate_model(classifier: Classifier) -> ModelHealthReport:
    """Run `classifier` against every registered student's own latest
    training photos. Mirrors the real inference path exactly: face-crop
    first, then BGR->RGB, same as app/routes/inference.py."""
    report = ModelHealthReport()

    for student_id in list_registered_students():
        version = latest_version(student_id)
        results: list[PhotoResult] = []

        for photo_path in list_photos(student_id, version):
            image = cv2.imread(str(photo_path))
            if image is None:
                continue
            image = crop_to_face(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_id, confidence, margin = classifier.predict(image)
            results.append(PhotoResult(photo_path.name, pred_id, confidence, margin))

        report.per_student[student_id] = results

    return report
