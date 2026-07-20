"""Trigger a full retrain of ALL registered students.

Retraining is always full (the closed-set softmax can't train per-student),
so this simply collects every student that has uploaded photos and triggers
the job. It calls the RUNNING service's /training/trigger endpoint on
localhost, so the live process does the training and hot-reloads the new
model itself — no `pm2 restart` needed afterwards. It then polls until the
job finishes and prints the result.

Usage (from face-service/, with the service running under pm2):
    venv/bin/python scripts/retrain_all.py

Prerequisite: the service must be up (pm2 status sipanda-face-service).
After it finishes, verify with:  venv/bin/python scripts/verify_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.config import settings
from app.storage.local_storage import list_registered_students

BASE_URL = f"http://127.0.0.1:{settings.port}/api"
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 15 * 60  # a full retrain on a small VPS can take a few minutes


def main() -> None:
    students = list_registered_students()
    if not students:
        print("Tidak ada siswa terdaftar (belum ada foto ter-upload) — tidak ada yang dilatih.")
        return

    print(f"Retrain {len(students)} siswa: {', '.join(s[:8] for s in students)}")

    try:
        resp = httpx.post(
            f"{BASE_URL}/training/trigger",
            json={"student_ids": students},
            timeout=30,
        )
    except httpx.ConnectError:
        print(
            f"Gagal konek ke service di {BASE_URL}. Pastikan service jalan: "
            "pm2 status sipanda-face-service"
        )
        sys.exit(1)

    if resp.status_code == 409:
        print("Ada training lain yang sedang berjalan. Tunggu sampai selesai lalu coba lagi.")
        sys.exit(1)
    resp.raise_for_status()

    job_id = resp.json()["job_id"]
    print(f"Job dimulai: {job_id}. Menunggu selesai (bisa beberapa menit)...")

    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
        try:
            r = httpx.get(f"{BASE_URL}/training/jobs/{job_id}", timeout=30)
        except httpx.HTTPError as exc:
            print(f"  (gagal cek status: {exc}, coba lagi)")
            continue

        if r.status_code == 404:
            continue  # background task belum sempat menulis status awal
        status = r.json()
        state = status.get("status")

        if state in ("running", None):
            print("  ...masih training")
            continue

        if state == "success":
            included = status.get("included_students", [])
            print(f"\nSELESAI ✓ — model versi {status.get('model_version')} "
                  f"({len(included)} siswa dilatih).")
            print("Model baru sudah otomatis aktif. Cek akurasi: "
                  "venv/bin/python scripts/verify_model.py")
        else:
            print(f"\nGAGAL ✗ — {status.get('error_message', status)}")
        return

    print("Timeout menunggu training selesai. Cek log: pm2 logs sipanda-face-service")


if __name__ == "__main__":
    main()
