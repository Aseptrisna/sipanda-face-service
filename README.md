# SIPANDA Face Recognition Service

Microservice Python yang **melatih & menjalankan CNN custom dari nol** (bukan model pretrained) untuk pengenalan wajah SIPANDA. Berkomunikasi dengan backend NestJS **hanya lewat HTTP + webhook** — service ini sepenuhnya stateless terhadap MongoDB backend, tidak pernah connect ke Mongo sama sekali.

## Arsitektur

- **Metode: CNN custom dilatih dari nol.** Ini bukan pilihan teknis biasa — ini syarat metodologi riset (skripsi). Model **tidak** memakai embedding pretrained (ArcFace/FaceNet).
- **Reuse langsung project `training metode cnn/`** (sibling folder) — service ini tidak mengimplementasikan ulang pipeline CNN, ia mengimpor & menjalankan modul `split_dataset.py`, `dataloader.py`, `model.py`, `trainer.py` dari project itu apa adanya. Lihat `app/jobs/training_job.py`.
- **Classifier tunggal, closed-set.** Semua siswa yang punya foto ter-upload dilatih dalam **satu model softmax bersama**. Konsekuensi penting: setiap `/training/trigger` = **retrain ulang total** dari nol, bukan incremental per-siswa. Ini pilihan yang disengaja untuk skala kecil (puluhan siswa) — kalau nanti ratusan/ribuan siswa, strategi ini perlu dievaluasi ulang (retrain terjadwal/batch, bukan tiap trigger).
- **Tidak ada face detection/alignment** (MTCNN dsb) — baik saat training maupun inference. Ini supaya preprocessing training dan inference tetap identik (foto mentah di-resize langsung ke 128x128). Konsekuensinya: foto yang dikirim untuk absensi sebaiknya sudah berupa crop wajah yang cukup rapat, bukan foto full-body/latar ramai.
- **State lokal, tanpa Mongo.** Foto tersimpan di disk (`storage/dataset/<student_id>/v{n}/`), model & label map di `storage/model/`, status per-siswa di `storage/model/student_status/<student_id>.json`. Field-field ini tidak pernah ditulis ke MongoDB `siswa` — itu domain backend NestJS.

## Kontrak dengan Backend NestJS

Endpoint & payload berikut **sudah diimplementasikan penuh** di sisi NestJS (`src/face-recognition/`) — service ini harus mengikuti kontrak yang sudah ada, bukan sebaliknya:

| Endpoint | Dipanggil oleh | Keterangan |
|---|---|---|
| `POST /training/upload` | NestJS (`FaceRecognitionClientService.uploadTraining`) | `{student_id, foto_urls, nama_display?}`. NestJS **selalu** kirim `student_id` yang sudah ada (siswa dibuat di NestJS dulu). |
| `POST /training/trigger` | NestJS | `{student_ids}` → retrain penuh, job jalan di background. |
| `GET /training/status/:student_id` | NestJS | Status dari state lokal service ini. |
| `DELETE /training/:student_id` | NestJS | Hapus foto + status lokal siswa itu. |
| `POST /inference/match` | NestJS (saat absensi wajah) | `{image_base64}` → `{student_id, confidence}`. |
| `POST {BACKEND_WEBHOOK_URL}` | **Service ini** → NestJS | Dipanggil balik setelah training selesai: `{student_id, status: 'trained'|'failed', version, model_version?, error_message?}`, header `x-webhook-secret` harus match `FACE_SERVICE_WEBHOOK_SECRET` di NestJS. |

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Isi `.env`:
- `CNN_PROJECT_DIR` — path relatif/absolut ke folder `training metode cnn/` (default sudah pas kalau struktur foldernya sama seperti sekarang: sibling dari `sipanda/`).
- `BACKEND_WEBHOOK_URL` — URL lengkap ke endpoint `training-complete` di backend NestJS yang jalan.
- `FACE_SERVICE_WEBHOOK_SECRET` — **harus identik** dengan `FACE_SERVICE_WEBHOOK_SECRET` di `sipanda/backend/.env`.
- Backend NestJS `.env`-nya sendiri: `FACE_SERVICE_BASE_URL=http://localhost:4001/api` — semua route di service ini dimount di bawah prefix `/api` supaya cocok dengan URL itu tanpa perlu ubah `.env` backend.

```bash
uvicorn app.main:app --port 4001
```

## Status Verifikasi

**Sudah diuji end-to-end nyata** (bukan cuma unit test) dengan 24 siswa asli dari dataset skripsi:
1. Upload foto 24 siswa (via `/training/upload`) ✅
2. Trigger retrain penuh → CNN benar-benar dilatih ulang lewat pipeline `training metode cnn/` ✅
3. Model + label map tersimpan (`storage/model/best_model.h5`, `label_map.json`) ✅
4. Webhook otomatis terkirim ke backend NestJS **yang benar-benar jalan** → `status_wajah` ke-24 siswa berubah jadi `terdaftar` di MongoDB asli ✅
5. `/inference/match` mengembalikan struktur response yang benar (confidence rendah karena data training cuma 3 foto/siswa — limitasi data, bukan bug) ✅
6. `/training/status/:id` dan `DELETE /training/:id` bekerja benar ✅

## Batasan yang Diketahui

- **Akurasi model rendah** dengan dataset saat ini (3 foto training/siswa) — expected, bukan bug. Lihat BAB IV untuk analisis lengkap.
- **Belum ada autentikasi service-to-service** selain webhook secret satu arah (NestJS → sini tidak ada API key).
- **Retrain penuh tiap trigger** — untuk skala besar, ini perlu diganti strategi batch/terjadwal.
