# VoiceShield AI

Real-time, zero-trust audio deepfake detection for the Indian telecom context —
AASIST-L ONNX inference, prosodic XAI, enterprise watermark verification,
cross-session speaker consistency (pgvector), role-aware mitigation and
I4C-format forensic PDF reports. **Runs entirely on free-tier services.**

## Architecture

```
Browser/Mic ──► WebSocket /ws/stream/{call_id} ──► RingBuffer (300 ms window, 100 ms hop)
                                                    │
                    ┌───────────────────────────────┤
                    ▼                               ▼
          Watermark verifier            AASIST-L ONNX (INT8, 2 threads)
          (4096-pt FFT, 7.0–7.5 kHz)    │  + prosodic XAI (jitter/shimmer/φ_cont)
                    │                   │  + speaker consistency (pgvector)
                    ▼                   ▼
              verified_enterprise   Risk fusion → synthetic_score
                                        │
                         ┌──────────────┼────────────────┐
                         ▼              ▼                ▼
                  Role thresholds   Incident DB     Alerts (Telegram /
                  child ≥0.70       (SQLAlchemy)    Resend email / webhook)
                  adult ≥0.85                          │
                                                       ▼
                                            I4C forensic PDF (ReportLab, IST)
```

## Quick start (dev)

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000 — dashboard served by FastAPI
```

Without `DATABASE_URL` the app runs on a local SQLite file; with Neon Postgres it
enables pgvector cross-session checks automatically.

## Training on Indian voices (en / hi / kn)

```bash
# 1. Data (all free): Common Voice en/hi/kn (bonafide) + any TTS (edge-tts
#    en-IN/hi-IN/kn-IN) or Kaggle deepfake sets (synthetic), laid out as:
#    data/<lang>/{bonafide,synthetic}/*.wav
# 2. Train (CPU fine):
pip install torch onnx onnxruntime onnxruntime-quantization
python -m training.train --data data --epochs 12
# 3. Deploy weights:
cp training/weights/aasist_l_int8.onnx backend/inference/model/aasist_l.onnx
```

The pipeline includes the full telecom augmentation stack (G.711 A-law/µ-law,
AMR-NB/AMR-WB round-trips, noise injection, packet-loss concealment, 300–3,400 Hz
bandwidth truncation) at 30–60% per-batch probability. Until trained weights are
dropped in, the backend uses a deterministic spectral-heuristic so every feature
stays testable.

## API surface

| Endpoint | Purpose |
| --- | --- |
| `WS /ws/stream/{call_id}?role=&language=&speaker=` | Real-time PCM analysis |
| `GET /api/health` | DB + pgvector status |
| `GET /api/sessions` / `/api/sessions/{id}/windows` | Session forensics |
| `GET /api/incidents` / `/{id}/report` | Incident list + I4C PDF |
| `POST /api/analyze` | One-shot file analysis |
| `POST /api/speakers/register` | Enroll trusted speaker voice |
| gRPC :50051 (`ENABLE_GRPC=true`) | Enterprise/banking contract (proto-first) |

## Free-tier integrations

* **Neon** — Postgres + pgvector (no credit card)
* **Resend** — 3,000 alert emails/month
* **Telegram Bot API** — unlimited free alerts
* **Sentry** — 5k errors/month (create a Python project in the Sentry UI, set `SENTRY_DSN`)
* No SMS anywhere — no credible free tier exists; email + Telegram + webhooks cover it.
