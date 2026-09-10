# Environment Variables

Copy this block into a local `.env` (git-ignored) or set them in your hosting
provider's env UI. All services have free tiers — no credit card anywhere.

```bash
# --- Database -------------------------------------------------------------
# Neon PostgreSQL free tier (no credit card): https://neon.tech
# Required for pgvector cross-session speaker consistency.
# Leave empty to use local SQLite fallback (dev only).
DATABASE_URL=

# --- Model ----------------------------------------------------------------
AASIST_ONNX_PATH=inference/model/aasist_l.onnx
ONNX_INTRA_OP_THREADS=2

# --- Risk thresholds (role-aware mitigation) --------------------------------
THRESHOLD_CHILD=0.70
THRESHOLD_ADULT=0.85

# --- Alerts (free tiers where possible) ------------------------------------
# Telegram Bot API (fully free): create bot via @BotFather, then chat id via /getUpdates
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
# Resend free tier (3,000 emails/mo): https://resend.com
RESEND_API_KEY=
ALERT_FROM_EMAIL=onboarding@resend.dev
ALERT_TO_EMAILS=
# Twilio SMS: trial/test credentials are free; production SMS has a cost.
# Enabled only when all four keys are set; otherwise ignored.
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=+10000000000
TWILIO_TO_NUMBER=
# Any generic JSON webhook sink (enterprise SIEM, Discord, Slack, n8n...)
GENERIC_WEBHOOK_URL=

# --- Observability ----------------------------------------------------------
# Sentry free tier (5k errors/mo) — create a Python/FastAPI project in the
# Sentry UI, platform = Python (FastAPI integration), then set:
SENTRY_DSN=

# --- Optional enterprise channel --------------------------------------------
ENABLE_GRPC=false
GRPC_PORT=50051
```

## Notes

* `DATABASE_URL` — paste your Neon connection string (`postgresql://user:pass@ep-...neon.tech/neondb?sslmode=require`).
  The app rewrites it to `postgresql+asyncpg://` automatically and enables pgvector on boot.
* Alerting degrades gracefully: with no keys set, incidents are still recorded and
  visible in the dashboard; alerts simply log "no channels configured".
* SMS lives behind a four-key gate (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
  `TWILIO_FROM_NUMBER`, `TWILIO_TO_NUMBER`). None of the Twilio channel code is
executed unless all four are set — Twilio test credentials + a trial number are
sufficient to validate end-to-end without paying for live SMS.
* For the Sentry project: create one with platform **Python → FastAPI**, then set
  `SENTRY_DSN`.
* For the report's "Sentry → create project → platform" question: select
  **Python → FastAPI** (server) or **JavaScript → React** (frontend). Set the DSN
  in `SENTRY_DSN`.
