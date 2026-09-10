"""VoiceShield AI — central configuration (Pydantic Settings, env-driven).

All values are optional at boot: the system degrades gracefully to
local-only mode when DATABASE_URL / alert credentials are absent.
Secrets are NEVER hardcoded — set them in the Freebuff environment UI.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # ---- Database (Neon PostgreSQL, free tier, pgvector) ----
    database_url: str | None = None  # postgresql+asyncpg://user:pass@host/db?sslmode=require

    # ---- Alerts (all free) ----
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    webhook_urls: str = ""  # comma-separated POST endpoints (Slack/Discord/custom)
    # Email delivery of forensic PDFs (Resend free tier, 100/day)
    resend_api_key: str | None = None
    resend_from_email: str = "VoiceShield AI <onboarding@resend.dev>"
    alert_email_recipients: str = ""  # comma-separated

    # ---- Monitoring (Sentry free tier) ----
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1  # stay within free tier

    # ---- ONNX model ----
    onnx_model_path: str = "inference/model/aasist_l.onnx"
    onnx_num_threads: int = 2

    # ---- Risk thresholds (role-aware) ----
    adult_threshold: float = 0.85
    child_threshold: float = 0.70

    # ---- Cross-session consistency (pgvector cosine) ----
    cross_session_enabled: bool = False
    cross_session_similarity_threshold: float = 0.30  # cosine distance
    cross_session_min_windows: int = 300  # 300 * 100ms hop = 30 s of audio
    embedding_dim: int = 64

    # ---- Audio pipeline ----
    sample_rate: int = 16000
    window_size: int = 4800   # 300 ms @ 16 kHz
    hop_size: int = 1600      # 100 ms hop (66% overlap)

    # ---- Mitigation / alerting behaviour ----
    alert_cooldown_seconds: float = 5.0
    max_alerts_per_session: int = 2

    # ---- Optional gRPC for enterprise integrations ----
    grpc_enabled: bool = False
    grpc_port: int = 50051

    # ---- Frontend ----
    frontend_dist: str = "frontend/dist"


@lru_cache
def get_settings() -> Settings:
    return Settings()
