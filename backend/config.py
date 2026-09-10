"""VoiceShield AI — central configuration (env-driven, free-tier friendly)."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "VoiceShield AI"
    VERSION: str = "1.0.0"
    ENV: str = "development"
    # Comma-separated list of allowed dashboard origins
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:4173"

    # --- Database (Neon PostgreSQL free tier / SQLite fallback) ---
    # postgresql+asyncpg://user:pass@host/db?sslmode=require
    DATABASE_URL: Optional[str] = None
    SQLITE_PATH: str = "voiceshield.db"

    # --- Model ---
    AASIST_ONNX_PATH: str = "inference/model/aasist_l.onnx"
    SPEAKER_ONNX_PATH: Optional[str] = None  # optional real speaker embedding model
    ONNX_INTRA_OP_THREADS: int = 2

    # --- Streaming ---
    SAMPLE_RATE: int = 16000
    WINDOW_MS: int = 300
    HOP_MS: int = 100
    MAX_FRAME_BYTES: int = 192000  # ~6 s of int16 mono @16 kHz per WS frame

    # --- Risk thresholds (role-aware mitigation) ---
    THRESHOLD_CHILD: float = 0.70
    THRESHOLD_ADULT: float = 0.85

    # --- Watermark verifier ---
    WATERMARK_MIN_FREQ: float = 7000.0   # VoLTE-optimized pilot band
    WATERMARK_MAX_FREQ: float = 7500.0
    WATERMARK_SNR_RATIO: float = 12.0    # peak must exceed 12x noise floor

    # --- Cross-session consistency ---
    EMBED_DIM: int = 256
    CS_MIN_REFERENCES: int = 3           # need >= 3 genuine refs before anomaly is raised
    CS_SIMILARITY_FLOOR: float = 0.75    # best-match cosine below this => anomaly
    CS_ANOMALY_BONUS: float = 0.10       # fused-score bonus when speaker mismatch

    # --- Alerting (free tiers only) ---
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    RESEND_API_KEY: Optional[str] = None
    ALERT_FROM_EMAIL: Optional[str] = "voiceshield@resend.dev"
    ALERT_TO_EMAILS: Optional[str] = None  # comma-separated
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_FROM_NUMBER: Optional[str] = None
    TWILIO_TO_NUMBER: Optional[str] = None
    GENERIC_WEBHOOK_URL: Optional[str] = None
    ALERT_TIMEOUT_S: float = 6.0

    # --- Observability ---
    SENTRY_DSN: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.2

    # --- gRPC (optional enterprise channel) ---
    ENABLE_GRPC: bool = False
    GRPC_PORT: int = 50051

    # --- Forensics ---
    REPORTS_DIR: str = "reports"

    # --- Feature extraction constants (spec) ---
    N_MELS: int = 80
    STFT_NFFT: int = 512
    STFT_HOP: int = 128
    WATERMARK_FFT: int = 4096

    @property
    def window_samples(self) -> int:
        return self.SAMPLE_RATE * self.WINDOW_MS // 1000  # 4800

    @property
    def hop_samples(self) -> int:
        return self.SAMPLE_RATE * self.HOP_MS // 1000  # 1600

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
