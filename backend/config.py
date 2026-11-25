"""Configuration module for RealtyVoice AI backend.

- Determines the project root using pathlib.
- Sets paths for data directory and SQLite database.
- Loads optional environment variables from a .env file.
- Exposes DATABASE_URL which can be overridden via env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (two levels up from this file: backend/config.py -> backend -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure we have the expected absolute path
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = PROJECT_ROOT.resolve()

# Data directory and default DB path
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "realtyvoice.db"

# Create the data directory if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env if present (does not raise if missing)
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# DATABASE_URL can be set in the environment; otherwise fall back to SQLite file URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Use SQLite URI format; ensure absolute path
    DATABASE_URL = f"sqlite+pysqlite:///{DB_PATH.as_posix()}"
# LLM provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
).rstrip("/")


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer, got {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")

    return value


GROQ_TIMEOUT_SECONDS = _positive_int_env("GROQ_TIMEOUT_SECONDS", 30)
AGENT_MAX_TOOL_ITERATIONS = _positive_int_env(
    "AGENT_MAX_TOOL_ITERATIONS",
    4,
)

# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("1", "true", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# CORS: comma-separated list of allowed origins.
# Default allows the local dev server only.
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:8000").strip()
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in _cors_raw.split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Sarvam AI (voice: STT + TTS)
# ---------------------------------------------------------------------------

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_BASE_URL = os.getenv(
    "SARVAM_BASE_URL",
    "https://api.sarvam.ai",
).rstrip("/")

# ---------------------------------------------------------------------------
# Mappls (location and amenity search)
# ---------------------------------------------------------------------------

MAPPLS_API_KEY = os.getenv("MAPPLS_API_KEY", "").strip()
MAPPLS_BASE_URL = os.getenv(
    "MAPPLS_BASE_URL",
    "https://atlas.mappls.com/api",
).rstrip("/")

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DB_PATH",
    "DATABASE_URL",
    "LLM_PROVIDER",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_BASE_URL",
    "GROQ_TIMEOUT_SECONDS",
    "AGENT_MAX_TOOL_ITERATIONS",
    "DEBUG",
    "LOG_LEVEL",
    "CORS_ORIGINS",
    "SARVAM_API_KEY",
    "SARVAM_BASE_URL",
    "MAPPLS_API_KEY",
    "MAPPLS_BASE_URL",
]

