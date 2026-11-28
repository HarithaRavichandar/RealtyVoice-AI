"""Configuration module for RealtyVoice AI backend.

- Determines the project root using pathlib.
- Sets paths for data directory and SQLite database.
- Loads optional environment variables from a .env file.
- Exposes DATABASE_URL which can be overridden via env.
"""
import os
import re
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


def parse_cors_origins(raw: str) -> list[str]:
    """Parse and validate a comma-separated list of allowed origins.

    Each origin must be ``scheme://host[:port]`` with no path. A wildcard is
    refused: this API is not meant to be callable from any website, and a
    wildcard combined with credentials is unsafe. Trailing slashes are
    removed and duplicates dropped.
    """
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError(
                "CORS_ORIGINS must list explicit origins; '*' is not allowed"
            )
        if not re.fullmatch(r"https?://[A-Za-z0-9.-]+(:\d{1,5})?", origin):
            raise ValueError(
                f"Invalid CORS origin {item.strip()!r}: use scheme://host[:port] "
                "with no path"
            )
        if origin not in origins:
            origins.append(origin)
    return origins


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


CORS_ORIGINS: list[str] = parse_cors_origins(_cors_raw)
# Credentials (cookies) are not used by this application, so they are off
# unless explicitly enabled.
CORS_ALLOW_CREDENTIALS = _bool_env("CORS_ALLOW_CREDENTIALS", False)

# ---------------------------------------------------------------------------
# Request protection
# ---------------------------------------------------------------------------

LOG_FORMAT = os.getenv("LOG_FORMAT", "json").strip().lower()
if LOG_FORMAT not in ("json", "text"):
    raise ValueError("LOG_FORMAT must be 'json' or 'text'")

# Rate limits are per client per minute. Set RATE_LIMIT_ENABLED=false to turn off.
RATE_LIMIT_ENABLED = _bool_env("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_AGENT_PER_MINUTE = _positive_int_env("RATE_LIMIT_AGENT_PER_MINUTE", 30)
RATE_LIMIT_VOICE_PER_MINUTE = _positive_int_env("RATE_LIMIT_VOICE_PER_MINUTE", 20)
RATE_LIMIT_DEFAULT_PER_MINUTE = _positive_int_env("RATE_LIMIT_DEFAULT_PER_MINUTE", 120)
# Only trust X-Forwarded-For when running behind a proxy you control.
TRUST_PROXY_HEADERS = _bool_env("TRUST_PROXY_HEADERS", False)

# Request body limits in bytes. Agent/JSON bodies are tiny (a 4000-character
# message); voice uploads carry audio plus multipart overhead.
MAX_JSON_BODY_BYTES = _positive_int_env("MAX_JSON_BODY_BYTES", 64 * 1024)
MAX_AUDIO_UPLOAD_BYTES = _positive_int_env("MAX_AUDIO_UPLOAD_BYTES", 10 * 1024 * 1024)

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
    "https://search.mappls.com/search",
).rstrip("/")

# ---------------------------------------------------------------------------
# Live property inventory (API.SH, which relays 99acres listings)
# ---------------------------------------------------------------------------
# Not wired into the agent yet. The key is read from the environment only.

APISH_API_KEY = os.getenv("APISH_API_KEY", "").strip()
APISH_BASE_URL = os.getenv(
    "APISH_BASE_URL",
    "https://api.api.sh/99acres/v1",
).rstrip("/")
APISH_TIMEOUT_SECONDS = _positive_int_env("APISH_TIMEOUT_SECONDS", 15)
# Initial policy: 3 credits per conversation, subject to confirmation against
# the API.SH plan. 0 disables live searches.


def _non_negative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


LIVE_CREDITS_PER_CONVERSATION = _non_negative_int_env("LIVE_CREDITS_PER_CONVERSATION", 3)

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
    "CORS_ALLOW_CREDENTIALS",
    "parse_cors_origins",
    "LOG_FORMAT",
    "RATE_LIMIT_ENABLED",
    "RATE_LIMIT_AGENT_PER_MINUTE",
    "RATE_LIMIT_VOICE_PER_MINUTE",
    "RATE_LIMIT_DEFAULT_PER_MINUTE",
    "TRUST_PROXY_HEADERS",
    "MAX_JSON_BODY_BYTES",
    "MAX_AUDIO_UPLOAD_BYTES",
    "SARVAM_API_KEY",
    "SARVAM_BASE_URL",
    "MAPPLS_API_KEY",
    "MAPPLS_BASE_URL",
    "APISH_API_KEY",
    "APISH_BASE_URL",
    "APISH_TIMEOUT_SECONDS",
    "LIVE_CREDITS_PER_CONVERSATION",
]

