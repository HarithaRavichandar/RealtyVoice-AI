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

__all__ = ["PROJECT_ROOT", "DATA_DIR", "DB_PATH", "DATABASE_URL"]

