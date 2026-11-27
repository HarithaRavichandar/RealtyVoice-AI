
"""FastAPI application entry point for RealtyVoice AI.

Provides:
- /health       - Backend health check
- /db-health    - Database connectivity check
- /              - Frontend application
- /static/*      - Frontend CSS and JavaScript
- /properties/*  - Property search API
- /api/*         - Agent and conversation API
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import text

from .config import CORS_ORIGINS, DEBUG, LOG_LEVEL
from .database import get_db


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": LOG_LEVEL,
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"level": LOG_LEVEL, "handlers": ["console"], "propagate": False},
        "sqlalchemy.engine": {"level": "WARNING", "handlers": ["console"], "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RealtyVoice AI starting up (debug=%s, log_level=%s)", DEBUG, LOG_LEVEL)
    try:
        from backend.database import engine
        from backend.services.schema_upgrades import ensure_lead_qualification_columns

        ensure_lead_qualification_columns(engine)
    except Exception:  # never block startup on an optional upgrade
        logger.exception("Lead qualification schema upgrade failed")
    yield
    logger.info("RealtyVoice AI shutting down")


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RealtyVoice AI",
    version="0.1.0",
    debug=DEBUG,
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS MIDDLEWARE
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# --------------------------------------------------
# FRONTEND STATIC FILES
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the RealtyVoice AI frontend."""

    index_file = FRONTEND_DIR / "index.html"

    return FileResponse(index_file)


# --------------------------------------------------
# PROPERTY SEARCH ROUTES
# --------------------------------------------------

from .routes.property_search import router as property_search_router

app.include_router(
    property_search_router,
    prefix="/properties"
)


# --------------------------------------------------
# HEALTH CHECK ROUTES
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "RealtyVoice backend alive"
    }


@app.get("/db-health")
def db_health(db=Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "message": "Database connection healthy"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        ) from exc


# --------------------------------------------------
# AGENT AND CONVERSATION ROUTES
# --------------------------------------------------

from .orchestrator.router import router as orchestrator_router

app.include_router(
    orchestrator_router,
    prefix="/api"
)


# --------------------------------------------------
# VOICE API ROUTES (Sarvam STT + TTS)
# --------------------------------------------------

from .orchestrator.voice_router import voice_router

app.include_router(
    voice_router,
    prefix="/api"
)
