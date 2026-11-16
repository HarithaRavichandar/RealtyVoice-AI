
"""FastAPI application entry point for RealtyVoice AI.

Provides:
- /health       - Backend health check
- /db-health    - Database connectivity check
- /              - Frontend application
- /static/*      - Frontend CSS and JavaScript
- /properties/*  - Property search API
- /api/*         - Agent and conversation API
"""

from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import text

from .database import get_db


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RealtyVoice AI",
    version="0.1.0"
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
