"""FastAPI application entry point for RealtyVoice AI.

Provides two simple health endpoints:
- `/health` – static JSON confirming the service is up.
- `/db-health` – runs a trivial SELECT 1 query using the database session.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import text
from .database import get_db

app = FastAPI(title="RealtyVoice AI", version="0.1.0")

# Register property search routes
from .routes.property_search import router as property_search_router
app.include_router(property_search_router, prefix="/properties")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RealtyVoice backend alive"}

@app.get("/db-health")
def db_health(db=Depends(get_db)):
    try:
        # Execute a lightweight query; works for any DB supporting SELECT 1
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connection healthy"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed",
        ) from exc


from .orchestrator.router import router as orchestrator_router
app.include_router(orchestrator_router, prefix="/api")
