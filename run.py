"""Convenient entry point to launch the FastAPI application.

Running ``python run.py`` is equivalent to:
    python -m uvicorn backend.main:app --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

