# RealtyVoice AI

## Project overview
RealtyVoice AI is an AI‑powered real‑estate voice sales assistant.  In **Phase 1A** we provide a minimal FastAPI backend that offers simple health‑check endpoints and a SQLite database foundation.  All higher‑level features (property catalogue, voice interaction, Gemini integration, etc.) will be added in later phases.

## Technology stack
- Python 3.11
- FastAPI
- Uvicorn (ASGI server)
- SQLAlchemy (ORM)
- Pydantic (data validation)
- python‑dotenv (environment variables)
- HTTPX (HTTP client library, for future external calls)
- pytest (testing)

## Prerequisites
- Windows 10 (or any OS that can run Python)
- Git (optional, for version control)
- A working Python 3.11 interpreter

## Setup
1. **Clone the repository** (if you haven’t already) and `cd` into the project root:
   ```powershell
   cd H:\RealtyVoice-AI
   ```
2. **Create and activate a virtual environment** (skip if the `.venv` is already repaired):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # PowerShell
   # or .\.venv\Scripts\activate   # cmd.exe
   ```
3. **Install dependencies**
   ```powershell
   python -m pip install -r requirements.txt
   ```
   This installs FastAPI, Uvicorn, SQLAlchemy, Pydantic, python‑dotenv, HTTPX and pytest.

## Running the application
You can start the server with the convenience script:
```powershell
python run.py
```
or directly via Uvicorn:
```powershell
python -m uvicorn backend.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### Health endpoints
- `GET /health` – returns `{"status":"ok","message":"RealtyVoice backend alive"}`
- `GET /db-health` – runs a lightweight `SELECT 1` against the SQLite DB; returns HTTP 200 on success or HTTP 503 on failure.

## Testing
Run the test suite (uses an in‑memory SQLite database, never touches the production file):
```powershell
python -m pytest -q
```
All tests should pass.

## Current scope (Phase 1A)
- Project structure and FastAPI skeleton
- Configuration handling (`backend/config.py`)
- SQLAlchemy engine and session foundation (`backend/database.py`)
- Simple health endpoints (`backend/main.py`)
- Minimal pytest suite (`tests/test_health.py`)
- Documentation, Git ignore, and run script

Future phases will add ORM models, seed data, richer API routes, voice interaction, Gemini integration, and dashboard UI.

