"""Database foundation for RealtyVoice AI backend.

- Creates the SQLAlchemy engine using the DATABASE_URL from config.
- Enforces SQLite foreign‑key constraints on each connection.
- Provides SessionLocal (sessionmaker) and Base (declarative base).
- Supplies a FastAPI dependency `get_db` that yields a session.
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

# Engine with SQLite specific flags
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

# Enable foreign keys for SQLite connections
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

# Base class for ORM models
Base = declarative_base()

def get_db():
    """FastAPI dependency that provides a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

__all__ = ["engine", "SessionLocal", "Base", "get_db"]

