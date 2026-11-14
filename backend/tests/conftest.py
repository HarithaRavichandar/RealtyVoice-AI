# -*- coding: utf-8 -*-
"""pytest fixtures for backend tests.

Provides a function‑scoped ``dbsession`` fixture that creates an isolated in‑memory
SQLite database using SQLAlchemy ``StaticPool``. The fixture mirrors the production
setup (foreign‑key pragma) but never touches the real ``data/realtyvoice.db``
file.

All ORM model classes are imported via ``backend.models`` so that their tables
are registered with ``Base.metadata`` before ``create_all`` is called.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the declarative base used by the application
from backend.database import Base
# Import all model classes to ensure they are registered with Base.metadata
import backend.models  # noqa: F401


@pytest.fixture(scope="function")
def dbsession():
    """Create a fresh in‑memory SQLite session for each test.

    The engine uses ``StaticPool`` so the same in‑memory database persists for the
    duration of the fixture. Foreign‑key constraints are enforced via the same
    PRAGMA used in the production ``backend.database`` module.
    """
    # In‑memory SQLite engine with required flags
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    # Enable foreign‑key enforcement
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables for the isolated engine
    Base.metadata.create_all(bind=engine)

    # Session factory bound to this engine
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()

