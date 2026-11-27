"""Shared UTC timestamp helper for SQLAlchemy model defaults.

Provides a timezone-aware UTC timestamp function to replace the deprecated
``datetime.utcnow()`` which is scheduled for removal in a future Python version.

SQLite compatibility note
-------------------------
SQLite stores datetime values as TEXT (ISO-8601 strings) or REAL/INTEGER and
does **not** preserve timezone information.  When SQLAlchemy reads a datetime
back from SQLite it returns a **naive** ``datetime`` (``tzinfo is None``)
regardless of whether the value written was timezone-aware.

This means:

- The ``default`` / ``onupdate`` callables receive a timezone-aware UTC
  datetime from ``utc_now()`` — this is correct and avoids the deprecated
  ``datetime.utcnow()`` call.
- After a round-trip through SQLite the returned datetime is naive but still
  represents UTC (the numeric value is unchanged).
- Application code that compares timestamps (``>=``, ``==``, ``>``) works
  correctly because both sides are naive-UTC after persistence.
- The column type remains ``DateTime`` (without ``timezone=True``) to avoid
  a schema migration.  Switching to ``DateTime(timezone=True)`` would be a
  schema change that requires a migration and is deferred intentionally.

Usage in model columns::

    from backend.models._time import utc_now

    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    The returned value has ``tzinfo=timezone.utc``.  SQLite will strip the
    timezone on storage, but the UTC instant is preserved.
    """
    return datetime.now(timezone.utc)
