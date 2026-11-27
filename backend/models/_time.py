"""Shared UTC timestamp helper for SQLAlchemy model defaults.

Provides a timezone-aware UTC timestamp function to replace the deprecated
``datetime.utcnow()`` which is scheduled for removal in a future Python version.

Usage in model columns::

    from backend.models._time import utc_now

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)
