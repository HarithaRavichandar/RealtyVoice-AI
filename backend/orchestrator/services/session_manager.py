"""Session manager utilities for the orchestrator.

Provides a thin wrapper to resolve a public external session identifier to the
internal ``Conversation`` ORM object.  No database write operations are performed.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ...models import Conversation


def resolve_session(session: Session, external_id: str) -> Conversation:
    """Resolve an external ``session_id`` to a ``Conversation``.

    Args:
        session: SQLAlchemy Session (read‑only for this operation).
        external_id: The opaque ``external_session_id`` stored on the
            ``Conversation`` model.

    Returns:
        The ``Conversation`` ORM instance.

    Raises:
        ValueError: If no conversation with the supplied ``external_id`` exists.
    """
    conv = (
        session.query(Conversation)
        .filter(Conversation.external_session_id == external_id)
        .first()
    )
    if conv is None:
        raise ValueError("Conversation not found")
    return conv

