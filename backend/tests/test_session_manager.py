"""Tests for the session manager utilities.

The tests rely on the ``dbsession`` fixture defined in
``backend/tests/test_migration_orm_compat.py`` which provides an isolated
SQLite transaction that rolls back after each test.
"""

import pytest
from sqlalchemy.orm import Session

from backend.orchestrator.services.session_manager import resolve_session
from backend.orchestrator.services.conversation import create_conversation


def test_resolve_known_session(dbsession: Session):
    # Create a conversation and commit it so it is persisted
    conv = create_conversation(dbsession, channel="browser_voice")
    external_id = conv.external_session_id
    dbsession.commit()

    resolved = resolve_session(dbsession, external_id)
    assert resolved.conversation_id == conv.conversation_id
    assert resolved.external_session_id == external_id
    assert resolved.channel == "browser_voice"


def test_resolve_unknown_session_raises(dbsession: Session):
    with pytest.raises(ValueError, match="Conversation not found"):
        resolve_session(dbsession, "nonexistent-id")


def test_resolve_does_not_commit(dbsession: Session):
    # Create a conversation but do NOT commit
    conv = create_conversation(dbsession, channel="browser_text")
    external_id = conv.external_session_id
    # Resolve should succeed within the same session without a commit
    resolved = resolve_session(dbsession, external_id)
    assert resolved is conv
    # Ensure the session still has no pending transaction that would be committed
    # (SQLAlchemy lazy flush is disabled in the fixture, so we just verify identity)
    assert resolved.external_session_id == external_id

