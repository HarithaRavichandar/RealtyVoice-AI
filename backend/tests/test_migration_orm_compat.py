#!/usr/bin/env python
"""ORM compatibility tests for the Phase 4B migration.
Runs against the disposable SQLite database
`data/realtyvoice_migration_test_20261113.sqlite`.
"""

import pathlib
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

# Import ORM models – they use the shared Base metadata
from backend.models.conversation import Conversation
from backend.models.message import Message
from backend.models.callback import CallbackRequest
from backend.models.consent import ConsentRecord

# Resolve the project root (backend/tests -> project root) and locate disposable DB
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DISPOSABLE_DB = PROJECT_ROOT / "data" / "realtyvoice_migration_test_20261113.sqlite"

# Sanity checks – ensure we are not pointing at the production DB and that the file exists
PRODUCTION_DB = PROJECT_ROOT / "data" / "realtyvoice.db"
# The disposable database is a local artefact (databases are git-ignored), so
# on a fresh checkout or in CI it does not exist. Skip then, rather than fail
# at import and break collection of the whole suite.
if not DISPOSABLE_DB.is_file():
    pytest.skip(
        f"Disposable migration test DB not found at {DISPOSABLE_DB}",
        allow_module_level=True,
    )
assert DISPOSABLE_DB != PRODUCTION_DB, "Test must not use the production database"

@pytest.fixture(scope="module")
def engine():
    # Build URL via SQLAlchemy's URL helper (avoids manual string fiddling)
    url = URL.create("sqlite+pysqlite", database=str(DISPOSABLE_DB))
    eng = create_engine(url, future=True, connect_args={"check_same_thread": False})

    # Ensure foreign‑key enforcement on every new DBAPI connection
    @event.listens_for(eng, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Verify pragma is on for a quick sanity check
    with eng.begin() as conn:
        row = conn.execute(text("PRAGMA foreign_keys"))
        assert row.fetchone()[0] == 1, "foreign_keys pragma not enabled on engine"

    yield eng
    eng.dispose()

@pytest.fixture(scope="function")
def dbsession(engine):
    """Provide a session with a surrounding transaction that is rolled back after each test.
    This pattern isolates tests even when the test code calls ``session.commit()``.
    """
    # Open a connection and start an outer transaction
    connection = engine.connect()
    transaction = connection.begin()

    # Bind a session to the connection (no autocommit/autoflush)
    SessionLocal = sessionmaker(bind=connection, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        # Close the session and roll back the outer transaction to discard any changes
        session.close()
        transaction.rollback()
        connection.close()

def test_foreign_key_enforcement(engine):
    # Engine fixture already verified pragma; this is a secondary sanity check
    with engine.begin() as conn:
        row = conn.execute(text("PRAGMA foreign_keys"))
        assert row.fetchone()[0] == 1, "foreign_keys pragma not enabled"

def test_anonymous_conversation_create(dbsession):
    conv = Conversation(
        lead_id=None,
        channel="browser_voice",
        external_session_id="test-session-1",
        temp_requirements={"foo": "bar"},
    )
    dbsession.add(conv)
    dbsession.commit()
    fetched = dbsession.get(Conversation, conv.conversation_id)
    assert fetched is not None
    assert fetched.lead_id is None
    assert fetched.temp_requirements == {"foo": "bar"}

def test_message_creation_with_tool_fields(dbsession):
    # Ensure a conversation exists
    conv = Conversation(
        lead_id=None,
        channel="browser_text",
        external_session_id="msg-test",
        temp_requirements=None,
    )
    dbsession.add(conv)
    dbsession.flush()

    msg = Message(
        conversation_id=conv.conversation_id,
        turn_id=1,
        role="assistant",
        content="Hello",
        tool_name="search",
        tool_call_id="call-123",
        tool_arguments={"q": "real estate"},
        tool_status="completed",
        tool_result={"results": []},
        tool_error=None,
    )
    dbsession.add(msg)
    dbsession.commit()

    fetched = dbsession.get(Message, msg.message_id)
    assert fetched is not None
    assert fetched.tool_name == "search"
    assert fetched.tool_arguments == {"q": "real estate"}

def test_callback_with_nullable_lead_and_conversation(dbsession):
    conv = Conversation(
        lead_id=None,
        channel="telephone",
        external_session_id="cb-test",
        temp_requirements=None,
    )
    dbsession.add(conv)
    dbsession.flush()

    cb = CallbackRequest(
        lead_id=None,
        conversation_id=conv.conversation_id,
        preferred_contact="email",
        preferred_date="2023-01-01",
        preferred_time="10:00",
        status="Pending",
    )
    dbsession.add(cb)
    dbsession.commit()

    fetched = dbsession.get(CallbackRequest, cb.callback_id)
    assert fetched is not None
    assert fetched.lead_id is None
    assert fetched.conversation_id == conv.conversation_id

def test_cascade_delete_conversation_effects(dbsession):
    # Create conversation with linked message, callback, consent
    conv = Conversation(
        lead_id=None,
        channel="browser_voice",
        external_session_id="cascade-test",
        temp_requirements=None,
    )
    dbsession.add(conv)
    dbsession.flush()

    msg = Message(
        conversation_id=conv.conversation_id,
        turn_id=1,
        role="assistant",
        content="Cascade test",
    )
    dbsession.add(msg)

    cb = CallbackRequest(
        lead_id=None,
        conversation_id=conv.conversation_id,
        preferred_contact="phone",
        preferred_date="2023-06-01",
        preferred_time="14:00",
        status="Pending",
    )
    dbsession.add(cb)

    consent = ConsentRecord(
        lead_id=None,
        consent_type="contact_storage",
        granted=True,
        source="web",
        conversation_id=conv.conversation_id,
        policy_version="1.0",
    )
    dbsession.add(consent)
    dbsession.commit()

    # Delete the conversation
    dbsession.delete(conv)
    dbsession.commit()

    # Message should be gone (cascade delete)
    assert dbsession.get(Message, msg.message_id) is None

    # Callback conversation_id should be NULL (ON DELETE SET NULL)
    refreshed_cb = dbsession.get(CallbackRequest, cb.callback_id)
    assert refreshed_cb is not None
    assert refreshed_cb.conversation_id is None

    # Consent conversation_id should be NULL (ON DELETE SET NULL)
    refreshed_consent = dbsession.get(ConsentRecord, consent.consent_id)
    assert refreshed_consent is not None
    assert refreshed_consent.conversation_id is None
