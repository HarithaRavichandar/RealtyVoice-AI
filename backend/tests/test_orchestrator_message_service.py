from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Conversation
from backend.orchestrator.services.message import add_message, add_tool_message, get_conversation_messages

@pytest.fixture
def dbsession():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()

def make_conversation(session, session_id="test-session"):
    conv = Conversation(lead_id=None, channel="browser_text", external_session_id=session_id, temp_requirements=None)
    session.add(conv)
    session.flush()
    return conv

def test_customer_message_creation(dbsession):
    conv = make_conversation(dbsession)
    msg = add_message(dbsession, conv.conversation_id, 1, "Hello", "customer")
    assert msg.message_id is not None
    assert msg.role == "customer"
    assert msg.content == "Hello"
    assert msg.tool_name is None

def test_assistant_message_creation(dbsession):
    conv = make_conversation(dbsession)
    msg = add_message(dbsession, conv.conversation_id, 1, "Welcome", "assistant")
    assert msg.role == "assistant"
    assert msg.content == "Welcome"

def test_invalid_role_rejected(dbsession):
    conv = make_conversation(dbsession)
    with pytest.raises(ValueError, match="Invalid role"):
        add_message(dbsession, conv.conversation_id, 1, "Invalid", "tool")

def test_tool_message_structured_audit(dbsession):
    conv = make_conversation(dbsession)
    args = {"location": "Chennai", "budget": 9000000}
    result = {"matches": [{"property_id": 12}]}
    msg = add_tool_message(dbsession, conv.conversation_id, 1, "Search executed", tool_name="search_properties", tool_call_id="call-1", tool_arguments=args, tool_status="completed", tool_result=result)
    assert msg.role == "tool"
    assert msg.tool_name == "search_properties"
    assert msg.tool_call_id == "call-1"
    assert msg.tool_arguments == args
    assert msg.tool_result == result
    assert msg.tool_status == "completed"

def test_messages_ordered_by_turn_and_message_id(dbsession):
    conv = make_conversation(dbsession)
    second = add_message(dbsession, conv.conversation_id, 2, "Second turn", "assistant")
    first = add_message(dbsession, conv.conversation_id, 1, "First message", "customer")
    same_turn = add_message(dbsession, conv.conversation_id, 1, "Same turn reply", "assistant")
    messages = get_conversation_messages(dbsession, conv.conversation_id)
    assert [m.message_id for m in messages] == [first.message_id, same_turn.message_id, second.message_id]

def test_missing_conversation_rejected_on_creation(dbsession):
    with pytest.raises(ValueError, match="Conversation not found"):
        add_message(dbsession, 999999, 1, "Orphan", "customer")
    with pytest.raises(ValueError, match="Conversation not found"):
        add_tool_message(dbsession, 999999, 1, "Orphan tool")

def test_missing_conversation_rejected_on_retrieval(dbsession):
    with pytest.raises(ValueError, match="Conversation not found"):
        get_conversation_messages(dbsession, 999999)

def test_service_does_not_commit_transaction(dbsession):
    conv = make_conversation(dbsession)
    conversation_id = conv.conversation_id
    add_message(dbsession, conversation_id, 1, "Rollback this", "customer")
    dbsession.rollback()
    assert dbsession.get(Conversation, conversation_id) is None
