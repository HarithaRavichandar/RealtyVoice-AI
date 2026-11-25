
"""API integration tests for the RealtyVoice AI agent endpoint."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base, get_db
from backend.models import Conversation, Message
from backend.orchestrator.schemas import CustomerRequirements
from backend.orchestrator.services.message import add_message


@pytest.fixture
def api_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client, TestingSessionLocal
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()


def create_test_conversation(client):
    response = client.post(
        "/api/conversations",
        json={"channel": "browser_text"},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_agent_turn_persists_customer_and_assistant_messages(
    api_client,
    monkeypatch,
):
    client, session_factory = api_client
    session_id = create_test_conversation(client)

    async def fake_run_agent_loop(session, session_id, incoming_message):
        conversation = (
            session.query(Conversation)
            .filter_by(external_session_id=session_id)
            .one()
        )

        add_message(
            session=session,
            conversation_id=conversation.conversation_id,
            turn_id=1,
            content=incoming_message,
            role="customer",
        )

        return SimpleNamespace(
            session_id=session_id,
            conversation_id=conversation.conversation_id,
            turn_id=1,
            reply="I can help you find a property.",
            spoken_reply="I can help you find a property.",
            requirements=CustomerRequirements(),
            tool_results=[],
            tool_iterations=0,
            iteration_limit_reached=False,
        )

    monkeypatch.setattr(
        "backend.orchestrator.router.run_agent_loop",
        fake_run_agent_loop,
    )

    response = client.post(
        "/api/agent/turn",
        json={
            "session_id": session_id,
            "message": "I need a property in Bengaluru",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["session_id"] == session_id
    assert data["turn_id"] == 1
    assert data["reply"] == "I can help you find a property."
    assert data["intent"] == "general_question"
    assert data["matched_properties"] == []
    assert data["next_action"]["action"] == "await_user"

    with session_factory() as db:
        conversation = (
            db.query(Conversation)
            .filter_by(external_session_id=session_id)
            .one()
        )

        messages = (
            db.query(Message)
            .filter_by(conversation_id=conversation.conversation_id)
            .order_by(Message.message_id)
            .all()
        )

        assert len(messages) == 2
        assert messages[0].role == "customer"
        assert messages[0].content == "I need a property in Bengaluru"
        assert messages[1].role == "assistant"
        assert messages[1].content == "I can help you find a property."
        assert messages[0].turn_id == messages[1].turn_id == 1


def test_agent_turn_returns_404_for_unknown_session(
    api_client,
    monkeypatch,
):
    client, _ = api_client

    async def should_not_run(**kwargs):
        pytest.fail("Agent service should not run for an unknown session")

    monkeypatch.setattr(
        "backend.orchestrator.router.run_agent_loop",
        should_not_run,
    )

    response = client.post(
        "/api/agent/turn",
        json={
            "session_id": "unknown-session",
            "message": "Find me a property",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"


def test_agent_turn_rejects_invalid_request(api_client):
    client, _ = api_client

    response = client.post(
        "/api/agent/turn",
        json={
            "session_id": "valid-session",
            "message": "   ",
        },
    )

    assert response.status_code == 422


def test_agent_turn_rolls_back_when_agent_fails(
    api_client,
    monkeypatch,
):
    client, session_factory = api_client
    session_id = create_test_conversation(client)

    async def failing_run_agent_loop(session, session_id, incoming_message):
        conversation = (
            session.query(Conversation)
            .filter_by(external_session_id=session_id)
            .one()
        )

        add_message(
            session=session,
            conversation_id=conversation.conversation_id,
            turn_id=1,
            content=incoming_message,
            role="customer",
        )

        raise RuntimeError("Simulated provider failure")

    monkeypatch.setattr(
        "backend.orchestrator.router.run_agent_loop",
        failing_run_agent_loop,
    )

    response = client.post(
        "/api/agent/turn",
        json={
            "session_id": session_id,
            "message": "Find me a property",
        },
    )

    assert response.status_code == 502

    with session_factory() as db:
        conversation = (
            db.query(Conversation)
            .filter_by(external_session_id=session_id)
            .one()
        )

        messages = (
            db.query(Message)
            .filter_by(conversation_id=conversation.conversation_id)
            .all()
        )

        assert messages == []
