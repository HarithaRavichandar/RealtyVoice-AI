import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import Base, get_db
from backend.models import Conversation, Message
from backend.orchestrator.services.message import add_message

@pytest.fixture
def api_client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, TestingSessionLocal
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()

def test_start_conversation(api_client):
    client, _ = api_client
    response = client.post("/api/conversations", json={"channel": "browser_text"})
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"]
    assert data["channel"] == "browser_text"
    assert data["created_at"]

def test_get_existing_conversation(api_client):
    client, _ = api_client
    created = client.post("/api/conversations", json={"channel": "browser_voice"}).json()
    response = client.get("/api/conversations/" + created["session_id"])
    assert response.status_code == 200
    assert response.json()["session_id"] == created["session_id"]

def test_get_unknown_conversation_returns_404(api_client):
    client, _ = api_client
    response = client.get("/api/conversations/does-not-exist")
    assert response.status_code == 404

def test_invalid_channel_returns_422(api_client):
    client, _ = api_client
    response = client.post("/api/conversations", json={"channel": "whatsapp"})
    assert response.status_code == 422

def test_list_conversation_messages(api_client):
    client, session_factory = api_client
    created = client.post("/api/conversations", json={"channel": "browser_text"}).json()
    with session_factory() as db:
        conversation = db.query(Conversation).filter_by(external_session_id=created["session_id"]).one()
        add_message(db, conversation.conversation_id, 1, "I need a 2 BHK in Chennai", "customer")
        add_message(db, conversation.conversation_id, 2, "What is your budget?", "assistant")
        db.commit()
    response = client.get("/api/conversations/" + created["session_id"] + "/messages")
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 2
    assert [m["role"] for m in messages] == ["customer", "assistant"]
    assert [m["turn_id"] for m in messages] == [1, 2]
    assert messages[0]["content"] == "I need a 2 BHK in Chennai"
    assert "tool_arguments" not in messages[0]
    assert "message_id" not in messages[0]

def test_list_messages_for_unknown_conversation_returns_404(api_client):
    client, _ = api_client
    response = client.get("/api/conversations/does-not-exist/messages")
    assert response.status_code == 404
