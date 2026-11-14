"""Conversation and message API routes for RealtyVoice AI."""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ..database import get_db
from .schemas import ConversationStartRequest, ConversationStartResponse
from .services.conversation import create_conversation, get_conversation_by_session_id
from .services.message import get_conversation_messages as load_conversation_messages

router = APIRouter()

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    turn_id: int
    role: str
    content: str
    created_at: datetime

@router.post("/conversations", response_model=ConversationStartResponse, status_code=status.HTTP_201_CREATED)
def start_conversation(request: ConversationStartRequest, db: Session = Depends(get_db)):
    try:
        conv = create_conversation(db, request.channel)
        db.commit()
        db.refresh(conv)
        return ConversationStartResponse(session_id=conv.external_session_id, channel=conv.channel, created_at=conv.created_at)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to create conversation") from exc

@router.get("/conversations/{session_id}", response_model=ConversationStartResponse)
def get_conversation(session_id: str, db: Session = Depends(get_db)):
    conv = get_conversation_by_session_id(db, session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationStartResponse(session_id=conv.external_session_id, channel=conv.channel, created_at=conv.created_at)

@router.get("/conversations/{session_id}/messages", response_model=List[MessageResponse])
def list_conversation_messages(session_id: str, db: Session = Depends(get_db)):
    conv = get_conversation_by_session_id(db, session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = load_conversation_messages(db, conv.conversation_id)
    return [MessageResponse(turn_id=m.turn_id, role=m.role, content=m.content, created_at=m.created_at) for m in messages]
