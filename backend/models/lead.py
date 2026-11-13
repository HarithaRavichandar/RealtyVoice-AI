"""SQLAlchemy model for Lead.

Fields:
- lead_id: primary key (int)
- customer_name, phone, email: nullable strings
- budget_min, budget_max: Numeric (INR) nullable (allow progressive creation)
- preferred_location, property_type, configuration, possession_preference: nullable strings
- qualification_score: Numeric nullable
- qualification_status: Enum (Hot, Warm, Cold, Unqualified) not nullable
- pipeline_status: Enum (New, Contacted, Visit Scheduled, Negotiation, Converted, Lost) not nullable
- created_at, updated_at: DateTime UTC
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Numeric, DateTime, Enum, Index
from sqlalchemy.orm import relationship
from ..database import Base

class Lead(Base):
    __tablename__ = "leads"
    __table_args__ = (
        Index("ix_leads_preferred_location", "preferred_location"),
        Index("ix_leads_property_type", "property_type"),
    )

    lead_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    budget_min = Column(Numeric(precision=14, scale=2), nullable=True)
    budget_max = Column(Numeric(precision=14, scale=2), nullable=True)
    preferred_location = Column(String, nullable=True)
    property_type = Column(String, nullable=True)
    configuration = Column(String, nullable=True)
    possession_preference = Column(String, nullable=True)
    qualification_score = Column(Numeric(precision=5, scale=2), nullable=True)
    qualification_status = Column(
        Enum("Hot", "Warm", "Cold", "Unqualified", name="qualification_status"),
        nullable=False,
    )
    pipeline_status = Column(
        Enum(
            "New",
            "Contacted",
            "Visit Scheduled",
            "Negotiation",
            "Converted",
            "Lost",
            name="pipeline_status",
        ),
        nullable=False,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    conversations = relationship(
        "Conversation",
        back_populates="lead",
        cascade="all, delete-orphan",
    )
    property_matches = relationship(
        "LeadPropertyMatch",
        back_populates="lead",
        cascade="all, delete-orphan",
    )
    callbacks = relationship(
        "CallbackRequest",
        back_populates="lead",
        cascade="save-update, merge",
    )
    consents = relationship(
        "ConsentRecord",
        back_populates="lead",
        cascade="save-update, merge",
    )

