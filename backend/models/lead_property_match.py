"""Association model for many‑to‑many Lead ↔ Property matches.

Fields:
- match_id: primary key
- lead_id: FK to leads (on delete cascade)
- property_id: FK to properties (on delete cascade)
- match_score: Float (nullable)
- created_at: DateTime UTC

Unique constraint on (lead_id, property_id) prevents duplicate matches.
"""
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from ..database import Base
from ._time import utc_now

class LeadPropertyMatch(Base):
    __tablename__ = "lead_property_matches"
    __table_args__ = (
        UniqueConstraint("lead_id", "property_id", name="uq_lead_property"),
        Index("ix_match_lead_id", "lead_id"),
        Index("ix_match_property_id", "property_id"),
    )

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer, ForeignKey("leads.lead_id", ondelete="CASCADE"), nullable=False)
    property_id = Column(Integer, ForeignKey("properties.property_id", ondelete="CASCADE"), nullable=False)
    match_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships back to parent objects
    lead = relationship("Lead", back_populates="property_matches")
    property = relationship("Property", back_populates="lead_matches")

