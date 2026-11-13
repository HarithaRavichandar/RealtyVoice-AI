"""SQLAlchemy model for Property.

Fields:
- property_id: primary key (int)
- project_name, location, property_type, configuration: strings (project_name and configuration nullable as per migration)
- price_min, price_max: Numeric (INR) for precise monetary values (now nullable)
- listing_price: Numeric (advertised price of a listing, nullable)
- possession_date: DateTime (UTC, nullable)
- amenities, nearby_locations: JSON (list of strings, nullable)
- availability_status, description: strings (nullable)
- provenance fields (nullable): source_name, source_record_id, source_url, source_location, source_update_date, import_timestamp, raw_source_metadata
- created_at, updated_at: DateTime (UTC, default now)

Indexes on location, property_type, configuration for faster search.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    DateTime,
    JSON,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from ..database import Base


class Property(Base):
    __tablename__ = "properties"
    __table_args__ = (
        Index("ix_properties_location", "location"),
        Index("ix_properties_property_type", "property_type"),
        Index("ix_properties_configuration", "configuration"),
        UniqueConstraint("source_name", "source_record_id", name="uq_source_provenance"),
    )

    property_id = Column(Integer, primary_key=True, autoincrement=True)
    project_name = Column(String, nullable=True)
    location = Column(String, nullable=False)
    property_type = Column(String, nullable=False)
    configuration = Column(String, nullable=True)
    price_min = Column(Numeric(precision=14, scale=2), nullable=True)
    price_max = Column(Numeric(precision=14, scale=2), nullable=True)
    listing_price = Column(Numeric(precision=14, scale=2), nullable=True)
    possession_date = Column(DateTime, nullable=True)
    amenities = Column(JSON, nullable=True)
    nearby_locations = Column(JSON, nullable=True)
    availability_status = Column(String, nullable=True)
    description = Column(String, nullable=True)
    source_name = Column(String, nullable=True)
    source_record_id = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    source_location = Column(String, nullable=True)
    source_update_date = Column(DateTime, nullable=True)
    import_timestamp = Column(DateTime, default=datetime.utcnow, nullable=True)
    raw_source_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationship to many‑to‑many matches
    lead_matches = relationship(
        "LeadPropertyMatch",
        back_populates="property",
        cascade="all, delete-orphan",
    )
