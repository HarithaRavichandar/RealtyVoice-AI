# -*- coding: utf-8 -*-
"""Pydantic schemas for the property search API.
"""

from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, field_serializer, model_validator


class PropertySearchItem(BaseModel):
    """Individual property representation returned by the search endpoint."""

    property_id: int
    project_name: Optional[str] = None
    location: str
    property_type: str
    configuration: Optional[str] = None
    price_min: Optional[Decimal] = None
    price_max: Optional[Decimal] = None
    listing_price: Optional[Decimal] = None
    possession_date: Optional[datetime] = None
    amenities: Optional[List[str]] = None
    nearby_locations: Optional[List[str]] = None
    availability_status: Optional[str] = None
    description: Optional[str] = None
    source_name: Optional[str] = None
    is_synthetic: bool = Field(False, alias="is_synthetic")
    price_note: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    @field_serializer("price_min", "price_max", "listing_price")
    @classmethod
    def _serialize_decimal(cls, v: Optional[Decimal], _info) -> Optional[int | float]:
        if v is None:
            return None
        return int(v) if v == int(v) else float(v)
    
    @field_validator('price_min', 'price_max', 'listing_price', mode='before')
    @classmethod
    def _ensure_decimal(cls, v):
        if v is None:
            return None
        return Decimal(v)



class PropertySearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total: int
    limit: int
    offset: int
    items: List[PropertySearchItem]


class PropertyDetailResponse(PropertySearchItem):
    """Detail view – currently identical to the item schema, but kept for clarity."""
    pass

