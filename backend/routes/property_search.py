# -*- coding: utf-8 -*-
"""FastAPI router for property search endpoints.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..services.property_search import PropertySearchService
from ..schemas.property_search import (
    PropertySearchResponse,
    PropertyDetailResponse,
    PropertySearchItem,
)

router = APIRouter()

# ---------------------------------------------------------------------
# Dependency that provides a PropertySearchService instance.
# Tests can override this dependency to inject a temporary SQLite factory.
# ---------------------------------------------------------------------
def get_property_search_service() -> PropertySearchService:
    """Return a fresh ``PropertySearchService`` using the global ``SessionLocal``.
    """
    return PropertySearchService(session_factory=SessionLocal)

# Helper to convert ORM model to response item.
def _orm_to_item(prop) -> PropertySearchItem:
    # Derive provenance fields
    price_note = "demo data" if prop.source_name == "synthetic" else "historical"
    is_synthetic = prop.source_name == "synthetic"
    # Build the schema instance using the ORM fields plus derived ones
    data = {
        "property_id": prop.property_id,
        "project_name": prop.project_name,
        "location": prop.location,
        "property_type": prop.property_type,
        "configuration": prop.configuration,
        "price_min": prop.price_min,
        "price_max": prop.price_max,
        "listing_price": prop.listing_price,
        "possession_date": prop.possession_date,
        "amenities": prop.amenities,
        "nearby_locations": prop.nearby_locations,
        "availability_status": prop.availability_status,
        "description": prop.description,
        "source_name": prop.source_name,
        "is_synthetic": is_synthetic,
        "price_note": price_note,
        "created_at": prop.created_at,
        "updated_at": prop.updated_at,
    }
    return PropertySearchItem(**data)

# ---------------------------------------------------------------------
# /properties/search
# ---------------------------------------------------------------------
@router.get("/search", response_model=PropertySearchResponse)
def search_properties(
    location: Optional[str] = Query(None, description="Case‑insensitive substring on location."),
    property_type: Optional[str] = Query(None, description="Exact property type filter."),
    configuration: Optional[str] = Query(None, description="BHK configuration filter (e.g. '2 BHK')."),
    min_price: Optional[Decimal] = Query(None, ge=0, description="Minimum budget (inclusive)."),
    max_price: Optional[Decimal] = Query(None, ge=0, description="Maximum budget (inclusive)."),
    possession_before: Optional[date] = Query(
        None, description="Only properties with possession_date on or before this date."
    ),
    availability: Optional[str] = Query(None, description="Exact availability_status filter."),
    sort: str = Query("price_asc", regex="^(price_asc|price_desc)$"),
    limit: int = Query(20, ge=0, le=100),
    offset: int = Query(0, ge=0),
    service: PropertySearchService = Depends(get_property_search_service),
):
    try:
        total, results = service.search(
            location=location,
            property_type=property_type,
            configuration=configuration,
            min_price=min_price,
            max_price=max_price,
            possession_before=possession_before,
            availability=availability,
            sort=sort,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    items = [_orm_to_item(p) for p in results]
    return PropertySearchResponse(total=total, limit=limit, offset=offset, items=items)

# ---------------------------------------------------------------------
# /properties/{property_id}
# ---------------------------------------------------------------------
@router.get("/{property_id}", response_model=PropertyDetailResponse)
def get_property(
    property_id: int,
    service: PropertySearchService = Depends(get_property_search_service),
):
    prop = service.get_by_id(property_id)
    if not prop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Property not found")
    return _orm_to_item(prop)
