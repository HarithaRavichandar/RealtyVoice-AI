# -*- coding: utf-8 -*-
"""
Tool Executor – Phase 5C Increment 3D + Phase 2 (Lead & Consent)

Implements execution for:
* search_properties
* get_property_details
* update_customer_requirements
* create_callback_request
* register_lead
* record_consent
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..tools.schemas import (
    SearchPropertiesRequest,
    GetPropertyDetailsRequest,
    UpdateCustomerRequirementsRequest,
    CreateCallbackRequest,
    RegisterLeadRequest,
    RecordConsentRequest,
    NearbyAmenitiesRequest,
)

logger = logging.getLogger(__name__)

from .conversation import update_temp_requirements

from ...models import (
    Conversation,
    Lead,
    ConsentRecord,
    CallbackRequest,
)

from backend.services.property_search import PropertySearchService
from backend.services.location_normalizer import suggest_locations
from backend.schemas.property_search import (
    PropertySearchItem,
    PropertyDetailResponse,
)


# Search argument -> stored CustomerRequirements field.
_SAVED_REQUIREMENT_FIELDS = {
    "min_budget": "budget_min",
    "max_budget": "budget_max",
    "location": "location",
    "property_type": "property_type",
    "configuration": "configuration",
    "possession_before": "possession_before",
    "availability": "availability",
}


def _apply_saved_requirements(
    session: Session,
    conversation_id: Optional[int],
    request: SearchPropertiesRequest,
) -> tuple[SearchPropertiesRequest, Dict[str, Any]]:
    """Fill search criteria the caller left out from saved requirements.

    Saved requirements are only fallbacks: any field the caller supplied
    explicitly (even as null) is kept as-is, so a new preference such as
    3 BHK always overrides a saved 2 BHK. Each saved value is validated
    together with the request; a value that is unusable (invalid, or in
    conflict with an explicit budget) is skipped instead of failing the
    search. Returns the final request and the ``applied_defaults`` mapping
    of search argument -> value that came from saved requirements.
    """
    if conversation_id is None:
        return request, {}

    conversation = session.get(Conversation, conversation_id)
    saved = conversation.temp_requirements if conversation else None
    if not isinstance(saved, dict) or not saved:
        return request, {}

    merged = request.model_dump(exclude_unset=True)
    applied: Dict[str, Any] = {}

    for search_field, saved_field in _SAVED_REQUIREMENT_FIELDS.items():
        if search_field in request.model_fields_set:
            continue
        value = saved.get(saved_field)
        if value is None or value == "":
            continue
        try:
            SearchPropertiesRequest.model_validate(
                {**merged, search_field: value}
            )
        except (ValidationError, ArithmeticError):
            # ArithmeticError covers decimal.InvalidOperation raised by the
            # budget converter for non-numeric saved values.
            logger.warning(
                "Ignoring unusable saved requirement %s for conversation %s",
                saved_field,
                conversation_id,
            )
            continue
        merged[search_field] = value
        applied[search_field] = value

    if not applied:
        return request, {}

    final = SearchPropertiesRequest.model_validate(merged)
    return final, final.model_dump(mode="json", include=set(applied))


def _location_suggestions(
    service: PropertySearchService, location: str
) -> list[dict[str, str]]:
    """Suggest alternatives when a location search found nothing.

    Built only from locations that exist in the database: similarly spelt
    names (confirm with the customer) and city-level labels that contain the
    area named (a parent city, which does not prove locality-level inventory).
    """
    return suggest_locations(location, service.distinct_locations())


def execute_tool(
    session: Session,
    tool_name: str,
    payload: Dict[str, Any],
    conversation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a tool using the caller-provided SQLAlchemy session."""

    # ---------------------------------------------------------------
    # Resolve the appropriate request schema
    # ---------------------------------------------------------------
    if tool_name == "search_properties":
        Schema = SearchPropertiesRequest
    elif tool_name == "get_property_details":
        Schema = GetPropertyDetailsRequest
    elif tool_name == "update_customer_requirements":
        Schema = UpdateCustomerRequirementsRequest
    elif tool_name == "create_callback_request":
        Schema = CreateCallbackRequest
    elif tool_name == "register_lead":
        Schema = RegisterLeadRequest
    elif tool_name == "record_consent":
        Schema = RecordConsentRequest
    elif tool_name == "nearby_amenities":
        Schema = NearbyAmenitiesRequest
    else:
        return {"error": "unknown_tool", "tool": tool_name}

    # ---------------------------------------------------------------
    # Validate the payload
    # ---------------------------------------------------------------
    try:
        request = Schema.model_validate(payload)
    except ValidationError as exc:
        return {
            "error": "validation",
            "tool": tool_name,
            "details": exc.errors(include_context=False),
        }

    # ---------------------------------------------------------------
    # Update customer requirements
    # ---------------------------------------------------------------
    if tool_name == "update_customer_requirements":
        if conversation_id is None:
            return {"error": "missing_conversation_id", "tool": tool_name}
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        updated_requirements = dict(conversation.temp_requirements or {})
        supplied_fields = request.model_dump(exclude_unset=True)
        updated_requirements.update(supplied_fields)
        try:
            update_temp_requirements(
                session=session,
                conversation_id=conversation_id,
                requirements=updated_requirements,
            )
        except ValueError:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        return {"tool": tool_name, "requirements": conversation.temp_requirements}

    # ---------------------------------------------------------------
    # Create callback request
    # ---------------------------------------------------------------
    if tool_name == "create_callback_request":
        if conversation_id is None:
            return {"error": "missing_conversation_id", "tool": tool_name}
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        if conversation.lead_id is None:
            return {
                "error": "lead_required",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        lead = session.get(Lead, conversation.lead_id)
        if lead is None:
            return {
                "error": "lead_not_found",
                "tool": tool_name,
                "lead_id": conversation.lead_id,
            }
        latest_consent = (
            session.query(ConsentRecord)
            .filter(
                ConsentRecord.conversation_id == conversation_id,
                ConsentRecord.lead_id == lead.lead_id,
                ConsentRecord.consent_type == "callback_contact",
            )
            .order_by(
                ConsentRecord.recorded_at.desc(),
                ConsentRecord.consent_id.desc(),
            )
            .first()
        )
        if latest_consent is None or not latest_consent.granted:
            return {
                "error": "callback_consent_required",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        callback = CallbackRequest(
            lead_id=lead.lead_id,
            conversation_id=conversation_id,
            preferred_contact=request.preferred_contact,
            preferred_date=request.preferred_date.isoformat(),
            preferred_time=request.preferred_time,
            reason=request.reason,
        )
        session.add(callback)
        session.flush()
        return {
            "tool": tool_name,
            "callback": {
                "callback_id": callback.callback_id,
                "lead_id": callback.lead_id,
                "conversation_id": callback.conversation_id,
                "preferred_contact": callback.preferred_contact,
                "preferred_date": callback.preferred_date,
                "preferred_time": callback.preferred_time,
                "reason": callback.reason,
                "status": callback.status,
            },
        }

    # ---------------------------------------------------------------
    # Register lead
    # ---------------------------------------------------------------
    if tool_name == "register_lead":
        if conversation_id is None:
            return {"error": "missing_conversation_id", "tool": tool_name}
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        if conversation.lead_id is not None:
            lead = session.get(Lead, conversation.lead_id)
            if lead is not None:
                if request.customer_name is not None:
                    lead.customer_name = request.customer_name
                if request.phone is not None:
                    lead.phone = request.phone
                if request.email is not None:
                    lead.email = request.email
                session.flush()
                return {
                    "tool": tool_name,
                    "lead": {
                        "lead_id": lead.lead_id,
                        "customer_name": lead.customer_name,
                        "phone": lead.phone,
                        "email": lead.email,
                        "pipeline_status": lead.pipeline_status,
                    },
                    "updated": True,
                }
        temp_req = conversation.temp_requirements or {}
        lead = Lead(
            customer_name=request.customer_name,
            phone=request.phone,
            email=request.email,
            budget_min=temp_req.get("budget_min"),
            budget_max=temp_req.get("budget_max"),
            preferred_location=temp_req.get("location"),
            property_type=temp_req.get("property_type"),
            configuration=temp_req.get("configuration"),
            possession_preference=temp_req.get("possession_before"),
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        session.add(lead)
        session.flush()
        conversation.lead_id = lead.lead_id
        session.flush()
        return {
            "tool": tool_name,
            "lead": {
                "lead_id": lead.lead_id,
                "customer_name": lead.customer_name,
                "phone": lead.phone,
                "email": lead.email,
                "pipeline_status": lead.pipeline_status,
            },
            "updated": False,
        }

    # ---------------------------------------------------------------
    # Record consent
    # ---------------------------------------------------------------
    if tool_name == "record_consent":
        if conversation_id is None:
            return {"error": "missing_conversation_id", "tool": tool_name}
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }
        if conversation.lead_id is None:
            return {
                "error": "lead_required",
                "tool": tool_name,
                "message": "Register the customer as a lead before recording consent.",
                "conversation_id": conversation_id,
            }
        lead = session.get(Lead, conversation.lead_id)
        if lead is None:
            return {"error": "lead_not_found", "tool": tool_name, "lead_id": conversation.lead_id}
        consent = ConsentRecord(
            lead_id=lead.lead_id,
            conversation_id=conversation_id,
            consent_type=request.consent_type,
            granted=request.granted,
            source="web",
            policy_version="1.0",
        )
        session.add(consent)
        session.flush()
        return {
            "tool": tool_name,
            "consent": {
                "consent_id": consent.consent_id,
                "lead_id": consent.lead_id,
                "consent_type": consent.consent_type,
                "granted": consent.granted,
            },
        }

    # ---------------------------------------------------------------
    # Nearby amenities (async – returns a deferred marker)
    # ---------------------------------------------------------------
    if tool_name == "nearby_amenities":
        return {
            "tool": tool_name,
            "_async_pending": True,
            "location": request.location,
            "categories": request.categories,
            "radius": request.radius,
        }

    # ---------------------------------------------------------------
    # Initialize property search service
    # ---------------------------------------------------------------
    service = PropertySearchService(
        session_factory=lambda: session,
        close_session=False,
    )

    # ---------------------------------------------------------------
    # Search properties
    # ---------------------------------------------------------------
    if tool_name == "search_properties":
        request, applied_defaults = _apply_saved_requirements(
            session, conversation_id, request
        )
        total, results = service.search(
            location=request.location,
            property_type=request.property_type,
            configuration=request.configuration,
            min_price=request.min_budget,
            max_price=request.max_budget,
            possession_before=request.possession_before,
            availability=request.availability,
        )
        items = [PropertySearchItem.model_validate(prop) for prop in results]
        response: Dict[str, Any] = {
            "tool": tool_name,
            "total": total,
            "items": [item.model_dump(mode="json") for item in items],
            "applied_defaults": applied_defaults,
        }
        if request.availability:
            response["availability_note"] = (
                "Most records in the dataset have no availability status; "
                "properties with an unknown status are excluded when an "
                "availability filter is used, so these results may be "
                "only a small part of what exists."
            )
        if total == 0 and request.location:
            suggestions = _location_suggestions(service, request.location)
            if suggestions:
                response["location_suggestions"] = suggestions
        return response

    # ---------------------------------------------------------------
    # Get property details
    # ---------------------------------------------------------------
    prop = service.get_by_id(request.property_id)
    if prop is None:
        return {"error": "not_found", "tool": tool_name, "property_id": request.property_id}
    detail = PropertyDetailResponse.model_validate(prop)
    return {"tool": tool_name, "property": detail.model_dump(mode="json")}


# ---------------------------------------------------------------
# Async executor for tools that need async I/O (Mappls nearby)
# ---------------------------------------------------------------

async def execute_nearby_amenities(
    location: str,
    categories: list[str] | None = None,
    radius: int = 3000,
) -> Dict[str, Any]:
    """Run the nearby_amenities tool asynchronously via Mappls API.

    Returns a tool-result dict ready for the LLM context.
    """
    from backend.config import MAPPLS_API_KEY, MAPPLS_BASE_URL
    from ..providers.mappls_provider import (
        MappslsProvider,
        MappslsError,
        clamp_radius,
    )

    if not MAPPLS_API_KEY:
        return {
            "tool": "nearby_amenities",
            "error": "mappls_not_configured",
            "message": "Mappls API key is not configured.",
        }

    try:
        async with MappslsProvider(
            api_key=MAPPLS_API_KEY,
            base_url=MAPPLS_BASE_URL,
        ) as provider:
            # Step 1: Geocode the location.
            geo = await provider.geocode(location)

            if geo is None:
                return {
                    "tool": "nearby_amenities",
                    "error": "geocode_failed",
                    "message": f"Could not find coordinates for '{location}'.",
                }

            # If geocode succeeded but no coordinates are present, report
            # a structured error so the LLM can inform the caller.
            if not geo.has_coordinates:
                return {
                    "tool": "nearby_amenities",
                    "error": "coordinates_unavailable",
                    "message": (
                        f"Geocode succeeded but no coordinates were "
                        f"returned for '{location}'."
                    ),
                    "formatted_address": geo.formatted_address,
                    "eloc": geo.eloc,
                }

            # Step 2: Search nearby amenities (only when we have valid coordinates).
            results = await provider.nearby_amenities(
                latitude=geo.latitude,
                longitude=geo.longitude,
                categories=categories,
                radius=radius,
            )

            # Step 3: Format for the LLM. A category whose lookup failed is
            # reported as failed, never as an empty list, so "lookup failed"
            # cannot be mistaken for "none found nearby".
            amenities: Dict[str, Any] = {}
            failed_categories: list[str] = []
            for cat_name, nearby_result in results.items():
                if nearby_result.error:
                    logger.warning(
                        "Nearby lookup for '%s' failed: %s",
                        cat_name,
                        nearby_result.error,
                    )
                    failed_categories.append(cat_name)
                    continue
                amenities[cat_name] = [
                    {
                        "name": place.name,
                        "category": place.category,
                        "distance_m": place.distance,
                        "latitude": place.latitude,
                        "longitude": place.longitude,
                        "address": place.address,
                        "eloc": place.eloc,
                        "phone": place.phone,
                    }
                    for place in nearby_result.places
                ]

            if failed_categories and not amenities:
                return {
                    "tool": "nearby_amenities",
                    "error": "mappls_api_error",
                    "message": (
                        "The amenity lookup failed for every requested "
                        "category."
                    ),
                    "failed_categories": failed_categories,
                }

            effective_radius = clamp_radius(radius)
            response: Dict[str, Any] = {
                "tool": "nearby_amenities",
                "location": location,
                "coordinates": {
                    "latitude": geo.latitude,
                    "longitude": geo.longitude,
                },
                "address": geo.formatted_address,
                "radius_m": effective_radius,
                "amenities": amenities,
            }
            if failed_categories:
                response["failed_categories"] = failed_categories
            if effective_radius != radius:
                response["radius_note"] = (
                    f"Requested radius {radius} m was adjusted to "
                    f"{effective_radius} m (supported range is 500-10000 m)."
                )
            return response

    except MappslsError as exc:
        logger.warning("Mappls API error for '%s': %s", location, exc)
        return {
            "tool": "nearby_amenities",
            "error": "mappls_api_error",
            "message": str(exc),
        }