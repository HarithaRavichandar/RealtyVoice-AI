
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

The executor validates incoming payloads against request schemas and uses
the caller-supplied SQLAlchemy session.

It does not commit, rollback, or close the session.
The caller remains responsible for transaction management.
"""

from __future__ import annotations

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
)

from .conversation import update_temp_requirements

from ...models import (
    Conversation,
    Lead,
    ConsentRecord,
    CallbackRequest,
)

from backend.services.property_search import PropertySearchService
from backend.schemas.property_search import (
    PropertySearchItem,
    PropertyDetailResponse,
)


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

    else:
        return {
            "error": "unknown_tool",
            "tool": tool_name,
        }

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
            return {
                "error": "missing_conversation_id",
                "tool": tool_name,
            }

        conversation = session.get(Conversation, conversation_id)

        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }

        # Preserve existing requirements.
        updated_requirements = dict(
            conversation.temp_requirements or {}
        )

        # Update only explicitly supplied fields.
        # Explicitly supplied None values are also retained.
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

        return {
            "tool": tool_name,
            "requirements": conversation.temp_requirements,
        }

    # ---------------------------------------------------------------
    # Create callback request
    # ---------------------------------------------------------------
    if tool_name == "create_callback_request":

        # Require a conversation ID.
        if conversation_id is None:
            return {
                "error": "missing_conversation_id",
                "tool": tool_name,
            }

        # Retrieve conversation.
        conversation = session.get(Conversation, conversation_id)

        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }

        # A callback requires an associated lead.
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

        # Retrieve the latest callback consent.
        # Do not filter by granted=True before selecting the latest record.
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

        # Reject if consent is missing or the latest consent was revoked.
        if latest_consent is None or not latest_consent.granted:
            return {
                "error": "callback_consent_required",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }

        # Create callback record.
        callback = CallbackRequest(
            lead_id=lead.lead_id,
            conversation_id=conversation_id,
            preferred_contact=request.preferred_contact,
            preferred_date=request.preferred_date.isoformat(),
            preferred_time=request.preferred_time,
            reason=request.reason,
        )

        session.add(callback)

        # Flush to obtain callback_id without committing.
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
            return {
                "error": "missing_conversation_id",
                "tool": tool_name,
            }

        conversation = session.get(Conversation, conversation_id)

        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }

        # If the conversation already has a lead, update it rather
        # than creating a duplicate.
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

        # Create a new lead and link it to the conversation.
        # Copy any requirements already collected during the conversation.
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

        # Link the conversation to the new lead.
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
            return {
                "error": "missing_conversation_id",
                "tool": tool_name,
            }

        conversation = session.get(Conversation, conversation_id)

        if conversation is None:
            return {
                "error": "conversation_not_found",
                "tool": tool_name,
                "conversation_id": conversation_id,
            }

        # A consent record requires an associated lead.
        if conversation.lead_id is None:
            return {
                "error": "lead_required",
                "tool": tool_name,
                "message": "Register the customer as a lead before recording consent.",
                "conversation_id": conversation_id,
            }

        lead = session.get(Lead, conversation.lead_id)

        if lead is None:
            return {
                "error": "lead_not_found",
                "tool": tool_name,
                "lead_id": conversation.lead_id,
            }

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
    # Initialize property search service
    # Keep the caller's session alive.
    # ---------------------------------------------------------------
    service = PropertySearchService(
        session_factory=lambda: session,
        close_session=False,
    )

    # ---------------------------------------------------------------
    # Search properties
    # ---------------------------------------------------------------
    if tool_name == "search_properties":

        total, results = service.search(
            location=request.location,
            property_type=request.property_type,
            configuration=request.configuration,
            min_price=None,
            max_price=request.max_budget,
            possession_before=request.possession_before,
            availability=None,
        )

        items = [
            PropertySearchItem.model_validate(prop)
            for prop in results
        ]

        return {
            "tool": tool_name,
            "total": total,
            "items": [
                item.model_dump(mode="json")
                for item in items
            ],
        }

    # ---------------------------------------------------------------
    # Get property details
    # ---------------------------------------------------------------
    prop = service.get_by_id(request.property_id)

    if prop is None:
        return {
            "error": "not_found",
            "tool": tool_name,
            "property_id": request.property_id,
        }

    detail = PropertyDetailResponse.model_validate(prop)

    return {
        "tool": tool_name,
        "property": detail.model_dump(mode="json"),
    }
