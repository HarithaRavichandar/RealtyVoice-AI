'''Tests for the tool request schemas and static registry.

The tests exercise validation rules (extra fields, negative numbers, whitespace
only strings, budget ordering) and the simple static lookup helper.
''' 

import pytest
from datetime import date
from backend.orchestrator.tools.schemas import (
    SearchPropertiesRequest,
    GetPropertyDetailsRequest,
    UpdateCustomerRequirementsRequest,
    CreateCallbackRequest,
)
from backend.orchestrator.services.tool_registry import (
    TOOL_REGISTRY,
    lookup_tool,
    ToolInfo,
)

# ---------------------------------------------------------------------------
# SearchPropertiesRequest tests
# ---------------------------------------------------------------------------

def test_search_properties_valid():
    req = SearchPropertiesRequest(
        location="Mumbai",
        property_type="apartment",
        configuration="2 BHK",
        max_budget="5000000",
        possession_before="2024-12-31",
    )
    assert req.location == "Mumbai"
    assert req.max_budget == 5000000
    assert req.possession_before.isoformat() == "2024-12-31"

def test_search_properties_negative_budget():
    with pytest.raises(ValueError, match="max_budget must be non‑negative"):
        SearchPropertiesRequest(max_budget="-1000")

def test_search_properties_unknown_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SearchPropertiesRequest(unknown="value")

def test_search_properties_whitespace_location():
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        SearchPropertiesRequest(location="   ")

# ---------------------------------------------------------------------------
# GetPropertyDetailsRequest tests
# ---------------------------------------------------------------------------

def test_get_property_details_valid():
    req = GetPropertyDetailsRequest(property_id=42)
    assert req.property_id == 42

@pytest.mark.parametrize("bad_id", [0, -5])
def test_get_property_details_invalid_id(bad_id):
    with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
        GetPropertyDetailsRequest(property_id=bad_id)

def test_get_property_details_unknown_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        GetPropertyDetailsRequest(property_id=1, bogus=123)

# ---------------------------------------------------------------------------
# UpdateCustomerRequirementsRequest tests
# ---------------------------------------------------------------------------

def test_update_customer_requirements_partial():
    req = UpdateCustomerRequirementsRequest(location="Delhi")
    assert req.location == "Delhi"
    # fields not supplied are None
    assert req.budget_min is None
    # model_fields_set contains only the explicitly provided field
    assert req.model_fields_set == {"location"}

def test_update_customer_requirements_budget_range():
    req = UpdateCustomerRequirementsRequest(budget_min="1000", budget_max="2000")
    assert req.budget_min == 1000
    assert req.budget_max == 2000

@pytest.mark.parametrize(
    "min_val, max_val, msg",
    [
        ("-100", None, "budget_min must be non‑negative"),
        (None, "-200", "budget_max must be non‑negative"),
        ("3000", "2000", "budget_min cannot exceed budget_max"),
    ],
)
def test_update_customer_requirements_invalid_budget(min_val, max_val, msg):
    kwargs = {}
    if min_val is not None:
        kwargs["budget_min"] = min_val
    if max_val is not None:
        kwargs["budget_max"] = max_val
    with pytest.raises(ValueError, match=msg):
        UpdateCustomerRequirementsRequest(**kwargs)

def test_update_customer_requirements_whitespace_fields():
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        UpdateCustomerRequirementsRequest(location="   ")
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        UpdateCustomerRequirementsRequest(configuration="\t")

def test_update_customer_requirements_unknown_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        UpdateCustomerRequirementsRequest(foo="bar")

# ---------------------------------------------------------------------------
# CreateCallbackRequest tests
# ---------------------------------------------------------------------------

def test_create_callback_valid():
    req = CreateCallbackRequest(
        preferred_contact="Phone",
        preferred_date=date(2026, 11, 20),
        preferred_time="10:30",
        reason="Please call back.",
    )
    assert req.preferred_contact == "Phone"
    assert req.preferred_date == date(2026, 11, 20)
    assert req.preferred_time == "10:30"
    assert req.reason == "Please call back."

def test_create_callback_missing_fields():
    # Missing preferred_contact
    with pytest.raises(ValueError, match="Field required"):
        CreateCallbackRequest(
            preferred_date=date(2026, 11, 20),
            preferred_time="10:30",
            reason="Please call back.",
        )
    # Missing preferred_date
    with pytest.raises(ValueError, match="Field required"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_time="10:30",
            reason="Please call back.",
        )
    # Missing preferred_time
    with pytest.raises(ValueError, match="Field required"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_date=date(2026, 11, 20),
            reason="Please call back.",
        )
    # Missing reason
    with pytest.raises(ValueError, match="Field required"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_date=date(2026, 11, 20),
            preferred_time="10:30",
        )

@pytest.mark.parametrize("bad_val", ["", "   ", "\n\t"])
def test_create_callback_whitespace_fields(bad_val):
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        CreateCallbackRequest(
            preferred_contact=bad_val,
            preferred_date=date(2026, 11, 20),
            preferred_time="10:30",
            reason="Please call back.",
        )
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_date=date(2026, 11, 20),
            preferred_time=bad_val,
            reason="Please call back.",
        )
    with pytest.raises(ValueError, match="value cannot be whitespace only"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_date=date(2026, 11, 20),
            preferred_time="10:30",
            reason=bad_val,
        )

def test_create_callback_unknown_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CreateCallbackRequest(
            preferred_contact="Phone",
            preferred_date=date(2026, 11, 20),
            preferred_time="10:30",
            reason="ok",
            extra=1,
        )

# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------

def test_tool_registry_contents():
    expected = {
        "search_properties": SearchPropertiesRequest,
        "get_property_details": GetPropertyDetailsRequest,
        "update_customer_requirements": UpdateCustomerRequirementsRequest,
        "create_callback_request": CreateCallbackRequest,
    }
    assert set(TOOL_REGISTRY.keys()) == set(expected.keys())
    for name, schema in expected.items():
        info = TOOL_REGISTRY[name]
        assert isinstance(info, ToolInfo)
        assert info.name == name
        assert info.input_schema is schema
        assert info.implementation is None

def test_lookup_tool_success():
    info = lookup_tool("search_properties")
    assert info.name == "search_properties"
    assert info.input_schema is SearchPropertiesRequest

def test_lookup_tool_failure():
    with pytest.raises(KeyError, match="Tool 'nonexistent' not found in registry"):
        lookup_tool("nonexistent")

