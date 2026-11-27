# -*- coding: utf-8 -*-
"""Property-to-lead matching and ranking (task 3.4)."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from backend.models import Conversation, Lead
from backend.models.lead_property_match import LeadPropertyMatch
from backend.models.property import Property
from backend.orchestrator.services.tool_executor import execute_tool
from backend.services.lead_matching import MIN_SCORE, evaluate, match_properties, save_matches


def prop(**kw):
    base = dict(location="bangalore", property_type="apartment", configuration="2 BHK",
                price_min=5_000_000, price_max=6_000_000, possession_date=datetime(2025, 6, 1),
                availability_status="Ready to Move")
    base.update(kw)
    return Property(**base)


def statuses(result):
    return {c.name: c.status for c in result.criteria}


# ---------------------------------------------------------------------------
# Scoring one property
# ---------------------------------------------------------------------------

def test_perfect_match_scores_100_with_every_criterion_matched():
    req = {"configuration": "2 BHK", "budget_min": Decimal("4000000"), "budget_max": Decimal("7000000"),
           "property_type": "Apartment", "possession_before": date(2026, 1, 1), "availability": "ready to move"}
    r = evaluate(prop(), req)
    assert r.score == 100
    assert set(statuses(r).values()) == {"match"}


def test_only_stated_requirements_count_towards_the_score():
    r = evaluate(prop(), {"configuration": "2 BHK"})
    assert r.score == 100 and list(statuses(r)) == ["configuration"]
    assert evaluate(prop(), {"location": "Bengaluru"}).score == 100  # location handled by the gate


def test_configuration_mismatch_lowers_the_score_by_its_weight():
    # configuration 25 of (25 + budget 35) -> budget matches, config does not
    r = evaluate(prop(configuration="3 BHK"), {"configuration": "2 BHK", "budget_max": Decimal("7000000")})
    assert statuses(r) == {"configuration": "mismatch", "budget": "match"}
    assert r.score == round(100 * 35 / 60)


@pytest.mark.parametrize("have", ["2 BHK", "2-bhk", "2  bhk", "2 bhk"])
def test_configuration_matching_ignores_case_dashes_and_spacing(have):
    assert statuses(evaluate(prop(configuration=have), {"configuration": "2 BHK"}))["configuration"] == "match"


@pytest.mark.parametrize(
    "low, high, bmin, bmax, status, credit_score",
    [
        (5_000_000, 6_000_000, 4_000_000, 7_000_000, "match", 100),
        (5_000_000, 9_000_000, None, 7_000_000, "partial", 60),    # range reaches past the maximum
        (3_000_000, 6_000_000, 5_000_000, None, "partial", 60),    # range starts below the minimum
        (7_300_000, 8_000_000, None, 7_000_000, "partial", 25),    # within 10% over: labelled near miss
        (9_000_000, 9_500_000, None, 7_000_000, "mismatch", 0),
        (1_000_000, 2_000_000, 5_000_000, None, "mismatch", 0),
    ],
)
def test_budget_grading(low, high, bmin, bmax, status, credit_score):
    req = {"budget_min": Decimal(bmin) if bmin else None, "budget_max": Decimal(bmax) if bmax else None}
    r = evaluate(prop(price_min=low, price_max=high), req)
    assert statuses(r)["budget"] == status
    assert r.score == credit_score


def test_listing_price_alone_is_used_when_there_is_no_range():
    r = evaluate(prop(price_min=None, price_max=None, listing_price=5_500_000), {"budget_max": Decimal("6000000")})
    assert statuses(r)["budget"] == "match"


def test_missing_data_is_unknown_not_a_match():
    r = evaluate(
        prop(price_min=None, price_max=None, listing_price=None, configuration=None,
             possession_date=None, availability_status=None, property_type=None),
        {"budget_max": Decimal("6000000"), "configuration": "2 BHK", "possession_before": date(2026, 1, 1),
         "availability": "Ready to Move", "property_type": "apartment"},
    )
    assert set(statuses(r).values()) == {"unknown"}
    assert r.score == 30


def test_possession_on_the_cutoff_date_matches_and_after_it_does_not():
    on = evaluate(prop(possession_date=datetime(2026, 1, 1)), {"possession_before": date(2026, 1, 1)})
    after = evaluate(prop(possession_date=datetime(2026, 1, 2)), {"possession_before": date(2026, 1, 1)})
    assert statuses(on)["possession"] == "match"
    assert statuses(after)["possession"] == "mismatch"


# ---------------------------------------------------------------------------
# Ranking over the database
# ---------------------------------------------------------------------------

@pytest.fixture
def stock(dbsession):
    rows = {
        "best": prop(),
        "pricey": prop(price_min=7_200_000, price_max=7_800_000),            # near miss on budget
        "wrong_size": prop(configuration="3 BHK"),
        "wrong_everything": prop(configuration="4 BHK", property_type="villa", price_min=30_000_000, price_max=35_000_000),
        "other_city": prop(location="mumbai"),
        "unknown_price": prop(price_min=None, price_max=None),
    }
    dbsession.add_all(rows.values())
    dbsession.flush()
    return {k: v.property_id for k, v in rows.items()}


REQ = {"location": "Bengaluru", "configuration": "2 BHK", "budget_max": Decimal("7000000"), "property_type": "apartment"}


def test_results_are_ranked_best_first_and_never_leave_the_location(dbsession, stock):
    results = match_properties(dbsession, REQ, limit=10)
    ids = [r.prop.property_id for r in results]
    assert ids[0] == stock["best"]
    assert stock["other_city"] not in ids
    assert stock["wrong_everything"] not in ids  # below the minimum score
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(r.score >= MIN_SCORE for r in results)


def test_near_miss_is_included_below_the_exact_match_and_labelled(dbsession, stock):
    results = match_properties(dbsession, REQ, limit=10)
    by_id = {r.prop.property_id: r for r in results}
    assert by_id[stock["best"]].score > by_id[stock["pricey"]].score
    assert statuses(by_id[stock["pricey"]])["budget"] == "partial"


def test_unknown_price_ranks_below_a_known_match(dbsession, stock):
    ids = [r.prop.property_id for r in match_properties(dbsession, REQ, limit=10)]
    assert ids.index(stock["best"]) < ids.index(stock["unknown_price"])


def test_limit_is_respected(dbsession, stock):
    assert len(match_properties(dbsession, REQ, limit=1)) == 1


def test_unknown_location_returns_nothing(dbsession, stock):
    assert match_properties(dbsession, {**REQ, "location": "Atlantis"}) == []


def test_ties_are_broken_by_property_id(dbsession):
    a, b = prop(), prop()
    dbsession.add_all([a, b])
    dbsession.flush()
    ids = [r.prop.property_id for r in match_properties(dbsession, {"configuration": "2 BHK"}, limit=5)]
    assert ids == sorted(ids)


def test_save_matches_upserts_scores(dbsession, stock):
    lead = Lead(qualification_status="Unqualified", pipeline_status="New")
    dbsession.add(lead)
    dbsession.flush()
    results = match_properties(dbsession, REQ, limit=3)

    save_matches(dbsession, lead.lead_id, results)
    save_matches(dbsession, lead.lead_id, results)  # idempotent

    rows = dbsession.query(LeadPropertyMatch).filter_by(lead_id=lead.lead_id).all()
    assert len(rows) == len(results)
    assert {r.property_id: r.match_score for r in rows} == {r.prop.property_id: float(r.score) for r in results}


# ---------------------------------------------------------------------------
# recommend_properties tool
# ---------------------------------------------------------------------------

@pytest.fixture
def convo(dbsession, stock):
    c = Conversation(channel="browser_text", external_session_id="m1", temp_requirements=None)
    dbsession.add(c)
    dbsession.flush()
    return c


def recommend(session, convo, **payload):
    return execute_tool(session, "recommend_properties", payload, conversation_id=convo.conversation_id)


def test_tool_needs_saved_requirements(dbsession, convo):
    assert recommend(dbsession, convo)["error"] == "no_requirements"


def test_tool_needs_a_conversation(dbsession):
    assert execute_tool(dbsession, "recommend_properties", {})["error"] == "missing_conversation_id"


@pytest.mark.parametrize("limit", [0, 11, -1])
def test_tool_rejects_bad_limits(dbsession, convo, limit):
    assert recommend(dbsession, convo, limit=limit)["error"] == "validation"


def test_tool_rejects_extra_arguments(dbsession, convo):
    assert recommend(dbsession, convo, location="Pune")["error"] == "validation"


def test_tool_returns_scored_items_with_criteria_and_the_requirements_used(dbsession, convo, stock):
    execute_tool(dbsession, "update_customer_requirements",
                 {"location": "Bengaluru", "configuration": "2 BHK", "budget_max": "7000000"},
                 conversation_id=convo.conversation_id)

    r = recommend(dbsession, convo, limit=3)

    assert r["total"] == len(r["items"]) <= 3
    top = r["items"][0]
    assert top["property_id"] == stock["best"]
    assert top["match_score"] == 100
    assert {c["criterion"] for c in top["criteria"]} == {"configuration", "budget"}
    assert r["requirements_used"]["location"] == "Bengaluru"
    assert r["requirements_used"]["budget_max"] == "7000000"


def test_tool_saves_matches_for_a_registered_lead(dbsession, convo, stock):
    execute_tool(dbsession, "update_customer_requirements", {"location": "Bengaluru", "configuration": "2 BHK"},
                 conversation_id=convo.conversation_id)
    execute_tool(dbsession, "register_lead", {"phone": "9876543210"}, conversation_id=convo.conversation_id)

    r = recommend(dbsession, convo)

    saved = dbsession.query(LeadPropertyMatch).filter_by(lead_id=convo.lead_id).count()
    assert saved == len(r["items"]) > 0


def test_tool_does_not_store_matches_without_a_lead(dbsession, convo, stock):
    execute_tool(dbsession, "update_customer_requirements", {"location": "Bengaluru"},
                 conversation_id=convo.conversation_id)
    recommend(dbsession, convo)
    assert dbsession.query(LeadPropertyMatch).count() == 0


def test_location_only_requirement_returns_everything_in_the_area(dbsession, convo, stock):
    execute_tool(dbsession, "update_customer_requirements", {"location": "Mumbai"},
                 conversation_id=convo.conversation_id)
    r = recommend(dbsession, convo)
    assert [i["property_id"] for i in r["items"]] == [stock["other_city"]]
    assert r["items"][0]["match_score"] == 100 and r["items"][0]["criteria"] == []


def test_matches_are_linked_when_the_lead_registers_after_the_recommendation(dbsession, convo, stock):
    execute_tool(dbsession, "update_customer_requirements", {"location": "Bengaluru", "configuration": "2 BHK"},
                 conversation_id=convo.conversation_id)
    assert recommend(dbsession, convo)["total"] > 0
    assert dbsession.query(LeadPropertyMatch).count() == 0

    execute_tool(dbsession, "register_lead", {"phone": "9876543210"}, conversation_id=convo.conversation_id)

    assert dbsession.query(LeadPropertyMatch).filter_by(lead_id=convo.lead_id).count() > 0


def test_registering_with_no_requirements_stores_no_matches(dbsession, convo):
    execute_tool(dbsession, "register_lead", {"phone": "9876543210"}, conversation_id=convo.conversation_id)
    assert dbsession.query(LeadPropertyMatch).count() == 0
