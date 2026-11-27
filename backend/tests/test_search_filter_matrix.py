# -*- coding: utf-8 -*-
"""Table-driven coverage of PropertySearchService filters and combinations.

Seven throwaway in-memory rows with distinct locations, types, sizes, price
ranges, possession dates and availability statuses make every expected result
easy to check by hand. Budget semantics: a property matches when its price
interval overlaps the requested range; a property with no price at all is
excluded whenever a budget bound is given.

    R1 bangalore apartment 2 BHK   5.0M-6.0M   2025-06-01  Ready to Move
    R2 bangalore apartment 3 BHK   8.0M-9.0M   2026-12-31  Under Construction
    R3 bangalore villa     4 BHK   20M-25M     2027-06-30  Under Construction
    R4 mumbai    apartment 2 BHK   9.0M-10.0M  2025-12-31  Ready to Move
    R5 mumbai    apartment 1 BHK   3.0M-3.5M   (none)      (none)
    R6 pune      apartment 2 BHK   (no price)  2026-01-01  (none)
    R7 pune      villa     3 BHK   12M-14M     2026-06-30  (none)
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from backend.models.property import Property
from backend.services.property_search import PropertySearchService

ROWS = {
    "R1": ("bangalore", "apartment", "2 BHK", 5_000_000, 6_000_000, datetime(2025, 6, 1), "Ready to Move"),
    "R2": ("bangalore", "apartment", "3 BHK", 8_000_000, 9_000_000, datetime(2026, 12, 31), "Under Construction"),
    "R3": ("bangalore", "villa", "4 BHK", 20_000_000, 25_000_000, datetime(2027, 6, 30), "Under Construction"),
    "R4": ("mumbai", "apartment", "2 BHK", 9_000_000, 10_000_000, datetime(2025, 12, 31), "Ready to Move"),
    "R5": ("mumbai", "apartment", "1 BHK", 3_000_000, 3_500_000, None, None),
    "R6": ("pune", "apartment", "2 BHK", None, None, datetime(2026, 1, 1), None),
    "R7": ("pune", "villa", "3 BHK", 12_000_000, 14_000_000, datetime(2026, 6, 30), None),
}


@pytest.fixture
def svc(dbsession):
    ids = {}
    for name, (loc, ptype, config, low, high, possession, status) in ROWS.items():
        prop = Property(
            location=loc,
            property_type=ptype,
            configuration=config,
            price_min=low,
            price_max=high,
            listing_price=None if low is None else (low + high) // 2,
            possession_date=possession,
            availability_status=status,
        )
        dbsession.add(prop)
        dbsession.flush()
        ids[prop.property_id] = name
    service = PropertySearchService(session_factory=lambda: dbsession, close_session=False)

    def run(**filters):
        total, results = service.search(limit=100, **filters)
        return total, [ids[p.property_id] for p in results]

    return run


def names(*selected):
    return set(selected)


@pytest.mark.parametrize(
    "filters, expected",
    [
        # --- no filters -----------------------------------------------------
        ({}, {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}),
        # --- location -------------------------------------------------------
        ({"location": "bangalore"}, {"R1", "R2", "R3"}),
        ({"location": "Bengaluru"}, {"R1", "R2", "R3"}),
        ({"location": "mumbai"}, {"R4", "R5"}),
        ({"location": "Pune"}, {"R6", "R7"}),
        # --- property type --------------------------------------------------
        ({"property_type": "apartment"}, {"R1", "R2", "R4", "R5", "R6"}),
        ({"property_type": "Apartment"}, {"R1", "R2", "R4", "R5", "R6"}),
        ({"property_type": "villa"}, {"R3", "R7"}),
        # --- configuration (case and dash variants) --------------------------
        ({"configuration": "2 BHK"}, {"R1", "R4", "R6"}),
        ({"configuration": "2 bhk"}, {"R1", "R4", "R6"}),
        ({"configuration": "2-BHK"}, {"R1", "R4", "R6"}),
        ({"configuration": "3 BHK"}, {"R2", "R7"}),
        # --- budget: single bounds (interval overlap) -------------------------
        ({"max_price": Decimal("6000000")}, {"R1", "R5"}),  # R6 has no price
        ({"max_price": Decimal("2000000")}, set()),
        ({"min_price": Decimal("10000000")}, {"R3", "R4", "R7"}),  # R4 ends exactly at 10M
        ({"min_price": Decimal("30000000")}, set()),
        # --- budget: both bounds ----------------------------------------------
        ({"min_price": Decimal("6000000"), "max_price": Decimal("9000000")}, {"R1", "R2", "R4"}),
        ({"min_price": Decimal("9000000"), "max_price": Decimal("9000000")}, {"R2", "R4"}),
        ({"min_price": Decimal("0"), "max_price": Decimal("100000000")}, {"R1", "R2", "R3", "R4", "R5", "R7"}),
        # --- possession (inclusive; unknown dates excluded) ---------------------
        ({"possession_before": date(2025, 12, 31)}, {"R1", "R4"}),
        ({"possession_before": date(2026, 6, 30)}, {"R1", "R4", "R6", "R7"}),
        ({"possession_before": date(2024, 1, 1)}, set()),
        # --- availability (case-insensitive exact status) -----------------------
        ({"availability": "Ready to Move"}, {"R1", "R4"}),
        ({"availability": "ready to move"}, {"R1", "R4"}),
        ({"availability": "Under Construction"}, {"R2", "R3"}),
        # --- combinations -------------------------------------------------------
        ({"location": "bangalore", "property_type": "apartment", "max_price": Decimal("9000000")}, {"R1", "R2"}),
        ({"location": "bangalore", "property_type": "apartment", "configuration": "3 BHK"}, {"R2"}),
        (
            {"location": "mumbai", "min_price": Decimal("9000000"), "possession_before": date(2025, 12, 31)},
            {"R4"},
        ),
        ({"configuration": "2 BHK", "availability": "Ready to Move", "max_price": Decimal("7000000")}, {"R1"}),
        ({"location": "Pune", "property_type": "villa", "configuration": "3 BHK"}, {"R7"}),
        # --- empty results ------------------------------------------------------
        ({"location": "bangalore", "configuration": "4 BHK", "property_type": "apartment"}, set()),
        ({"location": "pune", "max_price": Decimal("1000000")}, set()),
        ({"location": "mumbai", "availability": "Under Construction"}, set()),
        # --- unknown values -------------------------------------------------------
        ({"location": "Atlantis"}, set()),
        ({"property_type": "castle"}, set()),
        ({"configuration": "9 BHK"}, set()),
        ({"availability": "Sold"}, set()),
    ],
)
def test_filter_matrix(svc, filters, expected):
    total, found = svc(**filters)
    assert set(found) == expected
    assert total == len(expected)
    assert len(found) == len(set(found))  # no duplicates


def test_unpriced_property_is_found_without_a_budget_but_not_with_one(svc):
    assert "R6" in svc(location="pune")[1]
    assert "R6" not in svc(location="pune", max_price=Decimal("100000000"))[1]
    assert "R6" not in svc(location="pune", min_price=Decimal("0"))[1]


def test_sorting_and_pagination(svc):
    assert svc(location="bangalore")[1] == ["R1", "R2", "R3"]  # price ascending

    service_total, _ = svc(location="bangalore")
    assert service_total == 3


def test_sort_descending_and_page_window(dbsession):
    for name, (loc, ptype, config, low, high, possession, status) in ROWS.items():
        dbsession.add(
            Property(
                location=loc, property_type=ptype, configuration=config,
                price_min=low, price_max=high,
                listing_price=None if low is None else (low + high) // 2,
                possession_date=possession, availability_status=status,
            )
        )
    dbsession.flush()
    service = PropertySearchService(session_factory=lambda: dbsession, close_session=False)

    _total, desc = service.search(location="bangalore", sort="price_desc")
    assert [p.configuration for p in desc] == ["4 BHK", "3 BHK", "2 BHK"]

    total, page = service.search(location="bangalore", limit=2, offset=1)
    assert total == 3  # the count ignores pagination
    assert [p.configuration for p in page] == ["3 BHK", "4 BHK"]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"min_price": Decimal("9"), "max_price": Decimal("5")}, "min_price cannot be greater than max_price"),
        ({"min_price": Decimal("-1")}, "min_price cannot be negative"),
        ({"max_price": Decimal("-1")}, "max_price cannot be negative"),
        ({"limit": 101}, "limit cannot exceed 100"),
        ({"offset": -1}, "non"),
    ],
)
def test_invalid_arguments_are_rejected(dbsession, bad, message):
    service = PropertySearchService(session_factory=lambda: dbsession, close_session=False)
    with pytest.raises(ValueError, match=message):
        service.search(**bad)
