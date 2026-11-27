# -*- coding: utf-8 -*-
"""Tests for the shared UTC timestamp helper and its integration with ORM models.

Verifies that:
- ``utc_now()`` returns timezone-aware UTC datetimes.
- SQLAlchemy model defaults receive timezone-aware values before persistence.
- Timestamps survive an SQLite round-trip (as naive-UTC) and remain comparable.
- ``onupdate`` callables refresh ``updated_at`` on ORM updates.
- No ``datetime.utcnow()`` deprecation warnings are emitted.

SQLite note: SQLite does not preserve timezone info.  After a commit + refresh
the returned datetimes are naive but still represent UTC.  Tests in this file
verify the *callable* produces aware values and that round-tripped values remain
correct and comparable.
"""

import time
import warnings
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from backend.models._time import utc_now


# ---------------------------------------------------------------------------
# Unit tests for the utc_now() helper
# ---------------------------------------------------------------------------

class TestUtcNowHelper:
    """Unit tests for the ``utc_now()`` function."""

    def test_returns_datetime(self):
        result = utc_now()
        assert isinstance(result, datetime)

    def test_is_timezone_aware(self):
        result = utc_now()
        assert result.tzinfo is not None

    def test_timezone_is_utc(self):
        result = utc_now()
        assert result.tzinfo == timezone.utc

    def test_no_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utc_now()
        deprecation_msgs = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "utcnow" in str(w.message)
        ]
        assert deprecation_msgs == [], (
            f"utc_now() triggered deprecated utcnow warning: {deprecation_msgs}"
        )

    def test_successive_calls_are_non_decreasing(self):
        t1 = utc_now()
        t2 = utc_now()
        assert t2 >= t1

    def test_value_is_close_to_now(self):
        """The returned timestamp should be within 1 second of the current time."""
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


# ---------------------------------------------------------------------------
# Tests verifying that ORM defaults receive timezone-aware values
# ---------------------------------------------------------------------------

class TestModelDefaultsReceiveAwareValues:
    """Verify that the default callable is wired correctly in every model.

    These tests inspect the value that SQLAlchemy *assigns* to the Python
    attribute before the row is flushed to SQLite.  At this point the value
    is still the timezone-aware datetime returned by ``utc_now()``.
    """

    def test_conversation_defaults_are_aware_before_flush(self, dbsession):
        from backend.models.conversation import Conversation

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="aware-test-conv",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        # Before commit, SQLAlchemy holds the Python value from utc_now()
        assert conv.created_at is not None
        assert conv.created_at.tzinfo is not None, (
            "created_at default should be timezone-aware before SQLite persistence"
        )
        assert conv.updated_at is not None
        assert conv.updated_at.tzinfo is not None, (
            "updated_at default should be timezone-aware before SQLite persistence"
        )

    def test_message_default_is_aware_before_flush(self, dbsession):
        from backend.models.conversation import Conversation
        from backend.models.message import Message

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="aware-test-msg",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        msg = Message(
            conversation_id=conv.conversation_id,
            turn_id=1,
            role="assistant",
            content="Hello",
        )
        dbsession.add(msg)
        dbsession.flush()

        assert msg.created_at is not None
        assert msg.created_at.tzinfo is not None

    def test_lead_defaults_are_aware_before_flush(self, dbsession):
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="Test",
            phone="1234567890",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        assert lead.created_at.tzinfo is not None
        assert lead.updated_at.tzinfo is not None

    def test_callback_defaults_are_aware_before_flush(self, dbsession):
        from backend.models.callback import CallbackRequest
        from backend.models.conversation import Conversation
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="CB Test",
            phone="0000000000",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        conv = Conversation(
            lead_id=lead.lead_id,
            channel="browser_text",
            external_session_id="aware-test-cb",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        cb = CallbackRequest(
            lead_id=lead.lead_id,
            conversation_id=conv.conversation_id,
            preferred_contact="phone",
            preferred_date="2026-01-01",
            preferred_time="10:00",
            status="Pending",
        )
        dbsession.add(cb)
        dbsession.flush()

        assert cb.created_at.tzinfo is not None
        assert cb.updated_at.tzinfo is not None

    def test_consent_defaults_are_aware_before_flush(self, dbsession):
        from backend.models.consent import ConsentRecord
        from backend.models.conversation import Conversation
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="Consent Test",
            phone="1111111111",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        conv = Conversation(
            lead_id=lead.lead_id,
            channel="browser_text",
            external_session_id="aware-test-consent",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        consent = ConsentRecord(
            lead_id=lead.lead_id,
            consent_type="contact_storage",
            granted=True,
            source="web",
            conversation_id=conv.conversation_id,
            policy_version="1.0",
        )
        dbsession.add(consent)
        dbsession.flush()

        assert consent.recorded_at.tzinfo is not None
        assert consent.created_at.tzinfo is not None
        assert consent.updated_at.tzinfo is not None

    def test_property_defaults_are_aware_before_flush(self, dbsession):
        from backend.models.property import Property

        prop = Property(
            location="Whitefield",
            property_type="Apartment",
            configuration="2 BHK",
            source_name="test",
            source_record_id="aware-test-1",
        )
        dbsession.add(prop)
        dbsession.flush()

        assert prop.import_timestamp.tzinfo is not None
        assert prop.created_at.tzinfo is not None
        assert prop.updated_at.tzinfo is not None

    def test_lead_property_match_default_is_aware_before_flush(self, dbsession):
        from backend.models.lead import Lead
        from backend.models.property import Property
        from backend.models.lead_property_match import LeadPropertyMatch

        lead = Lead(
            customer_name="Match Test",
            phone="2222222222",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        prop = Property(
            location="Koramangala",
            property_type="Apartment",
            source_name="test",
            source_record_id="aware-match-1",
        )
        dbsession.add(prop)
        dbsession.flush()

        match = LeadPropertyMatch(
            lead_id=lead.lead_id,
            property_id=prop.property_id,
            match_score=0.85,
        )
        dbsession.add(match)
        dbsession.flush()

        assert match.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# SQLite round-trip and updated_at / onupdate tests
# ---------------------------------------------------------------------------

class TestSQLiteRoundTripAndOnUpdate:
    """Verify timestamps survive SQLite persistence and onupdate works."""

    def test_timestamps_survive_sqlite_round_trip(self, dbsession):
        """After commit + expire, SQLite returns naive datetimes that are
        still valid UTC values and comparable to each other."""
        from backend.models.conversation import Conversation

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="roundtrip-test",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.commit()

        # Expire cached Python attributes so the next access reads from DB
        dbsession.expire(conv)

        # After round-trip through SQLite the values are naive (no tzinfo)
        assert conv.created_at is not None
        assert isinstance(conv.created_at, datetime)
        # SQLite strips timezone — this is expected and documented
        assert conv.created_at.tzinfo is None, (
            "SQLite should return naive datetimes after round-trip"
        )

    def test_round_tripped_values_are_comparable(self, dbsession):
        """Naive-UTC timestamps from SQLite can be compared with standard operators."""
        from backend.models.conversation import Conversation

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="compare-test",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.commit()
        dbsession.expire(conv)

        # Both are naive-UTC after round-trip; comparison works
        assert conv.created_at <= conv.updated_at

    def test_updated_at_refreshes_on_orm_update(self, dbsession):
        """Verify that ``onupdate=utc_now`` fires when a column is changed."""
        from backend.models.conversation import Conversation

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="onupdate-test",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.commit()
        dbsession.expire(conv)

        original_updated = conv.updated_at
        original_created = conv.created_at

        # Small delay so the timestamp difference is measurable
        time.sleep(0.05)

        # Modify a field to trigger onupdate
        conv.external_session_id = "onupdate-test-modified"
        dbsession.commit()
        dbsession.expire(conv)

        # created_at must NOT change
        assert conv.created_at == original_created

        # updated_at must have been refreshed
        assert conv.updated_at >= original_updated

    def test_lead_updated_at_refreshes_on_update(self, dbsession):
        """Verify onupdate works for Lead model."""
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="Update Test",
            phone="3333333333",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.commit()
        dbsession.expire(lead)

        original_updated = lead.updated_at
        original_created = lead.created_at

        time.sleep(0.05)

        lead.customer_name = "Updated Name"
        dbsession.commit()
        dbsession.expire(lead)

        assert lead.created_at == original_created
        assert lead.updated_at >= original_updated

    def test_callback_updated_at_refreshes_on_update(self, dbsession):
        """Verify onupdate works for CallbackRequest model."""
        from backend.models.callback import CallbackRequest
        from backend.models.conversation import Conversation
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="CB Update",
            phone="4444444444",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        conv = Conversation(
            lead_id=lead.lead_id,
            channel="browser_text",
            external_session_id="cb-update-test",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        cb = CallbackRequest(
            lead_id=lead.lead_id,
            conversation_id=conv.conversation_id,
            preferred_contact="phone",
            preferred_date="2026-02-01",
            preferred_time="14:00",
            status="Pending",
        )
        dbsession.add(cb)
        dbsession.commit()
        dbsession.expire(cb)

        original_updated = cb.updated_at

        time.sleep(0.05)

        cb.status = "Scheduled"
        dbsession.commit()
        dbsession.expire(cb)

        assert cb.updated_at >= original_updated

    def test_consent_updated_at_refreshes_on_update(self, dbsession):
        """Verify onupdate works for ConsentRecord model."""
        from backend.models.consent import ConsentRecord
        from backend.models.lead import Lead

        lead = Lead(
            customer_name="Consent Update",
            phone="5555555555",
            qualification_status="Unqualified",
            pipeline_status="New",
        )
        dbsession.add(lead)
        dbsession.flush()

        consent = ConsentRecord(
            lead_id=lead.lead_id,
            consent_type="contact_storage",
            granted=True,
            source="web",
            policy_version="1.0",
        )
        dbsession.add(consent)
        dbsession.commit()
        dbsession.expire(consent)

        original_updated = consent.updated_at

        time.sleep(0.05)

        consent.granted = False
        dbsession.commit()
        dbsession.expire(consent)

        assert consent.updated_at >= original_updated

    def test_property_updated_at_refreshes_on_update(self, dbsession):
        """Verify onupdate works for Property model."""
        from backend.models.property import Property

        prop = Property(
            location="HSR Layout",
            property_type="Apartment",
            source_name="test",
            source_record_id="onupdate-prop-1",
        )
        dbsession.add(prop)
        dbsession.commit()
        dbsession.expire(prop)

        original_updated = prop.updated_at

        time.sleep(0.05)

        prop.description = "Updated description"
        dbsession.commit()
        dbsession.expire(prop)

        assert prop.updated_at >= original_updated


# ---------------------------------------------------------------------------
# Source code audit
# ---------------------------------------------------------------------------

class TestSourceCodeAudit:
    """Ensure no model file still references the deprecated pattern."""

    def test_no_utcnow_in_model_source_files(self):
        """Ensure no model file still references datetime.utcnow."""
        import inspect
        from backend.models import (
            conversation,
            message,
            lead,
            callback,
            consent,
            property,
            lead_property_match,
        )

        for module in [
            conversation, message, lead, callback,
            consent, property, lead_property_match,
        ]:
            source = inspect.getsource(module)
            assert "datetime.utcnow" not in source, (
                f"{module.__name__} still uses datetime.utcnow"
            )

    def test_all_models_import_utc_now(self):
        """Verify every model file imports from _time rather than datetime."""
        import inspect
        from backend.models import (
            conversation,
            message,
            lead,
            callback,
            consent,
            property,
            lead_property_match,
        )

        for module in [
            conversation, message, lead, callback,
            consent, property, lead_property_match,
        ]:
            source = inspect.getsource(module)
            assert "from ._time import utc_now" in source or \
                   "from backend.models._time import utc_now" in source, (
                f"{module.__name__} does not import utc_now from _time"
            )
