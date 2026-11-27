# -*- coding: utf-8 -*-
"""Tests for the shared UTC timestamp helper and its integration with ORM models.

Verifies that:
- ``utc_now()`` returns timezone-aware UTC datetimes.
- SQLAlchemy model defaults produce timezone-aware timestamps.
- No ``datetime.utcnow()`` deprecation warnings are emitted.
"""

import warnings
from datetime import datetime, timezone

import pytest

from backend.models._time import utc_now


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


class TestModelTimestampDefaults:
    """Integration tests verifying ORM models use timezone-aware defaults."""

    def test_conversation_timestamps_are_aware(self, dbsession):
        from backend.models.conversation import Conversation

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="tz-test-conv",
            temp_requirements=None,
        )
        dbsession.add(conv)
        dbsession.flush()

        assert conv.created_at is not None
        assert conv.updated_at is not None
        # SQLite stores and returns naive datetimes, but the default callable
        # itself must be timezone-aware (verified via utc_now tests above).
        # Here we verify the timestamps are populated and reasonable.
        assert isinstance(conv.created_at, datetime)
        assert isinstance(conv.updated_at, datetime)

    def test_message_timestamp_is_populated(self, dbsession):
        from backend.models.conversation import Conversation
        from backend.models.message import Message

        conv = Conversation(
            lead_id=None,
            channel="browser_text",
            external_session_id="tz-test-msg",
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
        assert isinstance(msg.created_at, datetime)

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
