# backend/models/__init__.py
"""Import all ORM model classes so that Base.metadata registers them.
This module is imported by the application and by tests to ensure tables are
created when Base.metadata.create_all() is called.
"""
from .property import Property
from .lead import Lead
from .conversation import Conversation
from .message import Message
from .callback import CallbackRequest
from .consent import ConsentRecord
from .lead_property_match import LeadPropertyMatch
from .live_credit import LiveCreditEntry