# -*- coding: utf-8 -*-
"""Additive, idempotent schema upgrades for existing SQLite databases.

``create_all`` creates missing tables but never adds columns to existing ones.
This adds the nullable lead-qualification columns (Phase 3) when they are
missing. It only ever runs ``ALTER TABLE ... ADD COLUMN`` - no data is
changed, moved or deleted - and does nothing on a database that is already
up to date or has no ``leads`` table yet.
"""

import logging
from typing import List

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

LEAD_QUALIFICATION_COLUMNS = {
    "purchase_timeline": "VARCHAR",
    "visit_readiness": "VARCHAR",
    "contact_preference": "VARCHAR",
    "buyer_intent": "VARCHAR",
    "score_breakdown": "JSON",
}


def ensure_lead_qualification_columns(engine: Engine) -> List[str]:
    """Add any missing lead-qualification columns. Returns the names added."""
    inspector = inspect(engine)
    if "leads" not in inspector.get_table_names():
        return []
    existing = {c["name"] for c in inspector.get_columns("leads")}
    missing = [c for c in LEAD_QUALIFICATION_COLUMNS if c not in existing]
    if not missing:
        return []
    with engine.begin() as conn:
        for column in missing:
            conn.execute(
                text(f"ALTER TABLE leads ADD COLUMN {column} {LEAD_QUALIFICATION_COLUMNS[column]}")
            )
    logger.info("Added lead qualification columns: %s", ", ".join(missing))
    return missing


def ensure_live_credit_ledger_table(engine: Engine) -> bool:
    """Create the live-credit ledger table when it is missing.

    Purely additive: it creates ``live_credit_ledger`` if absent and never
    touches any other table or row. Returns True when the table was created.
    """
    from backend.models.live_credit import LiveCreditEntry

    if "live_credit_ledger" in inspect(engine).get_table_names():
        return False
    LiveCreditEntry.__table__.create(bind=engine, checkfirst=True)
    logger.info("Created table live_credit_ledger")
    return True
