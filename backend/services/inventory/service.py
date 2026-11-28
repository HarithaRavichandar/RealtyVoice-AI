# -*- coding: utf-8 -*-
"""Search orchestration for inventory providers.

Rules enforced here (roadmap decisions 3, 4 and 6):

* One search calls exactly one provider, once. Demo and live results are never
  merged, and a failed live search never falls back to demo data.
* Live searches are credit-controlled per conversation. When the limit is
  reached no call is made.
* Live listings are re-checked against the request here (budget, bedrooms,
  possession, type). Clear mismatches are dropped; unknown is kept apart from
  mismatch; listings whose price cannot be checked against a requested budget
  are held out and counted, not recommended.
* Results are de-duplicated by source + record kind + external id.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.orchestrator.providers.failures import CUSTOMER_MESSAGES, FailureKind

from .base import PropertyProvider
from .credits import CreditLedger, CreditPolicy, InMemoryCreditLedger
from .dedup import dedupe
from .errors import InventoryConfigError, InventoryError, InventoryRequestError
from .models import (
    EXCLUDED_BEDROOMS_MISMATCH,
    EXCLUDED_BUDGET_MISMATCH,
    EXCLUDED_BUDGET_UNKNOWN,
    EXCLUDED_DUPLICATES,
    EXCLUDED_POSSESSION_MISMATCH,
    Inventory,
    NormalizedProperty,
    ProviderResult,
    SearchCriteria,
)
from .validation import (
    MISMATCH,
    UNKNOWN,
    check_bedrooms,
    check_budget,
    check_possession,
    check_property_type,
)

logger = logging.getLogger(__name__)

# Outcome statuses.
OK = "ok"
NO_RESULTS = "no_results"
CREDIT_LIMIT_REACHED = "credit_limit_reached"
INVALID_REQUEST = "invalid_request"
NOT_CONFIGURED = "not_configured"
FAILED = "failed"

# Failures after which API.SH did not do the work (refused calls). Everything
# else that was sent is counted as one credit: the documentation does not say
# otherwise, and zero is not assumed.
_FREE_FAILURES = {
    FailureKind.AUTH_FAILURE,
    FailureKind.QUOTA_EXCEEDED,
    FailureKind.RATE_LIMITED,
}

_MESSAGES = {
    CREDIT_LIMIT_REACHED: (
        "The limit of live listing searches for this conversation has been reached."
    ),
    INVALID_REQUEST: "A location is needed to search live listings.",
    NOT_CONFIGURED: "Live listings are not available right now.",
    NO_RESULTS: "No live listings matched this search.",
}


@dataclass(frozen=True)
class InventoryOutcome:
    """The result of one search, with its source and credit position."""

    status: str
    inventory: Inventory
    source: str
    result: Optional[ProviderResult] = None
    failure_kind: Optional[FailureKind] = None
    message: Optional[str] = None
    credits_charged: int = 0
    credits_used_total: Optional[int] = None
    credits_limit: Optional[int] = None
    # Always false: nothing here substitutes another inventory for a failure.
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _apply_checks(
    listing: NormalizedProperty, criteria: SearchCriteria
) -> Tuple[Optional[NormalizedProperty], Optional[str]]:
    budget = check_budget(listing, criteria.min_budget, criteria.max_budget)
    bedrooms = check_bedrooms(listing, criteria.configuration)
    possession = check_possession(listing, criteria.possession_before)
    if budget == MISMATCH:
        return None, EXCLUDED_BUDGET_MISMATCH
    if bedrooms == MISMATCH:
        return None, EXCLUDED_BEDROOMS_MISMATCH
    if possession == MISMATCH:
        return None, EXCLUDED_POSSESSION_MISMATCH
    if budget == UNKNOWN:
        return None, EXCLUDED_BUDGET_UNKNOWN
    checks = {
        "budget": budget,
        "bedrooms": bedrooms,
        "property_type": check_property_type(listing, criteria.property_type),
        "possession": possession,
    }
    return dataclasses.replace(listing, checks=checks), None


class InventorySearchService:
    """Runs one search against one named inventory."""

    def __init__(
        self,
        providers: Mapping[Inventory, PropertyProvider],
        *,
        ledger: Optional[CreditLedger] = None,
        policy: Optional[CreditPolicy] = None,
    ) -> None:
        self._providers = dict(providers)
        self._ledger: CreditLedger = ledger or InMemoryCreditLedger()
        self._policy = policy or CreditPolicy()

    async def search(
        self,
        inventory: Inventory,
        conversation_key: str,
        criteria: SearchCriteria,
    ) -> InventoryOutcome:
        provider = self._providers.get(inventory)
        if provider is None:
            raise ValueError(f"no provider is configured for {inventory.value!r} inventory")
        live = provider.is_live
        limit = self._policy.limit_per_conversation if live else None

        if live and self._ledger.used(conversation_key) >= self._policy.limit_per_conversation:
            return InventoryOutcome(
                status=CREDIT_LIMIT_REACHED,
                inventory=inventory,
                source=provider.source_name,
                message=_MESSAGES[CREDIT_LIMIT_REACHED],
                credits_used_total=self._ledger.used(conversation_key),
                credits_limit=limit,
            )

        try:
            result = await provider.search(criteria)
        except InventoryRequestError:
            return self._failure(provider, INVALID_REQUEST, None, 0, conversation_key, limit)
        except InventoryConfigError:
            return self._failure(provider, NOT_CONFIGURED, FailureKind.AUTH_FAILURE, 0, conversation_key, limit)
        except InventoryError as exc:
            charge = 0
            if live and exc.request_sent and exc.failure_kind not in _FREE_FAILURES:
                charge = 1
            return self._failure(provider, FAILED, exc.failure_kind, charge, conversation_key, limit)
        except Exception:
            if live:
                self._ledger.add(conversation_key, 1)  # a call may have been made
            raise

        charged = 0
        if live:
            charged = result.credits_used if result.credits_used is not None else 1
            self._ledger.add(conversation_key, charged)

        result = self._postprocess(result, criteria, live)
        status = OK if result.listings else NO_RESULTS
        return InventoryOutcome(
            status=status,
            inventory=inventory,
            source=provider.source_name,
            result=result,
            message=None if result.listings else _MESSAGES[NO_RESULTS],
            credits_charged=charged,
            credits_used_total=self._ledger.used(conversation_key) if live else None,
            credits_limit=limit,
        )

    # ------------------------------------------------------------------
    def _failure(
        self,
        provider: PropertyProvider,
        status: str,
        kind: Optional[FailureKind],
        charge: int,
        conversation_key: str,
        limit: Optional[int],
    ) -> InventoryOutcome:
        live = provider.is_live
        if charge:
            self._ledger.add(conversation_key, charge)
        message = _MESSAGES.get(status) if status != FAILED else CUSTOMER_MESSAGES.get(kind)
        return InventoryOutcome(
            status=status,
            inventory=provider.inventory,
            source=provider.source_name,
            failure_kind=kind,
            message=message,
            credits_charged=charge,
            credits_used_total=self._ledger.used(conversation_key) if live else None,
            credits_limit=limit,
        )

    @staticmethod
    def _postprocess(
        result: ProviderResult, criteria: SearchCriteria, live: bool
    ) -> ProviderResult:
        excluded: Dict[str, int] = dict(result.excluded)
        listings, removed = dedupe(result.listings)
        if removed:
            excluded[EXCLUDED_DUPLICATES] = excluded.get(EXCLUDED_DUPLICATES, 0) + removed
        if live:
            kept: List[NormalizedProperty] = []
            for listing in listings:
                checked, reason = _apply_checks(listing, criteria)
                if reason:
                    excluded[reason] = excluded.get(reason, 0) + 1
                else:
                    kept.append(checked)
            listings = kept
        return dataclasses.replace(result, listings=tuple(listings), excluded=excluded)
