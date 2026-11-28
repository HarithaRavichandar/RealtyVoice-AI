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
DUPLICATE_REQUEST = "duplicate_request"

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
    DUPLICATE_REQUEST: (
        "This live search was already made, so it was not sent again."
    ),
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
    # Split of ``credits_charged``: what the provider reported, and what was
    # counted toward the limit without confirmation. Only filled when the
    # ledger supports reservations.
    credits_confirmed: int = 0
    credits_unconfirmed: int = 0
    request_id: Optional[str] = None
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
        *,
        request_id: Optional[str] = None,
    ) -> InventoryOutcome:
        provider = self._providers.get(inventory)
        if provider is None:
            raise ValueError(f"no provider is configured for {inventory.value!r} inventory")
        live = provider.is_live
        limit = self._policy.limit_per_conversation if live else None
        # A ledger that can reserve records the request before it is sent and
        # refuses a repeated request id. Others keep the simple add() flow.
        reserving = live and request_id is not None and hasattr(self._ledger, "reserve")

        if live and self._ledger.used(conversation_key) >= self._policy.limit_per_conversation:
            return InventoryOutcome(
                status=CREDIT_LIMIT_REACHED,
                inventory=inventory,
                source=provider.source_name,
                message=_MESSAGES[CREDIT_LIMIT_REACHED],
                credits_used_total=self._ledger.used(conversation_key),
                credits_limit=limit,
                request_id=request_id,
            )

        if reserving and self._ledger.reserve(
            conversation_key, provider.source_name, request_id
        ) == "duplicate":
            return InventoryOutcome(
                status=DUPLICATE_REQUEST,
                inventory=inventory,
                source=provider.source_name,
                message=_MESSAGES[DUPLICATE_REQUEST],
                credits_used_total=self._ledger.used(conversation_key),
                credits_limit=limit,
                request_id=request_id,
            )

        def settle(confirmed: int, unconfirmed: int, outcome: str,
                   kind: Optional[FailureKind] = None) -> None:
            """Record the final credit position of this request."""
            if reserving:
                self._ledger.settle(
                    conversation_key, provider.source_name, request_id,
                    consumed=confirmed, unconfirmed=unconfirmed, outcome=outcome,
                    failure_kind=kind.value if kind else None,
                )
            elif confirmed + unconfirmed:
                self._ledger.add(conversation_key, confirmed + unconfirmed)

        try:
            result = await provider.search(criteria)
        except InventoryRequestError:
            settle(0, 0, INVALID_REQUEST)
            return self._failure(provider, INVALID_REQUEST, None, 0, conversation_key, limit,
                                 request_id=request_id)
        except InventoryConfigError:
            settle(0, 0, NOT_CONFIGURED, FailureKind.AUTH_FAILURE)
            return self._failure(provider, NOT_CONFIGURED, FailureKind.AUTH_FAILURE, 0,
                                 conversation_key, limit, request_id=request_id)
        except InventoryError as exc:
            # Refused calls are free per the API.SH documentation. Any other
            # failure after the request was sent is counted conservatively,
            # but as unconfirmed: its real cost is not known.
            charge = 0
            if live and exc.request_sent and exc.failure_kind not in _FREE_FAILURES:
                charge = 1
            settle(0, charge, FAILED, exc.failure_kind)
            return self._failure(provider, FAILED, exc.failure_kind, charge, conversation_key,
                                 limit, request_id=request_id, unconfirmed=charge)
        except Exception:
            # A call may have been made: keep the conservative hold.
            if reserving:
                pass  # the pending hold already counts
            elif live:
                self._ledger.add(conversation_key, 1)
            raise

        confirmed = unconfirmed = 0
        if live:
            if result.credits_used is not None:
                confirmed = result.credits_used
            else:
                unconfirmed = 1  # call succeeded but its cost was not reported
            settle(confirmed, unconfirmed, OK if result.listings else NO_RESULTS)
        charged = confirmed + unconfirmed

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
            credits_confirmed=confirmed,
            credits_unconfirmed=unconfirmed,
            request_id=request_id,
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
        *,
        request_id: Optional[str] = None,
        unconfirmed: int = 0,
    ) -> InventoryOutcome:
        live = provider.is_live
        # With a reserving ledger the charge was already recorded by settle().
        if charge and not (request_id is not None and hasattr(self._ledger, "reserve")):
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
            credits_unconfirmed=unconfirmed,
            request_id=request_id,
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
