# -*- coding: utf-8 -*-
"""Errors raised by inventory providers.

Each carries a ``failure_kind`` from the shared failure vocabulary
(``backend.orchestrator.providers.failures``) and ``request_sent``: whether a
call actually left the server. Nothing here ever contains a credential.
"""

from __future__ import annotations

from backend.orchestrator.providers.failures import FailureKind


class InventoryError(Exception):
    """Base error for inventory providers."""

    failure_kind = FailureKind.PROVIDER_ERROR
    request_sent = True


class InventoryRequestError(InventoryError):
    """The search cannot be sent as asked (for example, no location)."""

    request_sent = False


class InventoryConfigError(InventoryError):
    """The provider is not configured (for example, no API key)."""

    failure_kind = FailureKind.AUTH_FAILURE
    request_sent = False


class InventoryAuthError(InventoryError):
    failure_kind = FailureKind.AUTH_FAILURE


class InventoryQuotaError(InventoryError):
    failure_kind = FailureKind.QUOTA_EXCEEDED


class InventoryRateLimitError(InventoryError):
    failure_kind = FailureKind.RATE_LIMITED


class InventoryTimeoutError(InventoryError):
    failure_kind = FailureKind.TIMEOUT


class InventoryNetworkError(InventoryError):
    failure_kind = FailureKind.NETWORK_ERROR


class InventoryResponseError(InventoryError):
    failure_kind = FailureKind.INVALID_RESPONSE


class InventoryAPIError(InventoryError):
    """Any other error status from the service."""

    failure_kind = FailureKind.PROVIDER_ERROR

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}" if detail else f"HTTP {status_code}")
