# -*- coding: utf-8 -*-
"""Source-aware property inventory (roadmap Phase 5).

Demo (SQLite) and live (API.SH) inventories are separate providers behind one
interface. They are never merged or substituted for each other here. This
package is not yet wired into the agent's tools.
"""

from .apish_provider import ApishPropertyProvider
from .base import PropertyProvider
from .credits import CreditPolicy, InMemoryCreditLedger
from .errors import (
    InventoryAPIError,
    InventoryAuthError,
    InventoryConfigError,
    InventoryError,
    InventoryNetworkError,
    InventoryQuotaError,
    InventoryRateLimitError,
    InventoryRequestError,
    InventoryResponseError,
    InventoryTimeoutError,
)
from .mock_provider import MockPropertyProvider
from .models import (
    Inventory,
    NormalizedProperty,
    ProviderResult,
    RecordKind,
    SearchCriteria,
    SourceRef,
)
from .service import InventoryOutcome, InventorySearchService
from .sqlite_provider import SQLiteProvider

__all__ = [
    "ApishPropertyProvider",
    "CreditPolicy",
    "InMemoryCreditLedger",
    "Inventory",
    "InventoryAPIError",
    "InventoryAuthError",
    "InventoryConfigError",
    "InventoryError",
    "InventoryNetworkError",
    "InventoryOutcome",
    "InventoryQuotaError",
    "InventoryRateLimitError",
    "InventoryRequestError",
    "InventoryResponseError",
    "InventorySearchService",
    "InventoryTimeoutError",
    "MockPropertyProvider",
    "NormalizedProperty",
    "PropertyProvider",
    "ProviderResult",
    "RecordKind",
    "SQLiteProvider",
    "SearchCriteria",
    "SourceRef",
]
