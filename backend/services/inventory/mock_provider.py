# -*- coding: utf-8 -*-
"""A scripted provider for tests. Never touches the network or a database."""

from __future__ import annotations

from typing import List, Union

from .base import PropertyProvider
from .errors import InventoryError
from .models import Inventory, ProviderResult, SearchCriteria


class MockPropertyProvider(PropertyProvider):
    """Returns a prepared result, or raises a prepared error, and records calls."""

    def __init__(
        self,
        outcome: Union[ProviderResult, InventoryError, None] = None,
        *,
        source_name: str = "mock_live",
        inventory: Inventory = Inventory.LIVE,
    ) -> None:
        self.source_name = source_name
        self.inventory = inventory
        self._outcome = outcome or ProviderResult(source=source_name, inventory=inventory)
        self.calls: List[SearchCriteria] = []

    async def search(self, criteria: SearchCriteria) -> ProviderResult:
        self.calls.append(criteria)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome
