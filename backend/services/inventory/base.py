# -*- coding: utf-8 -*-
"""The property-provider interface (roadmap task 5.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Inventory, ProviderResult, SearchCriteria


class PropertyProvider(ABC):
    """A source of individual properties.

    A provider does one thing: run one search and return what it found,
    labelled with its source. It never merges with another provider and never
    falls back to another one; that decision stays with the caller.
    """

    #: Stable identifier of the source, for example ``apish_99acres``.
    source_name: str = ""
    #: ``live`` providers cost money per call and are credit-controlled.
    inventory: Inventory = Inventory.DEMO

    @property
    def is_live(self) -> bool:
        return self.inventory is Inventory.LIVE

    @abstractmethod
    async def search(self, criteria: SearchCriteria) -> ProviderResult:
        """Run one search. Raises ``InventoryError`` on failure."""
        raise NotImplementedError
