# -*- coding: utf-8 -*-
"""Per-conversation credit control for live inventory.

Policy (roadmap decision 4): one search is one API call; no automatic
pagination, no automatic ``fresh`` requests, no retries; a configurable
per-conversation limit (initially 3) stops further calls.

Nothing here assumes ``fresh``, ``raw`` or pagination are free. The ledger is
an interface so a database-backed version can replace the in-memory one when a
schema change is approved; until then counts reset when the process restarts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass(frozen=True)
class CreditPolicy:
    """How many API credits one conversation may spend on live searches."""

    limit_per_conversation: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.limit_per_conversation, bool) or not isinstance(
            self.limit_per_conversation, int
        ) or self.limit_per_conversation < 0:
            raise ValueError("limit_per_conversation must be a non-negative integer")


class CreditLedger(Protocol):
    def used(self, conversation_key: str) -> int: ...

    def add(self, conversation_key: str, credits: int) -> int: ...


class InMemoryCreditLedger:
    """Thread-safe, per-process ledger."""

    def __init__(self) -> None:
        self._used: Dict[str, int] = {}
        self._lock = threading.Lock()

    def used(self, conversation_key: str) -> int:
        with self._lock:
            return self._used.get(conversation_key, 0)

    def add(self, conversation_key: str, credits: int) -> int:
        if credits < 0:
            raise ValueError("credits cannot be negative")
        with self._lock:
            total = self._used.get(conversation_key, 0) + credits
            self._used[conversation_key] = total
            return total
