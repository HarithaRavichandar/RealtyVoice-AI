# -*- coding: utf-8 -*-
"""Source-aware de-duplication.

Identity is ``source + record kind + external id``. A project id and a
property id are different things even when the numbers are equal, and records
from different sources (demo vs live) are never compared with each other.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .models import NormalizedProperty


def dedupe(listings: Iterable[NormalizedProperty]) -> Tuple[List[NormalizedProperty], int]:
    """Keep the first listing for each identity. Returns ``(kept, removed)``."""
    seen = set()
    kept: List[NormalizedProperty] = []
    removed = 0
    for listing in listings:
        key = listing.ref.key
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(listing)
    return kept, removed
