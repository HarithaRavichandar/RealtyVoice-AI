# -*- coding: utf-8 -*-
"""Tolerant location matching helpers for property search.

Property locations in the dataset are mostly city-level slugs such as
``bangalore``, ``new-delhi`` or ``greater-noida``, plus a few labels like
``Whitefield, Bangalore``. Customers (and speech-to-text) spell the same
places differently: "Bengaluru", "New Delhi", "greater noida", "White Field".

Matching therefore compares *squashed* text: lower-cased with spaces and
common punctuation removed, so "White Field" and "whitefield" are equal.
Well-known alternate city names are expanded so "Bengaluru" also finds
records stored as "bangalore", and vice versa.
"""

from __future__ import annotations

import difflib
from typing import Iterable, List

# Characters removed when squashing. The SQL side of the search removes the
# same set, so Python and SQL stay consistent.
SQUASH_CHARS = " -,._'/"

# Groups of names that refer to the same city. Every member of a group is
# treated as equivalent when matching.
ALIAS_GROUPS = (
    ("bangalore", "bengaluru", "bengalooru", "bangaluru"),
    ("gurgaon", "gurugram"),
    ("mumbai", "bombay"),
    ("kolkata", "calcutta"),
    ("chennai", "madras"),
    ("visakhapatnam", "vizag"),
    ("pune", "poona"),
    ("kochi", "cochin"),
    ("trichy", "tiruchirappalli"),
)


def squash(text: str) -> str:
    """Lower-case ``text`` and drop spaces and common punctuation."""
    lowered = text.lower()
    return "".join(ch for ch in lowered if ch not in SQUASH_CHARS)


def location_variants(text: str) -> List[str]:
    """Return the squashed forms of ``text`` including alias spellings.

    "Whitefield, Bengaluru" -> ["whitefieldbengaluru", "whitefieldbangalore",
    ...]. Returns an empty list when nothing is left after squashing.
    """
    base = squash(text)
    if not base:
        return []

    variants = [base]
    for group in ALIAS_GROUPS:
        for term in group:
            if term in base:
                for other in group:
                    if other != term:
                        variant = base.replace(term, other)
                        if variant not in variants:
                            variants.append(variant)
    return variants


def _display(part: str) -> str:
    """Human-friendly form of a stored location part (de-slugged)."""
    return " ".join(part.replace("-", " ").split()).title()


def suggest_locations(
    query: str,
    known_locations: Iterable[str],
    limit: int = 3,
    cutoff: float = 0.75,
) -> List[str]:
    """Suggest known locations that look similar to ``query``.

    Each stored location is considered both as a whole and by its
    comma-separated parts ("Whitefield, Bangalore" -> "Whitefield" and
    "Bangalore"), so a misspelt neighbourhood can still be matched. Only
    values that actually exist in ``known_locations`` are returned.
    """
    wanted = squash(query)
    if not wanted:
        return []

    candidates: dict[str, str] = {}
    for location in known_locations:
        if not location:
            continue
        candidates.setdefault(squash(location), _display(location))
        for part in location.split(","):
            key = squash(part)
            if key:
                candidates.setdefault(key, _display(part))

    matches = difflib.get_close_matches(
        wanted, list(candidates), n=limit, cutoff=cutoff
    )
    return [candidates[m] for m in matches]