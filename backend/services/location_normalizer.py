# -*- coding: utf-8 -*-
"""India-wide, data-driven location matching for property search.

Property locations in the dataset are mostly city-level slugs such as
``bangalore``, ``new-delhi`` or ``greater-noida``, plus a few locality
labels such as ``Whitefield, Bangalore`` or ``Kolkata, Salt Lake``. Customers
(and speech-to-text) spell the same places differently: "Bengaluru",
"New Delhi", "greater noida", "White Field".

The matching is *label based*: it is resolved against the distinct location
labels stored in the database (about a hundred), never against every row.
The search then filters with an exact ``IN (matching labels)`` query.

Matching rules, in order:

1. Names are compared *squashed*: lower-cased with spaces and common
   punctuation removed, so "White Field", "white-field" and "whitefield"
   are equal.
2. Alternate city and locality names come from ``location_aliases.json``
   (Bengaluru / Bangalore, Gurugram / Gurgaon, Delhi / New Delhi,
   Electronic City / E-City, ...). Several aliased names in one query are
   combined ("E-City Bengaluru" also tries "Electronic City Bangalore").
3. A label matches exactly when the whole label, or one of its
   comma-separated parts, equals the query (or an alias spelling of it).
   So "Noida" matches ``noida`` but not ``greater-noida``, and "Patna" does
   not match ``visakhapatnam``. The parts of a label are order independent
   ("Salt Lake, Kolkata" matches ``Kolkata, Salt Lake``).
4. Only when nothing matches exactly does a partial (substring) match
   apply, so a typed prefix such as "mumba" still works.

Nothing here creates inventory: suggestions for searches that found nothing
are built only from locations that exist in the database.
"""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Characters removed when squashing.
SQUASH_CHARS = " -,._'/"
_SQUASH_RE = re.compile("[" + re.escape(SQUASH_CHARS) + "]")

ALIAS_FILE = Path(__file__).with_name("location_aliases.json")

AliasGroups = Tuple[Tuple[str, ...], ...]


@lru_cache(maxsize=8192)
def squash(text: str) -> str:
    """Lower-case ``text`` and drop spaces and common punctuation."""
    return _SQUASH_RE.sub("", text.lower())


# ---------------------------------------------------------------------------
# Alias configuration
# ---------------------------------------------------------------------------

def load_alias_groups(path: Optional[Path] = None) -> AliasGroups:
    """Load alias groups from a JSON file (default: ``location_aliases.json``).

    Names are squashed. Raises ``ValueError`` when a group has fewer than two
    distinct names or when a name appears in more than one group.
    """
    source = Path(path) if path is not None else ALIAS_FILE
    with open(source, encoding="utf-8") as fh:
        raw = json.load(fh)

    groups: List[Tuple[str, ...]] = []
    seen: Dict[str, int] = {}
    for index, group in enumerate(raw.get("groups", [])):
        names: List[str] = []
        for name in group:
            key = squash(name)
            if key and key not in names:
                names.append(key)
        if len(names) < 2:
            raise ValueError(
                f"Alias group {index} needs at least two distinct names: {group!r}"
            )
        for key in names:
            if key in seen:
                raise ValueError(
                    f"Alias name {key!r} appears in groups {seen[key]} and {index}"
                )
            seen[key] = index
        groups.append(tuple(names))
    return tuple(groups)


@lru_cache(maxsize=1)
def default_alias_groups() -> AliasGroups:
    """Alias groups from the bundled JSON file (loaded once)."""
    return load_alias_groups()


def _groups(groups: Optional[AliasGroups]) -> AliasGroups:
    return default_alias_groups() if groups is None else groups


# ---------------------------------------------------------------------------
# Query variants and label matching
# ---------------------------------------------------------------------------

def location_variants(text: str, groups: Optional[AliasGroups] = None) -> List[str]:
    """Squashed forms of ``text`` including alias spellings.

    "Whitefield, Bengaluru" -> ["whitefieldbengaluru", "whitefieldbangalore",
    "whitefieldbengalooru", ...]. Returns an empty list when nothing is left
    after squashing.
    """
    base = squash(text)
    if not base:
        return []

    variants = [base]
    for group in _groups(groups):
        # Apply each group to every variant built so far, so a query that
        # uses two aliased names ("E-City Bengaluru") still reaches the stored
        # label ("Electronic City, Bangalore").
        for current in list(variants):
            present = [name for name in group if name in current]
            if not present:
                continue
            term = max(present, key=len)  # the most specific name in the group
            for other in group:
                if other != term:
                    variant = current.replace(term, other)
                    if variant not in variants:
                        variants.append(variant)
    return variants


def _parts(label: str) -> List[str]:
    return [p for p in (squash(p) for p in label.split(",")) if p]


def _label_keys(label: str) -> set:
    """Squashed names a label can be matched by (exact matching)."""
    full = squash(label)
    keys = {full}
    parts = _parts(label)
    keys.update(parts)
    if 1 < len(parts) <= 3:
        keys.add("".join(reversed(parts)))  # "Salt Lake, Kolkata" order
    return keys


def resolve_locations(
    query: str,
    labels: Iterable[str],
    groups: Optional[AliasGroups] = None,
) -> List[str]:
    """Return the stored location labels that match ``query``.

    Exact matches (whole label or one of its comma-separated parts, alias
    spellings included) win. If there are none, labels that merely contain
    the query are returned so a typed prefix still works. An empty list means
    nothing matched.
    """
    variants = location_variants(query, groups)
    if not variants:
        return []

    wanted = set(variants)
    exact: List[str] = []
    partial: List[str] = []
    for label in labels:
        if not label:
            continue
        if _label_keys(label) & wanted:
            exact.append(label)
        else:
            full = squash(label)
            if any(v in full for v in variants):
                partial.append(label)
    return exact or partial


# ---------------------------------------------------------------------------
# Zero-result suggestions
# ---------------------------------------------------------------------------

def _display(part: str) -> str:
    """Human-friendly form of a stored location part (de-slugged)."""
    return " ".join(part.replace("-", " ").split()).title()


def _tokens(query: str) -> List[str]:
    return [t for t in re.split(r"[\s,\-./_']+", query) if t]


def suggest_locations(
    query: str,
    labels: Iterable[str],
    groups: Optional[AliasGroups] = None,
    limit: int = 5,
    cutoff: float = 0.75,
) -> List[Dict[str, str]]:
    """Suggest real locations when ``query`` matched nothing.

    Every suggestion is built from ``labels`` (the locations that exist in the
    database); nothing is invented. Two kinds are returned:

    * ``similar_name`` - a stored name or part of one that is spelt similarly
      (for speech-to-text slips such as "Whitefeild"). The caller should
      confirm it with the customer.
    * ``parent_city`` - the query names a place inside a city that exists as
      a city-level label (for example "Indiranagar, Bengaluru" -> Bangalore).
      This only means the dataset has city-level records; it says nothing
      about inventory in the specific area.
    """
    labels = [l for l in labels if l and l.strip()]
    wanted = squash(query)
    if not wanted:
        return []

    suggestions: List[Dict[str, str]] = []
    used: set = set()

    # --- similarly spelt names, from whole labels and their parts ----------
    candidates: Dict[str, str] = {}
    for label in labels:
        candidates.setdefault(squash(label), _display(label))
        for part in label.split(","):
            key = squash(part)
            if key:
                candidates.setdefault(key, _display(part))
    for key in difflib.get_close_matches(wanted, list(candidates), n=limit, cutoff=cutoff):
        suggestions.append(
            {
                "location": candidates[key],
                "type": "similar_name",
                "note": (
                    "A similarly spelt location exists in the data; confirm "
                    "with the customer that this is what they meant."
                ),
            }
        )
        used.add(key)

    # --- parent city: a city-level label named inside the query ------------
    cities: Dict[str, str] = {}
    for label in labels:
        if "," not in label:
            cities[squash(label)] = _display(label)

    tokens = _tokens(query)
    found: List[Tuple[int, int, str]] = []  # (run length, start, city key)
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            if end - start == len(tokens):
                continue  # the whole query is not "a place inside a city"
            run = "".join(tokens[start:end])
            for variant in location_variants(run, groups):
                if variant in cities:
                    found.append((end - start, start, variant))
    parent_count = 0
    for _length, _start, key in sorted(found, key=lambda f: (-f[0], f[1])):
        if key in used:
            continue
        if parent_count >= 2:  # keep the list short and unambiguous
            break
        used.add(key)
        parent_count += 1
        suggestions.append(
            {
                "location": cities[key],
                "type": "parent_city",
                "note": (
                    f"No records match '{query}'. '{cities[key]}' is a "
                    "city-level label in the data; it does not confirm any "
                    "listings in the specific area named."
                ),
            }
        )

    return suggestions[:limit]
