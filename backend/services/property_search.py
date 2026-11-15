# -*- coding: utf-8 -*-
"""Property search service.

Provides reusable search functionality over the ``Property`` ORM model.
All filtering, sorting and pagination are performed at the SQL level using
SQLAlchemy core expressions – the whole table is never loaded into Python.
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, date
from typing import Any, Callable, List, Optional, Tuple

from sqlalchemy import (
    asc,
    desc,
    func,
    and_,
    or_,
    case,
    select,
    text,
)
from sqlalchemy.orm import Session

from ..models.property import Property
from ..database import SessionLocal


class PropertySearchService:
    """Search service for ``Property`` records.

    Parameters
    ----------
    session_factory:
        Callable that returns a new ``Session``.  This mirrors the pattern used
        by ``PropertyImporter`` and enables easy testing via a temporary SQLite
        engine.
    """

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal, close_session: bool = True) -> None:
        """Create a PropertySearchService.

        Parameters
        ----------
        session_factory: Callable that returns a new Session (defaults to ``SessionLocal``).
        close_session: If ``True`` (default) the service will close the Session it creates.
                       Callers that provide their own Session and want to keep it open
                       should pass ``close_session=False``.
        """
        self.session_factory = session_factory
        self.close_session = close_session

    # ---------------------------------------------------------------------
    # Helper methods – price interval calculations
    # ---------------------------------------------------------------------
    @staticmethod
    def _price_low() -> Any:
        """SQL expression for the known lower bound of a property's price.

        Preference order: ``price_min`` → ``listing_price`` → ``price_max``.
        Returns ``NULL`` when all three columns are ``NULL``.
        """
        return func.coalesce(Property.price_min, Property.listing_price, Property.price_max)

    @staticmethod
    def _price_high() -> Any:
        """SQL expression for the known upper bound of a property's price.

        Preference order: ``price_max`` → ``listing_price`` → ``price_min``.
        Returns ``NULL`` when all three columns are ``NULL``.
        """
        return func.coalesce(Property.price_max, Property.listing_price, Property.price_min)

    # ---------------------------------------------------------------------
    # Core public API
    # ---------------------------------------------------------------------
    def search(
        self,
        *,
        location: Optional[str] = None,
        property_type: Optional[str] = None,
        configuration: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        possession_before: Optional[date] = None,
        availability: Optional[str] = None,
        sort: str = "price_asc",
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[int, List[Property]]:
        """Perform a property search.

        Returns the total matching count *before* pagination and a list of
        ``Property`` ORM objects limited by ``limit``/``offset``.
        """
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non‑negative")
        if limit > 100:
            raise ValueError("limit cannot exceed 100")
        if min_price is not None and min_price < 0:
            raise ValueError("min_price cannot be negative")
        if max_price is not None and max_price < 0:
            raise ValueError("max_price cannot be negative")
        if (
            min_price is not None
            and max_price is not None
            and min_price > max_price
        ):
            raise ValueError("min_price cannot be greater than max_price")

        session = self.session_factory()
        try:
            stmt = select(Property)
            filters = []

            # ----- location -----
            if location:
                stmt = stmt.where(Property.location.ilike(f"%{location}%"))

            # ----- property_type -----
            if property_type:
                stmt = stmt.where(Property.property_type.ilike(property_type))

            # ----- configuration (BHK) -----
            if configuration:
                # Normalise dash variants and whitespace to a single space, then compare exact (case‑insensitive)
                norm_input = (
                    configuration.replace("\u2011", " ")
                    .replace("\u2013", " ")
                    .replace("\u2014", " ")
                    .replace("-", " ")
                    .strip()
                )
                # Apply same normalisation to the column value
                config_col = func.replace(Property.configuration, "-", " ")
                config_col = func.replace(config_col, "\u2011", " ")
                config_col = func.replace(config_col, "\u2013", " ")
                config_col = func.replace(config_col, "\u2014", " ")
                config_col = func.trim(config_col)
                stmt = stmt.where(func.lower(config_col) == func.lower(norm_input))


            # ----- possession_before -----
            if possession_before:
                stmt = stmt.where(
                    Property.possession_date.isnot(None),
                    Property.possession_date <= possession_before,
                )

            # ----- availability -----
            if availability:
                stmt = stmt.where(
                    Property.availability_status.isnot(None),
                    Property.availability_status.ilike(availability),
                )

            # ----- price interval budget -----
            price_low = self._price_low()
            price_high = self._price_high()
            if min_price is not None and max_price is not None:
                # Both bounds – require known interval (both low and high not null)
                filters.append(
                    and_(price_low.isnot(None), price_high.isnot(None))
                )
                filters.append(price_low <= max_price)
                filters.append(price_high >= min_price)
            elif max_price is not None:
                # Upper bound only – low must be known
                filters.append(price_low.isnot(None))
                filters.append(price_low <= max_price)
            elif min_price is not None:
                # Lower bound only – high must be known
                filters.append(price_high.isnot(None))
                filters.append(price_high >= min_price)

            if filters:
                stmt = stmt.where(and_(*filters))

            # ----- total count before pagination -----
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = session.execute(count_stmt).scalar_one()

            # ----- sorting -----
            if sort == "price_desc":
                order_expr = desc(price_high)
            else:  # default price_asc
                order_expr = asc(price_low)
            # Nulls last for deterministic ordering
            stmt = stmt.order_by(order_expr.nullslast(), Property.property_id)

            # ----- pagination -----
            stmt = stmt.limit(limit).offset(offset)

            results = session.execute(stmt).scalars().all()
            return total, results
        finally:
            if self.close_session:
                session.close()

    def get_by_id(self, property_id: int) -> Optional[Property]:
        """Return a single ``Property`` by primary key or ``None`` if missing."""
        session = self.session_factory()
        try:
            return session.get(Property, property_id)
        finally:
            if self.close_session:
                session.close()

