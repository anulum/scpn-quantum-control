# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- strict shot-count coercion helpers
"""Shared strict count coercion utilities for hardware adapters."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from numbers import Integral


def strict_non_negative_count(value: object, *, field_name: str = "count") -> int:
    """Return a validated non-negative integer count.

    This is intentionally strict: non-integral numerics (e.g. ``1.5``) and opaque
    numeric strings (e.g. ``"1.5"``) are rejected instead of silently truncated.
    """

    integer = _strict_int(value, field_name=field_name)
    if integer < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return integer


def _strict_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must be an integer")
        try:
            decimal_value = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if decimal_value != decimal_value.to_integral_value():
            raise ValueError(f"{field_name} must be an integer")
        return int(decimal_value)
    raise ValueError(f"{field_name} must be an integer")
