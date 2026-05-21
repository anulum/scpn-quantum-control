# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- strict shot-count coercion helpers
"""Shared strict count coercion utilities for hardware adapters."""

from __future__ import annotations

import string
from collections.abc import Sequence
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


def strict_integer_value(value: object, *, field_name: str = "value") -> int:
    """Return a strictly validated integer value."""

    return _strict_int(value, field_name=field_name)


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


def strict_binary_bitstring_key(value: object, *, field_name: str = "count key") -> str:
    """Normalise a count key to a strict binary bitstring."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must be a non-empty bitstring")
        if text != value:
            raise ValueError(f"{field_name} must not include whitespace padding")
        if any(char not in {"0", "1"} for char in text):
            raise ValueError(f"{field_name} must contain only binary digits")
        return text
    if isinstance(value, Sequence):
        if not value:
            raise ValueError(f"{field_name} must be a non-empty bitstring")
        bits = [strict_non_negative_count(item, field_name=field_name) for item in value]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"{field_name} must contain only binary digits")
        return "".join(str(bit) for bit in bits)
    raise ValueError(f"{field_name} must be a bitstring or sequence of bits")


def strict_fixed_width_bitstring_key(
    value: object,
    *,
    width: int,
    field_name: str = "count key",
) -> str:
    """Normalise and validate a binary bitstring key with exact width."""

    if width <= 0:
        raise ValueError("bitstring width must be positive")
    key = strict_binary_bitstring_key(value, field_name=field_name)
    if len(key) > width:
        raise ValueError(f"{field_name} must be exactly {width} bits wide")
    return key.zfill(width)


def strict_provider_job_id(value: object, *, field_name: str = "provider job id") -> str:
    """Validate and canonicalise provider job identifiers."""

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if any(char in string.whitespace and char != " " for char in text):
        raise ValueError(f"{field_name} must not include control whitespace")
    if any(ord(char) < 32 for char in text):
        raise ValueError(f"{field_name} must not include control characters")
    return text


def strict_shot_conservation(
    counts: dict[str, int], *, expected_shots: int, field_name: str = "shot count"
) -> int:
    """Validate that decoded counts exactly match expected provider shots."""

    if expected_shots <= 0:
        raise ValueError(f"{field_name} expectation must be positive")
    observed = sum(counts.values())
    if observed != expected_shots:
        raise ValueError(f"{field_name} mismatch: expected {expected_shots}, observed {observed}")
    return expected_shots
