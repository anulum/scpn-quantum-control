# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace lossless JSON transport
"""Read and write bounded JSON without losing numeric token identity."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .canonical import MAX_DEPTH, MAX_INTEGER_DIGITS, canonical_bytes

MAX_JSON_BYTES = 128 * 1024 * 1024
"""Expanded workspace JSON byte ceiling, distinct from archive admission."""


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"$.{key}: duplicate key")
        result[key] = value
    return result


def _integer(token: str) -> int | float:
    if len(token.lstrip("-")) > MAX_INTEGER_DIGITS:
        raise ValueError("$: integer scalar too large")
    return -0.0 if token == "-0" else int(token)


def _constant(token: str) -> object:
    raise ValueError(f"$: non-JSON constant {token}")


def _check_text(text: str) -> None:
    if len(text) > MAX_JSON_BYTES or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("$: JSON byte limit exceeded")
    depth = 0
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise ValueError("$: JSON depth exceeded")
        elif char in "]}":
            depth -= 1


def read_json(text: str) -> object:
    """Decode exact integer/float tokens and reject duplicate object members.

    Parameters
    ----------
    text
        JSON text within the expanded workspace byte/depth ceilings.

    Returns
    -------
    object
        Independent JSON containers, preserving integer, float and negative zero.

    Raises
    ------
    ValueError
        Syntax, Unicode, numeric value, duplicate key or resource limit fails.

    """
    _check_text(text)
    value: object = json.loads(
        text, parse_int=_integer, parse_constant=_constant, object_pairs_hook=_pairs
    )
    canonical_bytes("workspace_json.v1", value)
    return value


def _wire(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


def write_json(value: object) -> str:
    """Serialize exact scalar types without conflating floats with integers.

    Parameters
    ----------
    value
        Finite JSON-shaped values with supported Unicode and bounded depth.

    Returns
    -------
    str
        JSON with decimal/exponent markers on floats, including negative zero.

    Raises
    ------
    ValueError
        A value or resulting text exceeds the supported representation.

    """
    canonical_bytes("workspace_json.v1", value)
    text = json.dumps(_wire(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    _check_text(text)
    return text
