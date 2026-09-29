# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace canonical encoding
"""Encode typed values without conflating integer, float or user-array tags."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping

MAX_DEPTH = 64
"""Maximum nested value depth admitted by the workspace format."""
MAX_INTEGER_DIGITS = 4096
"""Bounded decimal scalar capacity, shared with the browser reader."""


def _scalar_string(value: str, path: str) -> str:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{path}: invalid Unicode scalar")
    return value


def _tag(value: object, path: str, depth: int, ancestors: set[int]) -> object:
    if depth > MAX_DEPTH or (depth == MAX_DEPTH and isinstance(value, (Mapping, list, tuple))):
        raise ValueError(f"{path}: depth exceeds {MAX_DEPTH}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _scalar_string(value, path)
    if isinstance(value, int):
        # 4096 decimal digits need at most 13607 bits; avoid costly str on huge inputs.
        if value.bit_length() > 13607:
            raise ValueError(f"{path}: integer scalar too large")
        decimal = str(value)
        if len(decimal.lstrip("-")) > MAX_INTEGER_DIGITS:
            raise ValueError(f"{path}: integer scalar too large")
        return ["integer", decimal]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float")
        return ["float64", struct.pack(">d", value).hex()]
    if not isinstance(value, (Mapping, list, tuple)):
        raise ValueError(f"{path}: unsupported value type")
    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"{path}: ancestor cycle")
    ancestors.add(identity)
    try:
        if isinstance(value, Mapping):
            keys: list[str] = []
            for key in value:
                if not isinstance(key, str):
                    raise ValueError(f"{path}: object key must be a string")
                keys.append(_scalar_string(key, path))
            return [
                "object",
                [
                    [key, _tag(value[key], f"{path}.{key}", depth + 1, ancestors)]
                    for key in sorted(keys, key=lambda item: item.encode("utf-8"))
                ],
            ]
        return [
            "array",
            [
                _tag(item, f"{path}[{index}]", depth + 1, ancestors)
                for index, item in enumerate(value)
            ],
        ]
    finally:
        ancestors.remove(identity)


def canonical_bytes(schema: str, body: object) -> bytes:
    """Encode a typed canonical tree with its schema domain prefix.

    Parameters
    ----------
    schema
        Nonempty scalar string without CR or LF.
    body
        Finite JSON-shaped values; integers and binary64 floats stay distinct.

    Returns
    -------
    bytes
        Schema, LF and compact UTF-8 tagged JSON.

    Raises
    ------
    ValueError
        Prefix, scalar, key, recursion depth or ancestor cycle is unsupported.

    """
    if not schema or "\n" in schema or "\r" in schema:
        raise ValueError("schema: invalid domain prefix")
    _scalar_string(schema, "schema")
    tagged = _tag(body, "$", 0, set())
    return (
        schema.encode("utf-8")
        + b"\n"
        + json.dumps(tagged, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def canonical_digest(schema: str, body: object) -> str:
    """Hash the exact schema-prefixed canonical bytes.

    Parameters
    ----------
    schema
        Nonempty schema domain without CR or LF.
    body
        Values admitted by ``canonical_bytes``.

    Returns
    -------
    str
        Lowercase SHA-256 hex without an algorithm prefix.

    """
    return hashlib.sha256(canonical_bytes(schema, body)).hexdigest()
