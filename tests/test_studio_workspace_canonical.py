# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace canonical conformance tests
"""Exercise the public codec against explicit cross-language byte oracles."""

import json
import math
import struct
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control.studio_workspace.canonical import (
    canonical_bytes,
    canonical_digest,
)

_CORPUS = json.loads((Path(__file__).parent / "data/studio_workspace/canonical.json").read_text())


def _materialise(descriptor: dict[str, object]) -> object:
    kind = descriptor["kind"]
    if kind == "null":
        return None
    if kind == "integer":
        return int(str(descriptor["decimal"]))
    if kind == "float64":
        return cast(float, struct.unpack(">d", bytes.fromhex(str(descriptor["bits"])))[0])
    if kind == "array":
        return [_materialise(item) for item in cast(list[dict[str, object]], descriptor["items"])]
    if kind == "object":
        return {
            str(key): _materialise(cast(dict[str, object], value))
            for key, value in cast(list[tuple[object, object]], descriptor["entries"])
        }
    if kind == "string_codepoints":
        return "".join(chr(int(value, 16)) for value in cast(list[str], descriptor["hex"]))
    return descriptor["value"]


@pytest.mark.parametrize("case", _CORPUS["cases"], ids=lambda case: case["id"])
def test_explicit_byte_oracle(case: dict[str, object]) -> None:
    """Match independently declared bytes, including scalar-type distinctions."""
    value = _materialise(cast(dict[str, object], case["input_descriptor"]))
    schema = str(case["schema"])
    if "expected" in case:
        with pytest.raises(ValueError):
            canonical_bytes(schema, value)
    else:
        assert canonical_bytes(schema, value).hex() == case["expected_canonical_utf8_hex"]
        assert canonical_digest(schema, value) == case["expected_sha256"]


def test_repeated_alias_is_not_a_cycle() -> None:
    """Repeated immutable content remains valid while ancestor cycles refuse."""
    child: list[object] = [1]
    assert canonical_bytes("example.v1", [child, child]) == canonical_bytes(
        "example.v1", [[1], [1]]
    )
    child.append(child)
    with pytest.raises(ValueError, match="cycle"):
        canonical_bytes("example.v1", child)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, b"bytes", {1: "bad"}, object()])
def test_unsupported_values_fail_closed(value: object) -> None:
    """Reject unsupported objects instead of stringifying or coercing them."""
    with pytest.raises(ValueError):
        canonical_bytes("example.v1", value)


@pytest.mark.parametrize("schema", ["", "contains\nnewline", "contains\rreturn", "\ud800"])
def test_invalid_schema_prefix(schema: str) -> None:
    """Reject ambiguous prefixes and invalid Unicode before hashing."""
    with pytest.raises(ValueError):
        canonical_bytes(schema, None)


def test_depth_and_domain_separation() -> None:
    """Bound recursion and distinguish otherwise identical schema bodies."""
    value: object = 1
    for _ in range(65):
        value = [value]
    with pytest.raises(ValueError, match="depth"):
        canonical_bytes("example.v1", value)
    assert canonical_digest("one.v1", {}) != canonical_digest("two.v1", {})
    assert canonical_bytes("example.v1", {"\ue000": 1, "😀": 2}).startswith(b"example.v1\n")


def test_invalid_object_key_scalar() -> None:
    """Validate Unicode keys before attempting their UTF-8 ordering."""
    with pytest.raises(ValueError, match="Unicode"):
        canonical_bytes("example.v1", {"\udfff": None})


@pytest.mark.parametrize("value", [10**4096, -(10**4096), 1 << 14000])
def test_oversized_integer_refused(value: int) -> None:
    """Refuse integers outside the shared scalar resource budget."""
    with pytest.raises(ValueError, match="integer scalar too large"):
        canonical_bytes("example.v1", value)


def test_integer_capacity_boundary() -> None:
    """Retain all decimal digits at the maximum supported scalar width."""
    decimal = "9" * 4096
    assert (
        canonical_bytes("example.v1", int(decimal))
        == ('example.v1\n["integer","' + decimal + '"]').encode()
    )
