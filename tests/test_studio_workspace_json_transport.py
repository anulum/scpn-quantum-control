# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — workspace lossless JSON tests
"""Verify public JSON transport preserves exact canonical scalar identities."""

import json
import math
from pathlib import Path

import pytest

from scpn_quantum_control.studio_workspace.canonical import canonical_bytes, canonical_digest
from scpn_quantum_control.studio_workspace.json_transport import read_json, write_json

_CORPUS = json.loads((Path(__file__).parent / "data/studio_workspace/transport.json").read_text())


@pytest.mark.parametrize("case", _CORPUS["cases"], ids=lambda case: case["id"])
def test_lossless_transport_oracle(case: dict[str, object]) -> None:
    """Read declared tokens and retain their semantic types across serialization."""
    if case["expectation"] == "reject":
        with pytest.raises(ValueError):
            read_json(str(case["input_json"]))
        return
    value = read_json(str(case["input_json"]))
    restored = read_json(write_json(value))
    assert canonical_bytes("example.v1", restored) == canonical_bytes("example.v1", value)
    if "expected_digest" in case:
        assert canonical_digest(str(case["schema"]), value) == case["expected_digest"]
    if case["id"] == "negative_zero_token":
        assert isinstance(value, float) and math.copysign(1.0, value) == -1.0


@pytest.mark.parametrize(
    "text", ["NaN", "Infinity", "-Infinity", "", "true false", "[", '"\\uDC00"']
)
def test_invalid_json(text: str) -> None:
    """Refuse non-JSON extensions, malformed syntax and invalid scalar strings."""
    with pytest.raises(ValueError):
        read_json(text)


def test_depth_and_caller_mutation() -> None:
    """Refuse excess nesting and create independent decoded containers."""
    with pytest.raises(ValueError, match="depth"):
        read_json("[" * 65 + "0" + "]" * 65)
    original = {"nested": [1, -0.0, 1.0, 9007199254740993]}
    text = write_json(original)
    original["nested"].append(2)
    assert write_json(read_json(text)) == text


def test_writing_rejects_unsupported_inputs() -> None:
    """Reject nonfinite and non-JSON values at the public export boundary."""
    for value in (float("nan"), {1: "bad"}, {"bad": "\ud800"}, object()):
        with pytest.raises(ValueError):
            write_json(value)


@pytest.mark.parametrize(
    "text",
    ["[1,]", '{"key":1,}', "{1:2}", '"unterminated', '"bad\\x"', "01", "1e9999", "\u00a0true"],
)
def test_json_grammar_is_not_extended(text: str) -> None:
    """Refuse malformed numbers, delimiters, strings and non-JSON whitespace."""
    with pytest.raises(ValueError):
        read_json(text)


def test_integer_digit_budget_applies_before_conversion() -> None:
    """Keep the exact admitted token and reject its first over-budget digit."""
    token = "9" * 4096
    assert write_json(read_json(token)) == token
    with pytest.raises(ValueError, match="integer scalar too large"):
        read_json(token + "9")


@pytest.mark.parametrize(
    "character,count", [("x", 128 * 1024 * 1024 + 1), ("é", 64 * 1024 * 1024 + 1)]
)
def test_expanded_json_limit_counts_utf8_bytes(character: str, count: int) -> None:
    """Reject oversized text before allocating a parsed workspace graph."""
    with pytest.raises(ValueError, match="byte limit"):
        read_json(character * count)
