# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Normalization Tests
"""Tests for provider-independent metadata normalization primitives."""

from __future__ import annotations

import ast
import inspect

import pytest

import scpn_quantum_control.hardware.provider_capability_discovery as provider_capability_discovery
import scpn_quantum_control.hardware.provider_capability_normalization as normalization


class _Metadata:
    """Object-backed metadata used to exercise optional access."""

    name = "  target  "
    count = 7
    enabled = True
    status = "operational"

    def value(self) -> int:
        """Return deterministic callable metadata."""
        return 11

    def failing_value(self) -> int:
        """Raise to exercise fail-closed optional calls."""
        raise RuntimeError("metadata unavailable")


class _BrokenProperty:
    """Metadata object whose property access fails."""

    @property
    def broken(self) -> str:
        """Raise while reading a provider property."""
        raise RuntimeError("property unavailable")


class _ProgramSpec:
    """Program-spec double exposing an alias field."""

    def __init__(self, alias: str) -> None:
        """Store the declared program alias."""
        self.alias = alias


def test_optional_metadata_access_and_calls_fail_closed() -> None:
    """Normalize mapping/object access without leaking provider exceptions."""
    metadata = _Metadata()

    assert normalization._optional_attr({"name": "mapping"}, "name") == "mapping"
    assert normalization._optional_attr(metadata, "name") == "  target  "
    assert normalization._optional_attr(_BrokenProperty(), "broken") is None
    assert normalization._optional_attr(None, "name") is None
    assert normalization._optional_noarg_call(metadata, "value") == 11
    assert normalization._optional_noarg_call(metadata, "failing_value") is None
    assert normalization._optional_noarg_call(metadata, "count") == 7


def test_attribute_candidates_and_first_available_value_preserve_order() -> None:
    """Select the first present provider metadata value deterministically."""
    first = {"value": None, "fallback": "first"}
    second = {"value": "second"}

    assert normalization._attr_candidates(
        first,
        second,
        names=("value", "fallback"),
    ) == [None, "first", "second", None]
    assert (
        normalization._first_available_attr(
            first,
            second,
            names=("value", "fallback"),
        )
        == "first"
    )


def test_text_integer_and_boolean_selection_is_fail_closed() -> None:
    """Normalize scalar fields while rejecting absent or invalid values."""
    metadata = _Metadata()

    assert normalization._first_optional_text_attr(metadata, names=("name",)) == "target"
    assert (
        normalization._first_text_attr(metadata, names=("name",), field_name="target name")
        == "target"
    )
    assert (
        normalization._first_optional_int_attr(
            {"enabled": True, "count": 0},
            metadata,
            names=("enabled", "count"),
        )
        == 7
    )
    assert (
        normalization._first_optional_int_attr(
            {"count": 0},
            names=("count",),
            minimum=0,
        )
        == 0
    )
    assert (
        normalization._first_positive_int_attr(
            metadata,
            names=("count",),
            field_name="qubit count",
        )
        == 7
    )
    assert normalization._first_bool_attr(metadata, names=("enabled",)) is True

    with pytest.raises(ValueError, match="target name"):
        normalization._first_text_attr({}, names=("name",), field_name="target name")
    with pytest.raises(ValueError, match="qubit count"):
        normalization._first_positive_int_attr(
            {"count": 0},
            names=("count",),
            field_name="qubit count",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("online", True),
        (" READY ", True),
        ("maintenance", False),
        ("retired", False),
        ("unknown", None),
    ],
)
def test_online_state_vocabulary(value: str, expected: bool | None) -> None:
    """Normalize provider status text through the shared vocabulary."""
    assert normalization._online_state_from_text(value) is expected


def test_first_online_state_accepts_booleans_and_status_text() -> None:
    """Select boolean or textual provider availability metadata."""
    assert normalization._first_online_attr({"online": False}) is False
    assert normalization._first_online_attr({"status": "available"}) is True
    assert normalization._first_online_attr({"status": "unknown"}) is None


def test_program_spec_and_string_tuple_normalization_is_stable() -> None:
    """Normalize provider program declarations without duplicates."""
    program = _ProgramSpec("qir.v1")

    assert normalization._program_spec_name(program) == "qir.v1"
    assert normalization._string_tuple_from_value(
        [" openqasm3 ", program, "openqasm3", ""],
    ) == ("openqasm3", "qir.v1")
    assert normalization._string_tuple_from_value(
        {"quil": object(), "qasm": object()},
    ) == ("quil", "qasm")
    assert normalization._string_tuple_from_value(None) == ()


def test_declared_ir_formats_require_non_empty_provider_metadata() -> None:
    """Fail closed when provider metadata declares no executable IR."""
    metadata = {"formats": ["openqasm3", "qir"]}

    assert normalization._first_string_tuple_attr(
        metadata,
        names=("formats",),
    ) == ("openqasm3", "qir")
    assert normalization._declared_ir_formats(
        metadata,
        names=("formats",),
        field_name="provider IR",
    ) == ("openqasm3", "qir")

    with pytest.raises(ValueError, match="provider IR"):
        normalization._declared_ir_formats(
            {},
            names=("formats",),
            field_name="provider IR",
        )


def test_normalization_leaf_has_no_provider_backedge() -> None:
    """Keep normalization independent of provider adapters and SDKs."""
    tree = ast.parse(inspect.getsource(normalization))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any("provider_capability" in module for module in imported_modules)


def test_normalization_helpers_are_exact_discovery_aliases() -> None:
    """Preserve every private helper identity through the discovery facade."""
    for name in normalization.__all__:
        assert getattr(provider_capability_discovery, name) is getattr(normalization, name)


def test_normalization_leaf_excludes_provider_snapshot_adapters() -> None:
    """Keep provider-specific snapshot adapters outside normalization."""
    tree = ast.parse(inspect.getsource(normalization))
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert not any(name.startswith("snapshot_from_") for name in function_names)
