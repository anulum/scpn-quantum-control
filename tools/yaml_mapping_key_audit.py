# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — semantic YAML mapping-key spelling audit
"""Inspect composed YAML mapping keys without collapsing duplicate entries."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class _YamlComposer(Protocol):
    """Minimal typed surface required from the PyYAML module."""

    def compose(self, stream: str) -> object | None:
        """Compose one YAML document into its node graph."""


def _attribute(value: object, name: str) -> object:
    """Return one dynamically provided PyYAML node attribute."""
    return cast(object, getattr(value, name, None))


def _node_kind(node: object) -> str:
    """Return a validated PyYAML node kind."""
    kind = _attribute(node, "id")
    if not isinstance(kind, str):
        raise ValueError("composed YAML node has no string id")
    return kind


def _mark_index(mark: object) -> int:
    """Return a validated source offset from a PyYAML mark."""
    index = _attribute(mark, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("composed YAML node has an invalid source mark")
    return index


def _scalar_key_contains_escape(node: object, source: str) -> bool:
    """Return whether one double-quoted scalar key uses a YAML escape."""
    if _node_kind(node) != "scalar" or _attribute(node, "style") != '"':
        return False
    start = _mark_index(_attribute(node, "start_mark"))
    end = _mark_index(_attribute(node, "end_mark"))
    if end < start or end > len(source):
        raise ValueError("composed YAML scalar has invalid source bounds")
    return "\\" in source[start:end]


def _node_has_escaped_mapping_key(
    node: object,
    source: str,
    visited: set[int],
) -> bool:
    """Traverse a composed node graph while tolerating recursive aliases."""
    identity = id(node)
    if identity in visited:
        return False
    visited.add(identity)

    kind = _node_kind(node)
    value = _attribute(node, "value")
    if kind == "scalar":
        return False
    if kind == "sequence":
        if not isinstance(value, list):
            raise ValueError("composed YAML sequence has invalid children")
        return any(
            _node_has_escaped_mapping_key(child, source, visited)
            for child in cast(list[object], value)
        )
    if kind != "mapping":
        raise ValueError(f"unsupported composed YAML node kind: {kind}")
    if not isinstance(value, list):
        raise ValueError("composed YAML mapping has invalid entries")
    for entry in cast(list[object], value):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("composed YAML mapping has an invalid entry")
        key_node, value_node = entry
        if _scalar_key_contains_escape(key_node, source):
            return True
        if _node_has_escaped_mapping_key(key_node, source, visited):
            return True
        if _node_has_escaped_mapping_key(value_node, source, visited):
            return True
    return False


def has_escaped_double_quoted_mapping_key(source: str) -> bool:
    """Return whether YAML uses an escape in any double-quoted mapping key.

    Parameters
    ----------
    source:
        Complete YAML document text. Both block and flow mappings are
        inspected through PyYAML's compose tree, which retains duplicate key
        nodes and raw source marks.

    Returns
    -------
    bool
        Whether any double-quoted mapping-key scalar contains a backslash
        escape. Escaped scalar values remain allowed.

    Raises
    ------
    ValueError
        If PyYAML is unavailable, the document cannot be composed, or the
        returned node graph violates the expected node contract.

    """
    try:
        composer = cast(_YamlComposer, import_module("yaml"))
        root = composer.compose(source)
    except Exception as exc:
        raise ValueError("workflow text is not valid composable YAML") from exc
    if root is None:
        return False
    return _node_has_escaped_mapping_key(root, source, set())
