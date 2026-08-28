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

PROTECTED_WORKFLOW_MAPPING_KEYS = frozenset(
    {
        "jobs",
        "security",
        "steps",
        "run",
        "if",
        "continue-on-error",
        "shell",
        "defaults",
        "<<",
    }
)


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


def _node_source(node: object, source: str) -> tuple[str, int, int]:
    """Return one composed node's validated source span."""
    start = _mark_index(_attribute(node, "start_mark"))
    end = _mark_index(_attribute(node, "end_mark"))
    if end < start or end > len(source):
        raise ValueError("composed YAML scalar has invalid source bounds")
    return source[start:end], start, end


def _scalar_key_contains_escape(node: object, source: str) -> bool:
    """Return whether one double-quoted scalar key uses a YAML escape."""
    if _node_kind(node) != "scalar" or _attribute(node, "style") != '"':
        return False
    spelling, _, _ = _node_source(node, source)
    return "\\" in spelling


def _scalar_key_value(node: object) -> str:
    """Return one validated composed scalar value."""
    value = _attribute(node, "value")
    if not isinstance(value, str):
        raise ValueError("composed YAML scalar has no string value")
    return value


def _uses_canonical_block_key_syntax(node: object, source: str, value: str) -> bool:
    """Return whether a protected scalar key uses plain block syntax."""
    spelling, start, end = _node_source(node, source)
    style = _attribute(node, "style")
    if style == '"' and "\\" in spelling:
        return True
    if style is not None or spelling != value:
        return False

    line_start = source.rfind("\n", 0, start) + 1
    prefix = source[line_start:start]
    if prefix.strip(" \t") not in {"", "-"}:
        return False

    line_end = source.find("\n", end)
    if line_end < 0:
        line_end = len(source)
    suffix = source[end:line_end].lstrip(" \t")
    return suffix.startswith(":")


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


def _collect_unsafe_unescaped_protected_keys(
    node: object,
    source: str,
    protected_keys: frozenset[str],
    visited: set[int],
    found: dict[str, None],
) -> None:
    """Collect noncanonical protected keys and every YAML merge key."""
    identity = id(node)
    if identity in visited:
        return
    visited.add(identity)

    kind = _node_kind(node)
    value = _attribute(node, "value")
    if kind == "scalar":
        return
    if kind == "sequence":
        if not isinstance(value, list):
            raise ValueError("composed YAML sequence has invalid children")
        for child in cast(list[object], value):
            _collect_unsafe_unescaped_protected_keys(
                child,
                source,
                protected_keys,
                visited,
                found,
            )
        return
    if kind != "mapping":
        raise ValueError(f"unsupported composed YAML node kind: {kind}")
    if not isinstance(value, list):
        raise ValueError("composed YAML mapping has invalid entries")
    for entry in cast(list[object], value):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("composed YAML mapping has an invalid entry")
        key_node, value_node = entry
        if _node_kind(key_node) == "scalar":
            key_value = _scalar_key_value(key_node)
            if key_value in protected_keys and (
                key_value == "<<"
                or not _uses_canonical_block_key_syntax(key_node, source, key_value)
            ):
                found.setdefault(key_value, None)
        _collect_unsafe_unescaped_protected_keys(
            key_node,
            source,
            protected_keys,
            visited,
            found,
        )
        _collect_unsafe_unescaped_protected_keys(
            value_node,
            source,
            protected_keys,
            visited,
            found,
        )


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


def unsafe_unescaped_protected_mapping_keys(
    source: str,
) -> tuple[str, ...]:
    """Return protected mapping keys with unsafe unescaped syntax.

    Parameters
    ----------
    source:
        Complete YAML document text. The duplicate-preserving compose graph is
        traversed recursively, including aliases and complex mapping keys.

    Returns
    -------
    tuple[str, ...]
        Unique unsafe protected values in document order. Ordinary protected
        keys are safe only as plain implicit block keys (``key:`` or
        ``- key:``). YAML merge keys (``<<``) are always unsafe, including
        their plain block spelling. Double-quoted keys containing YAML escapes are omitted because
        :func:`has_escaped_double_quoted_mapping_key` owns that stricter,
        all-key rejection category.

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
        return ()
    found: dict[str, None] = {}
    _collect_unsafe_unescaped_protected_keys(
        root,
        source,
        PROTECTED_WORKFLOW_MAPPING_KEYS,
        set(),
        found,
    )
    return tuple(found)
