# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — abstract effect values and container storage
"""Carry storage provenance through abstract values and shared container state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

_UNKNOWN = object()


@dataclass(frozen=True, slots=True)
class _Value:
    value: object = _UNKNOWN
    captured: bool = False
    active: bool = False
    local: bool = False
    elements: tuple[_Value, ...] | None = None
    keywords: tuple[tuple[object, _Value], ...] | None = None
    variadic: _Value | None = None
    storage_origins: tuple[object, ...] = ()
    container_kind: Literal["list", "tuple", "dict", "generator"] | None = None

    def __post_init__(self) -> None:
        """Give locally owned container metadata a stable abstract storage identity."""
        if (
            self.local
            and not self.storage_origins
            and (self.elements is not None or self.keywords is not None)
        ):
            object.__setattr__(self, "storage_origins", (object(),))


def _is_static_mapping_key(value: object) -> bool:
    """Admit primitive keys whose hashing and equality cannot call user code."""
    return type(value) in (str, int, bool, float, complex, bytes, type(None))


def _merge_values(values: list[_Value]) -> _Value:
    """Join callable identity and element storage origins across possible control paths."""
    if not values:
        return _Value()
    first = values[0]
    identity = first.value if all(value.value is first.value for value in values) else _UNKNOWN
    element_groups = [value.elements for value in values if value.elements is not None]
    elements = None
    variadic_values = [value.variadic for value in values if value.variadic is not None]
    if len(element_groups) == len(values):
        if all(len(group) == len(element_groups[0]) for group in element_groups):
            elements = tuple(
                _merge_values(list(column)) for column in zip(*element_groups, strict=True)
            )
        else:
            alternatives = [element for group in element_groups for element in group]
            elements = ()
            variadic_values.extend(alternatives)
    elif element_groups:
        elements = ()
        variadic_values.extend(element for group in element_groups for element in group)
        variadic_values.extend(value for value in values if value.elements is None)
    keyword_groups = [dict(value.keywords) for value in values if value.keywords is not None]
    keywords = None
    if len(keyword_groups) == len(values):
        names = dict.fromkeys(name for group in keyword_groups for name in group)
        keywords = tuple(
            (name, _merge_values([group.get(name, _Value()) for group in keyword_groups]))
            for name in names
        )
    origins = {id(origin): origin for value in values for origin in value.storage_origins}
    return _Value(
        identity,
        any(value.captured for value in values),
        any(value.active for value in values),
        all(value.local for value in values),
        elements,
        keywords,
        _merge_values(variadic_values) if variadic_values else None,
        tuple(origins.values()),
        first.container_kind
        if all(value.container_kind == first.container_kind for value in values)
        else None,
    )


def _value_signature(value: _Value) -> tuple[object, ...]:
    """Compare abstract storage bindings without invoking captured equality protocols."""
    return (
        id(value.value),
        value.captured,
        value.active,
        value.local,
        value.container_kind,
        None
        if value.elements is None
        else tuple(_value_signature(item) for item in value.elements),
        None
        if value.keywords is None
        else tuple((name, _value_signature(item)) for name, item in value.keywords),
        None if value.variadic is None else _value_signature(value.variadic),
    )


class _Storage:
    """Resolve and join shared container metadata without evaluating user protocols."""

    def __init__(self) -> None:
        self.storage: dict[int, _Value] = {}

    def resolve(self, value: _Value) -> _Value:
        """Read current container metadata through all possible storage identities."""
        if not value.storage_origins:
            return value
        candidates = []
        for origin in value.storage_origins:
            identity = id(origin)
            if identity not in self.storage:
                self.storage[identity] = replace(value, storage_origins=(origin,))
            candidates.append(self.storage[identity])
        current = candidates[0] if len(candidates) == 1 else _merge_values(candidates)
        if value.captured or not value.local or value.active != current.active:
            current = _merge_values([value, current])
        return current

    def store(self, value: _Value, updated: _Value) -> None:
        """Replace one known container, or weakly update each possible alias target."""
        current = self.resolve(value)
        for origin in current.storage_origins:
            incoming = replace(updated, storage_origins=(origin,))
            if len(current.storage_origins) > 1:
                incoming = _merge_values([self.storage[id(origin)], incoming])
            self.storage[id(origin)] = incoming

    def snapshot(self, value: _Value, storage: dict[int, _Value]) -> _Value:
        """Freeze nested alias projections against one control path's storage state."""
        memo: dict[int, _Value] = {}

        def project(item: _Value) -> _Value:
            """Retain shared references while projecting a finite abstract value graph."""
            if id(item) in memo:
                return memo[id(item)]
            memo[id(item)] = item
            candidates = [storage.get(id(origin), item) for origin in item.storage_origins]
            current = _merge_values(candidates) if candidates else item
            result = replace(
                current,
                elements=None
                if current.elements is None
                else tuple(project(child) for child in current.elements),
                keywords=None
                if current.keywords is None
                else tuple((name, project(child)) for name, child in current.keywords),
                variadic=None if current.variadic is None else project(current.variadic),
            )
            memo[id(item)] = result
            return result

        return project(value)

    def join_storage(self, first: dict[int, _Value], second: dict[int, _Value]) -> None:
        """Join possible heap updates without erasing a branch's captured buffer origin."""
        joined = {}
        for identity in first.keys() | second.keys():
            original = first[identity] if identity in first else second[identity]
            joined[identity] = _merge_values(
                [self.snapshot(original, first), self.snapshot(original, second)]
            )
        self.storage = joined
