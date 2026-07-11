# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Capability Normalization
"""Provider-independent metadata access and normalization primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _optional_attr(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    try:
        return getattr(source, name)
    except Exception:
        return None


def _optional_noarg_call(source: Any, name: str) -> Any:
    candidate = _optional_attr(source, name)
    if not callable(candidate):
        return candidate
    try:
        return candidate()
    except Exception:
        return None


def _first_available_attr(*sources: Any, names: tuple[str, ...]) -> Any:
    for value in _attr_candidates(*sources, names=names):
        if value is not None:
            return value
    return None


def _attr_candidates(*sources: Any, names: tuple[str, ...]) -> list[Any]:
    candidates: list[Any] = []
    for source in sources:
        if source is None:
            continue
        for name in names:
            candidates.append(_optional_attr(source, name))
    return candidates


def _first_text_attr(*sources: Any, names: tuple[str, ...], field_name: str) -> str:
    value = _first_optional_text_attr(*sources, names=names)
    if value is None:
        raise ValueError(f"{field_name} must be provided by provider metadata")
    return value


def _first_optional_text_attr(*sources: Any, names: tuple[str, ...]) -> str | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_positive_int_attr(*sources: Any, names: tuple[str, ...], field_name: str) -> int:
    value = _first_optional_int_attr(*sources, names=names)
    if value is None:
        raise ValueError(f"{field_name} must be provided by provider metadata")
    return value


def _first_optional_int_attr(
    *sources: Any,
    names: tuple[str, ...],
    minimum: int = 1,
) -> int | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= minimum:
            return value
    return None


def _first_bool_attr(*sources: Any, names: tuple[str, ...]) -> bool | None:
    for value in _attr_candidates(*sources, names=names):
        if isinstance(value, bool):
            return value
    return None


def _first_online_attr(*sources: Any) -> bool | None:
    for value in _attr_candidates(
        *sources,
        names=("online", "is_online", "available", "status"),
    ):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"online", "available", "active", "ready", "operational"}:
                return True
            if normalized in {
                "offline",
                "unavailable",
                "inactive",
                "retired",
                "maintenance",
            }:
                return False
    return None


def _online_state_from_text(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"online", "available", "active", "ready", "operational"}:
        return True
    if normalized in {
        "offline",
        "unavailable",
        "inactive",
        "retired",
        "maintenance",
    }:
        return False
    return None


def _declared_ir_formats(
    *sources: Any, names: tuple[str, ...], field_name: str
) -> tuple[str, ...]:
    formats = _first_string_tuple_attr(*sources, names=names)
    if not formats:
        raise ValueError(f"{field_name} must declare supported IR formats")
    return formats


def _first_string_tuple_attr(*sources: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    for value in _attr_candidates(*sources, names=names):
        items = _string_tuple_from_value(value)
        if items:
            return items
    return ()


def _string_tuple_from_value(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, str):
        return (value.strip(),) if isinstance(value, str) and value.strip() else ()
    if isinstance(value, Mapping):
        return _string_tuple_from_value(value.keys())
    try:
        iterator = iter(value)
    except TypeError:
        return ()
    items: list[str] = []
    for item in iterator:
        normalized: str | None
        if isinstance(item, str) and item.strip():
            normalized = item.strip()
        else:
            normalized = _program_spec_name(item)
        if normalized and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _program_spec_name(value: Any) -> str | None:
    for name in ("alias", "program_type", "package", "name", "__name__"):
        candidate = _optional_attr(value, name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


__all__ = [
    "_optional_attr",
    "_optional_noarg_call",
    "_first_available_attr",
    "_attr_candidates",
    "_first_text_attr",
    "_first_optional_text_attr",
    "_first_positive_int_attr",
    "_first_optional_int_attr",
    "_first_bool_attr",
    "_first_online_attr",
    "_online_state_from_text",
    "_declared_ir_formats",
    "_first_string_tuple_attr",
    "_string_tuple_from_value",
    "_program_spec_name",
]
