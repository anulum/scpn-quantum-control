# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — compare claimed values only under compatible assumptions
"""Compare claimed values only when what they measure, and how, is the same.

Two values are comparable only when their keys agree on every field of
:data:`KEY_FIELDS`: estimand, unit, model, dataset, calibration, noise and the
uncertainty method. Equal numbers with different estimands or units are
*non-comparable*, never a contradiction. A missing assumption makes the
comparison *insufficient evidence*. Values of different producers or producer
versions are compared only when the external producer and version have an
accepted equivalence (the federated gate); local comparisons need none.

A comparable pair is judged by a declared criterion only:

- ``interval_overlap``: agreement when both intervals overlap, otherwise
  disagreement;
- ``same_sign``: agreement when both intervals exclude zero on the same side,
  disagreement when on opposite sides, insufficient evidence when either
  interval includes zero.

Nothing here promotes a claim or a certificate. Filters never hide
counter-evidence: a comparison that involves a negative value, or that
disagrees, stays listed beside whatever a filter selects.

Each value cites the committed JSON it comes from: the file, its sha256 and a
pointer to the field. :func:`verify_value_sources` requires the value to equal
that field exactly, so no value can be typed in.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from ..studio_workspace.canonical import canonical_digest

KEY_FIELDS: tuple[str, ...] = (
    "estimand",
    "unit",
    "model",
    "dataset",
    "calibration",
    "noise",
    "uq_method",
)
"""Fields two values must share to be comparable."""

ComparisonStatus = Literal["agreement", "disagreement", "non_comparable", "insufficient_evidence"]
Criterion = Literal["interval_overlap", "same_sign"]
CRITERIA: tuple[Criterion, ...] = ("interval_overlap", "same_sign")
PRODUCER_EQUIVALENCE_NOT_ACCEPTED = "producer_equivalence_not_accepted"
"""Reason of a federated comparison without accepted producer equivalence."""

VALUES_SCHEMA = "scpn_qc_studio_comparable_values_v1"
"""Schema of a declared set of values and the pairs to compare."""
RECORD_SCHEMA = "scpn_qc_studio_claim_comparisons_v1"
"""Schema of a sealed comparison record."""
RECORD_DIGEST_SCHEMA = "scpn_qc_studio_claim_comparisons_record_v1"
"""Domain of a comparison record's canonical digest."""

_HEX64 = re.compile(r"[0-9a-f]{64}")


class ComparabilityError(ValueError):
    """A value, a pair or a cited source cannot be admitted.

    Parameters
    ----------
    message
        What was refused and why.

    """


@dataclass(frozen=True, slots=True)
class ComparableValue:
    """A claimed value with everything that decides what it can be compared with.

    Parameters
    ----------
    value_id
        Stable identifier.
    producer
        The producing studio or package.
    producer_version
        Its exact version.
    key
        ``(field, value)`` for each of :data:`KEY_FIELDS`; ``None`` marks a missing assumption.
    estimate
        The point estimate, or ``None``.
    interval
        ``(low, high)``, or ``None``.
    negative
        Whether the value is counter-evidence, such as a failed replication.
    source
        ``path``, ``sha256`` and the pointers of the estimate and interval bounds.

    """

    value_id: str
    producer: str
    producer_version: str
    key: tuple[tuple[str, str | None], ...]
    estimate: float | None
    interval: tuple[float, float] | None
    negative: bool
    source: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    """The judgement of one declared pair.

    Parameters
    ----------
    left
        Identifier of the first value.
    right
        Identifier of the second value.
    criterion
        The declared criterion.
    status
        ``agreement``, ``disagreement``, ``non_comparable`` or ``insufficient_evidence``.
    reasons
        Why, field by field.
    federated
        Whether the values come from different producers or versions.
    negative
        Whether either value is counter-evidence.

    """

    left: str
    right: str
    criterion: Criterion
    status: ComparisonStatus
    reasons: tuple[str, ...]
    federated: bool
    negative: bool


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparabilityError(f"{path} must be non-empty text")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ComparabilityError(f"{path} must be a finite number")
    return float(value)


def parse_comparable_value(payload: object, path: str = "$") -> ComparableValue:
    """Read one comparable value from its JSON form.

    Parameters
    ----------
    payload
        ``{value_id, producer, producer_version, key, estimate, interval, negative, source}``.
    path
        Address used in refusals.

    Returns
    -------
    ComparableValue
        The value.

    Raises
    ------
    ComparabilityError
        A field is missing, of the wrong type, or the interval is reversed.

    """
    if not isinstance(payload, Mapping):
        raise ComparabilityError(f"{path} must be an object")
    key = payload.get("key")
    if not isinstance(key, Mapping) or set(key) - set(KEY_FIELDS):
        raise ComparabilityError(f"{path}.key must be an object of {', '.join(KEY_FIELDS)}")
    fields: list[tuple[str, str | None]] = []
    for name in KEY_FIELDS:
        value = key.get(name)
        fields.append((name, None if value is None else _text(value, f"{path}.key.{name}")))
    estimate = payload.get("estimate")
    interval = payload.get("interval")
    bounds: tuple[float, float] | None = None
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ComparabilityError(f"{path}.interval must be [low, high]")
        bounds = (
            _number(interval[0], f"{path}.interval[0]"),
            _number(interval[1], f"{path}.interval[1]"),
        )
        if bounds[0] > bounds[1]:
            raise ComparabilityError(f"{path}.interval must not be reversed")
    negative = payload.get("negative")
    if not isinstance(negative, bool):
        raise ComparabilityError(f"{path}.negative must be a boolean")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ComparabilityError(f"{path}.source must be an object")
    return ComparableValue(
        value_id=_text(payload.get("value_id"), f"{path}.value_id"),
        producer=_text(payload.get("producer"), f"{path}.producer"),
        producer_version=_text(payload.get("producer_version"), f"{path}.producer_version"),
        key=tuple(fields),
        estimate=None if estimate is None else _number(estimate, f"{path}.estimate"),
        interval=bounds,
        negative=negative,
        source=tuple(
            sorted(
                (str(name), _text(value, f"{path}.source.{name}"))
                for name, value in source.items()
            )
        ),
    )


def _resolve(document: object, pointer: str) -> object:
    current = document
    for part in pointer.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ComparabilityError(f"pointer {pointer} does not resolve")
    return current


def verify_value_sources(values: Iterable[ComparableValue], root: Path) -> None:
    """Require every value to equal the committed field it cites.

    A value's source names ``path`` and ``sha256``, and the pointers
    ``estimate``, ``low`` and ``high`` for the numbers it carries. A dotted
    pointer walks object keys and list indices.

    Parameters
    ----------
    values
        The values to verify.
    root
        Repository root the paths are relative to.

    Raises
    ------
    ComparabilityError
        A path is unsafe or absent, the digest differs, a pointer does not
        resolve, a number has no pointer, or a cited field differs from the value.

    """
    for value in values:
        source = dict(value.source)
        path = source.get("path", "")
        where = f"{value.value_id}: {path}"
        if not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ComparabilityError(f"{value.value_id}: source path must be a safe relative path")
        if not _HEX64.fullmatch(source.get("sha256", "")):
            raise ComparabilityError(f"{value.value_id}: source sha256 must be a lowercase digest")
        try:
            data = (root / path).read_bytes()
        except OSError as error:
            raise ComparabilityError(f"{where} cannot be read") from error
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ComparabilityError(f"{where} does not match its recorded sha256")
        try:
            document = json.loads(data.decode("utf-8"))
        except ValueError as error:
            raise ComparabilityError(f"{where} is not JSON") from error
        claimed: dict[str, float | None] = {
            "estimate": value.estimate,
            "low": None if value.interval is None else value.interval[0],
            "high": None if value.interval is None else value.interval[1],
        }
        for name, number in claimed.items():
            if number is None:
                continue
            pointer = source.get(name)
            if pointer is None:
                raise ComparabilityError(f"{where}: {name} has no source pointer")
            if _resolve(document, pointer) != number:
                raise ComparabilityError(f"{where}: {name} differs from {pointer}")


def _sign(interval: tuple[float, float]) -> int:
    low, high = interval
    return 1 if low > 0 else -1 if high < 0 else 0


_Verdict = tuple[ComparisonStatus, tuple[str, ...]]


def _assumption_verdict(
    left: ComparableValue,
    right: ComparableValue,
    sides: set[tuple[str, str]],
    home: tuple[str, str],
    accepted: frozenset[tuple[str, str]],
) -> _Verdict | None:
    missing = tuple(
        f"{name} is not stated for {item.value_id}"
        for item in (left, right)
        for name, value in item.key
        if value is None
    )
    if missing:
        return "insufficient_evidence", missing
    differing = tuple(
        f"{name} differs: {a} / {b}"
        for (name, a), (_, b) in zip(left.key, right.key, strict=True)
        if a != b
    )
    if differing:
        return "non_comparable", differing
    unaccepted = sorted(side for side in sides if side != home and side not in accepted)
    if len(sides) > 1 and unaccepted:
        return "non_comparable", tuple(
            f"{PRODUCER_EQUIVALENCE_NOT_ACCEPTED}: {producer} {version}"
            for producer, version in unaccepted
        )
    return None


def _criterion_verdict(
    left: tuple[float, float] | None, right: tuple[float, float] | None, criterion: Criterion
) -> _Verdict:
    if left is None or right is None:
        return "insufficient_evidence", (f"{criterion} needs an interval for both values",)
    if criterion == "interval_overlap":
        overlap = left[0] <= right[1] and right[0] <= left[1]
        return (
            ("agreement", ("the intervals overlap",))
            if overlap
            else ("disagreement", ("the intervals do not overlap",))
        )
    signs = (_sign(left), _sign(right))
    if 0 in signs:
        return "insufficient_evidence", (
            "an interval includes zero, so its sign is not established",
        )
    if signs[0] == signs[1]:
        return "agreement", ("both intervals exclude zero on the same side",)
    return "disagreement", ("the intervals exclude zero on opposite sides",)


def compare_values(
    left: ComparableValue,
    right: ComparableValue,
    criterion: Criterion,
    *,
    home: tuple[str, str],
    accepted: frozenset[tuple[str, str]] = frozenset(),
) -> Comparison:
    """Judge one pair under a declared criterion, refusing incompatible assumptions.

    Parameters
    ----------
    left
        The first value.
    right
        The second value.
    criterion
        ``interval_overlap`` or ``same_sign``.
    home
        The local producer and version; its own values need no equivalence.
    accepted
        Producers and versions with an accepted equivalence (the federated gate).

    Returns
    -------
    Comparison
        Insufficient evidence for a missing assumption or value; non-comparable for
        a differing key field or an unaccepted external producer; otherwise
        agreement or disagreement under the criterion.

    Raises
    ------
    ComparabilityError
        The criterion is unknown.

    """
    if criterion not in CRITERIA:
        raise ComparabilityError(f"unknown criterion {criterion!r}")
    sides = {(left.producer, left.producer_version), (right.producer, right.producer_version)}
    status, reasons = _assumption_verdict(
        left, right, sides, home, accepted
    ) or _criterion_verdict(left.interval, right.interval, criterion)
    return Comparison(
        left.value_id,
        right.value_id,
        criterion,
        status,
        reasons,
        len(sides) > 1,
        left.negative or right.negative,
    )


def filter_comparisons(
    comparisons: Iterable[Comparison], statuses: Iterable[ComparisonStatus]
) -> tuple[tuple[Comparison, ...], tuple[Comparison, ...]]:
    """Select comparisons by status without hiding counter-evidence.

    Parameters
    ----------
    comparisons
        The judged pairs.
    statuses
        The statuses to show.

    Returns
    -------
    tuple[tuple[Comparison, ...], tuple[Comparison, ...]]
        The selected comparisons, and every other comparison that involves a
        negative value or disagrees, which stays discoverable.

    """
    wanted = set(statuses)
    items = tuple(comparisons)
    shown = tuple(item for item in items if item.status in wanted)
    counter = tuple(
        item
        for item in items
        if item.status not in wanted and (item.negative or item.status == "disagreement")
    )
    return shown, counter


def value_to_wire(value: ComparableValue) -> dict[str, object]:
    """Return the JSON form of a value.

    Parameters
    ----------
    value
        The value.

    Returns
    -------
    dict[str, object]
        The form :func:`parse_comparable_value` reads back.

    """
    return {
        "value_id": value.value_id,
        "producer": value.producer,
        "producer_version": value.producer_version,
        "key": dict(value.key),
        "estimate": value.estimate,
        "interval": None if value.interval is None else list(value.interval),
        "negative": value.negative,
        "source": dict(value.source),
    }


def comparison_to_wire(comparison: Comparison) -> dict[str, object]:
    """Return the JSON form of a comparison.

    Parameters
    ----------
    comparison
        The judged pair.

    Returns
    -------
    dict[str, object]
        Its values, criterion, status, reasons and flags.

    """
    return {
        "left": comparison.left,
        "right": comparison.right,
        "criterion": comparison.criterion,
        "status": comparison.status,
        "reasons": list(comparison.reasons),
        "federated": comparison.federated,
        "negative": comparison.negative,
    }


def _identity(payload: object, path: str) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ComparabilityError(f"{path} must be an object")
    return _text(payload.get("producer"), f"{path}.producer"), _text(
        payload.get("version"), f"{path}.version"
    )


def _criterion(value: object, path: str) -> Criterion:
    for criterion in CRITERIA:
        if value == criterion:
            return criterion
    raise ComparabilityError(f"{path} is not one of {', '.join(CRITERIA)}")


def build_comparison_record(document: object, root: Path) -> dict[str, object]:
    """Verify declared values against their sources, compare the declared pairs and seal the result.

    Parameters
    ----------
    document
        ``{schema, home, accepted, values, pairs}`` under :data:`VALUES_SCHEMA`; each pair
        is ``{left, right, criterion}``.
    root
        Repository root the cited sources are relative to.

    Returns
    -------
    dict[str, object]
        The record under :data:`RECORD_SCHEMA`, ending with its canonical ``record_digest``.

    Raises
    ------
    ComparabilityError
        The document is malformed, a value differs from its source, an identifier
        repeats, or a pair names an unknown value or criterion.

    """
    if not isinstance(document, Mapping) or document.get("schema") != VALUES_SCHEMA:
        raise ComparabilityError(f"the declaration must be a {VALUES_SCHEMA} object")
    home = _identity(document.get("home"), "$.home")
    accepted_items = document.get("accepted", [])
    values_items = document.get("values")
    pairs = document.get("pairs")
    if not (
        isinstance(accepted_items, list)
        and isinstance(values_items, list)
        and isinstance(pairs, list)
    ):
        raise ComparabilityError("$.accepted, $.values and $.pairs must be arrays")
    accepted = frozenset(
        _identity(item, f"$.accepted[{index}]") for index, item in enumerate(accepted_items)
    )
    values = [
        parse_comparable_value(item, f"$.values[{index}]")
        for index, item in enumerate(values_items)
    ]
    by_id = {item.value_id: item for item in values}
    if len(by_id) != len(values):
        raise ComparabilityError("$.values repeats a value_id")
    verify_value_sources(values, root)
    comparisons: list[Comparison] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ComparabilityError(f"$.pairs[{index}] must be an object")
        names = [_text(pair.get(side), f"$.pairs[{index}].{side}") for side in ("left", "right")]
        missing = [name for name in names if name not in by_id]
        if missing:
            raise ComparabilityError(
                f"$.pairs[{index}] names unknown values: {', '.join(missing)}"
            )
        criterion = _criterion(pair.get("criterion"), f"$.pairs[{index}].criterion")
        comparisons.append(
            compare_values(
                by_id[names[0]], by_id[names[1]], criterion, home=home, accepted=accepted
            )
        )
    record: dict[str, object] = {
        "schema": RECORD_SCHEMA,
        "home": {"producer": home[0], "version": home[1]},
        "accepted": [
            {"producer": producer, "version": version} for producer, version in sorted(accepted)
        ],
        "values": [value_to_wire(item) for item in values],
        "comparisons": [comparison_to_wire(item) for item in comparisons],
    }
    record["record_digest"] = canonical_digest(RECORD_DIGEST_SCHEMA, record)
    return record


__all__ = [
    "CRITERIA",
    "KEY_FIELDS",
    "PRODUCER_EQUIVALENCE_NOT_ACCEPTED",
    "RECORD_DIGEST_SCHEMA",
    "RECORD_SCHEMA",
    "VALUES_SCHEMA",
    "ComparabilityError",
    "ComparableValue",
    "Comparison",
    "ComparisonStatus",
    "Criterion",
    "build_comparison_record",
    "compare_values",
    "comparison_to_wire",
    "filter_comparisons",
    "parse_comparable_value",
    "value_to_wire",
    "verify_value_sources",
]
