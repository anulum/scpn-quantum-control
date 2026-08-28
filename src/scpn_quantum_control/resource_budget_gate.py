# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — compile & dense resource budget gate
"""Fail-closed compile & dense resource budget product surface.

Productises versioned budget **dimensions** for sparse Pauli/compile construction
and dense Hilbert-space allocations. Composes the low-level guards in
:mod:`compile_budget` and :mod:`dense_budget` so estimates and enforce decisions
use the same formulas (no silent diverging budget math).

Does **not** invent host-RAM enterprise capacity claims or rewrite all
compiler/linalg call sites (full call-site enforcement remains open).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from .compile_budget import (
    DEFAULT_PAULI_BUDGET_CAP_GIB,
    estimate_pauli_operator,
)
from .dense_budget import (
    DEFAULT_DENSE_BUDGET_CAP_GIB,
    GIB,
    estimate_dense_allocation,
)

BudgetFamily = Literal["compile_pauli", "dense_hilbert"]
"""Budget family vocabulary for catalogue rows."""

CheckOutcome = Literal["allowed", "refused"]
"""Structured budget check outcomes."""

RESOURCE_BUDGET_GATE_SCHEMA: Final[str] = "resource_budget_gate.v1"
"""JSON schema identifier for serialised budget payloads."""

RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY: Final[str] = (
    "resource budget gate only; estimates compose compile_budget/dense_budget "
    "formulas with explicit GiB caps; exceed-budget is refused fail-closed; "
    "does not claim production OOM immunity or invent host-RAM green capacity"
)
"""Shared claim boundary for dimensions, estimates, and decisions."""


@dataclass(frozen=True, slots=True)
class BudgetDimension:
    """One versioned resource-budget dimension in the product catalogue.

    Attributes
    ----------
    budget_id
        Stable catalogue identifier.
    family
        ``compile_pauli`` or ``dense_hilbert``.
    summary
        Short description.
    default_max_gib
        Explicit GiB cap used for deterministic product checks (not invent-green
        host discovery as a capacity claim).
    label
        Human-readable allocation label passed to low-level estimators.
    as_of
        Inventory date label.
    claim_boundary
        Non-promotional claim boundary.

    """

    budget_id: str
    family: BudgetFamily
    summary: str
    default_max_gib: float
    label: str
    as_of: str = "2026-07-23"
    claim_boundary: str = RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate budget-dimension invariants."""
        if not self.budget_id or not self.budget_id.strip():
            raise ValueError("budget_id must be non-empty")
        if self.family not in {"compile_pauli", "dense_hilbert"}:
            raise ValueError(f"unknown budget family: {self.family!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.default_max_gib <= 0:
            raise ValueError("default_max_gib must be positive")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this dimension."""
        return {
            "budget_id": self.budget_id,
            "family": self.family,
            "summary": self.summary,
            "default_max_gib": self.default_max_gib,
            "label": self.label,
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ResourceBudgetEstimate:
    """Structured resource estimate against a catalogue budget dimension.

    Attributes
    ----------
    budget_id
        Catalogue dimension used.
    family
        Budget family.
    n_qubits
        Requested qubit/oscillator count.
    bytes_required
        Estimated bytes required.
    budget_bytes
        Active budget in bytes.
    gib_required
        Required GiB.
    budget_gib
        Budget GiB.
    within_budget
        Whether ``bytes_required <= budget_bytes``.
    detail
        Extra estimator fields (term_count, dimension, …).

    """

    budget_id: str
    family: BudgetFamily
    n_qubits: int
    bytes_required: int
    budget_bytes: int
    gib_required: float
    budget_gib: float
    within_budget: bool
    detail: Mapping[str, object]
    claim_boundary: str = RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate estimate invariants."""
        if not self.budget_id or not self.budget_id.strip():
            raise ValueError("budget_id must be non-empty")
        if self.family not in {"compile_pauli", "dense_hilbert"}:
            raise ValueError(f"unknown budget family: {self.family!r}")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if self.bytes_required < 0 or self.budget_bytes <= 0:
            raise ValueError("bytes fields must be non-negative with positive budget")
        if self.within_budget != (self.bytes_required <= self.budget_bytes):
            raise ValueError("within_budget inconsistent with byte fields")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this estimate."""
        return {
            "budget_id": self.budget_id,
            "family": self.family,
            "n_qubits": self.n_qubits,
            "bytes_required": self.bytes_required,
            "budget_bytes": self.budget_bytes,
            "gib_required": self.gib_required,
            "budget_gib": self.budget_gib,
            "within_budget": self.within_budget,
            "detail": dict(self.detail),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ResourceBudgetDecision:
    """Fail-closed check decision for a resource budget request.

    Attributes
    ----------
    budget_id
        Catalogue dimension used.
    outcome
        ``allowed`` or ``refused``.
    allowed
        Whether the request is within budget.
    n_qubits
        Requested qubit count.
    bytes_required
        Estimated requirement.
    budget_bytes
        Active budget.
    reason
        Human-readable decision reason.
    blockers
        Non-empty when refused.

    """

    budget_id: str
    outcome: CheckOutcome
    allowed: bool
    n_qubits: int
    bytes_required: int
    budget_bytes: int
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if not self.budget_id or not self.budget_id.strip():
            raise ValueError("budget_id must be non-empty")
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.outcome != "allowed":
            raise ValueError("allowed decisions must use outcome=allowed")
        if not self.allowed and self.outcome != "refused":
            raise ValueError("refused decisions must use outcome=refused")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this decision."""
        return {
            "budget_id": self.budget_id,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "n_qubits": self.n_qubits,
            "bytes_required": self.bytes_required,
            "budget_bytes": self.budget_bytes,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


def _dim(
    budget_id: str,
    *,
    family: BudgetFamily,
    summary: str,
    default_max_gib: float,
    label: str,
) -> BudgetDimension:
    """Build one catalogue dimension."""
    return BudgetDimension(
        budget_id=budget_id,
        family=family,
        summary=summary,
        default_max_gib=default_max_gib,
        label=label,
    )


_CANONICAL_DIMENSIONS: Final[tuple[BudgetDimension, ...]] = (
    _dim(
        "compile_pauli_default",
        family="compile_pauli",
        summary=(
            "Default sparse Pauli / compile construction budget "
            f"(cap {DEFAULT_PAULI_BUDGET_CAP_GIB} GiB explicit product default)."
        ),
        default_max_gib=float(DEFAULT_PAULI_BUDGET_CAP_GIB),
        label="sparse Pauli operator (product gate)",
    ),
    _dim(
        "dense_hilbert_default",
        family="dense_hilbert",
        summary=(
            "Default dense Hilbert-space allocation budget "
            f"(cap {DEFAULT_DENSE_BUDGET_CAP_GIB} GiB explicit product default)."
        ),
        default_max_gib=float(DEFAULT_DENSE_BUDGET_CAP_GIB),
        label="dense Hilbert-space object (product gate)",
    ),
    _dim(
        "compile_pauli_tight",
        family="compile_pauli",
        summary="Tight compile/Pauli budget for CI-style exceed-budget tests (0.001 GiB).",
        default_max_gib=0.001,
        label="sparse Pauli operator (tight)",
    ),
    _dim(
        "dense_hilbert_tight",
        family="dense_hilbert",
        summary="Tight dense budget for CI-style exceed-budget tests (0.001 GiB).",
        default_max_gib=0.001,
        label="dense Hilbert-space object (tight)",
    ),
)


def _catalogue_map() -> dict[str, BudgetDimension]:
    """Return budget_id → dimension map; refuse blanks/duplicates."""
    mapping: dict[str, BudgetDimension] = {}
    for row in _CANONICAL_DIMENSIONS:
        key = row.budget_id.strip()
        if not key:
            raise RuntimeError("resource budget catalogue contains blank budget_id")
        if key in mapping:
            raise RuntimeError(f"duplicate budget_id in catalogue: {key!r}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError("resource budget catalogue must be non-empty")
    return mapping


_DIMENSION_BY_ID: Final[Mapping[str, BudgetDimension]] = _catalogue_map()


def list_budget_dimension_ids() -> tuple[str, ...]:
    """Return all budget dimension identifiers in catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered budget identifiers.

    """
    return tuple(row.budget_id for row in _CANONICAL_DIMENSIONS)


def get_budget_dimension(budget_id: str) -> BudgetDimension:
    """Return one budget dimension or raise for blank/unknown identifiers.

    Parameters
    ----------
    budget_id
        Catalogue budget key.

    Returns
    -------
    BudgetDimension
        Matching dimension.

    Raises
    ------
    ValueError
        If ``budget_id`` is blank or unknown (fail closed).

    """
    if not budget_id or not str(budget_id).strip():
        raise ValueError("budget_id must be a non-empty string")
    key = str(budget_id).strip()
    try:
        return _DIMENSION_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown budget_id {key!r}; refuse invent-green resource plan "
            f"(known_count={len(_DIMENSION_BY_ID)})"
        ) from exc


def iter_budget_dimensions(
    *,
    family: BudgetFamily | None = None,
) -> tuple[BudgetDimension, ...]:
    """Return filtered budget dimensions in stable order.

    Parameters
    ----------
    family
        Optional family filter.

    Returns
    -------
    tuple[BudgetDimension, ...]
        Matching dimensions.

    """
    rows: Iterable[BudgetDimension] = _CANONICAL_DIMENSIONS
    if family is not None:
        rows = (row for row in rows if row.family == family)
    return tuple(rows)


def estimate_resource_budget(
    budget_id: str,
    *,
    n_qubits: int,
    max_gib: float | None = None,
    include_zz: bool = False,
    dense_rank: int = 2,
    dense_object_count: int = 1,
) -> ResourceBudgetEstimate:
    """Estimate resource usage against a catalogue budget dimension.

    Composes :func:`estimate_pauli_operator` or :func:`estimate_dense_allocation`
    with an explicit GiB cap (catalogue default or override).

    Parameters
    ----------
    budget_id
        Catalogue budget key.
    n_qubits
        Qubit/oscillator count (must be ``>= 1``).
    max_gib
        Optional override GiB cap; defaults to the catalogue dimension default.
    include_zz
        Pauli estimator option (compile family only).
    dense_rank
        Dense rank (dense family only).
    dense_object_count
        Dense object count (dense family only).

    Returns
    -------
    ResourceBudgetEstimate
        Structured estimate with within-budget flag.

    Raises
    ------
    ValueError
        If identifiers / dimensions are invalid.
    TypeError
        If ``n_qubits`` is not an integer (from low-level estimators).

    """
    dimension = get_budget_dimension(budget_id)
    if not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    cap = float(dimension.default_max_gib if max_gib is None else max_gib)
    if cap <= 0:
        raise ValueError("max_gib must be positive")

    if dimension.family == "compile_pauli":
        estimate = estimate_pauli_operator(
            n_qubits,
            include_zz=include_zz,
            max_gib=cap,
            label=dimension.label,
        )
        detail: dict[str, object] = {
            "term_count": estimate.term_count,
            "label_chars": estimate.label_chars,
            "include_zz": include_zz,
            "low_level": "estimate_pauli_operator",
        }
        return ResourceBudgetEstimate(
            budget_id=dimension.budget_id,
            family=dimension.family,
            n_qubits=estimate.n_qubits,
            bytes_required=estimate.bytes_required,
            budget_bytes=estimate.budget_bytes,
            gib_required=estimate.gib_required,
            budget_gib=estimate.budget_gib,
            within_budget=estimate.bytes_required <= estimate.budget_bytes,
            detail=detail,
        )

    dense = estimate_dense_allocation(
        n_qubits,
        rank=dense_rank,
        object_count=dense_object_count,
        max_gib=cap,
        label=dimension.label,
    )
    detail = {
        "dimension": dense.dimension,
        "shape": list(dense.shape),
        "dtype": dense.dtype,
        "object_count": dense.object_count,
        "rank": dense_rank,
        "low_level": "estimate_dense_allocation",
    }
    return ResourceBudgetEstimate(
        budget_id=dimension.budget_id,
        family=dimension.family,
        n_qubits=dense.n_qubits,
        bytes_required=dense.bytes_required,
        budget_bytes=dense.budget_bytes,
        gib_required=dense.gib_required,
        budget_gib=dense.budget_gib,
        within_budget=dense.bytes_required <= dense.budget_bytes,
        detail=detail,
    )


def check_resource_budget(
    budget_id: str,
    *,
    n_qubits: int,
    max_gib: float | None = None,
    include_zz: bool = False,
    dense_rank: int = 2,
    dense_object_count: int = 1,
) -> ResourceBudgetDecision:
    """Check a request against budget and return a structured decision.

    Exceed-budget requests are **refused** with non-empty blockers (never silent
    allow). Within-budget requests are allowed with estimate fields.

    Parameters
    ----------
    budget_id
        Catalogue budget key.
    n_qubits
        Qubit/oscillator count.
    max_gib
        Optional GiB cap override.
    include_zz
        Pauli option.
    dense_rank
        Dense rank.
    dense_object_count
        Dense object count.

    Returns
    -------
    ResourceBudgetDecision
        Allowed or refused decision.

    Raises
    ------
    ValueError
        If identifiers / dimensions are invalid.

    """
    estimate = estimate_resource_budget(
        budget_id,
        n_qubits=n_qubits,
        max_gib=max_gib,
        include_zz=include_zz,
        dense_rank=dense_rank,
        dense_object_count=dense_object_count,
    )
    if estimate.within_budget:
        return ResourceBudgetDecision(
            budget_id=estimate.budget_id,
            outcome="allowed",
            allowed=True,
            n_qubits=estimate.n_qubits,
            bytes_required=estimate.bytes_required,
            budget_bytes=estimate.budget_bytes,
            reason=(
                f"within budget: {estimate.gib_required:.6f} GiB required of "
                f"{estimate.budget_gib:.6f} GiB ({estimate.family})"
            ),
            blockers=(),
        )
    blocker = (
        f"exceeds budget: {estimate.gib_required:.6f} GiB required > "
        f"{estimate.budget_gib:.6f} GiB cap for {estimate.family} "
        f"(n_qubits={estimate.n_qubits})"
    )
    return ResourceBudgetDecision(
        budget_id=estimate.budget_id,
        outcome="refused",
        allowed=False,
        n_qubits=estimate.n_qubits,
        bytes_required=estimate.bytes_required,
        budget_bytes=estimate.budget_bytes,
        reason="resource budget refuse: " + blocker,
        blockers=(blocker,),
    )


def enforce_resource_budget(
    budget_id: str,
    *,
    n_qubits: int,
    max_gib: float | None = None,
    include_zz: bool = False,
    dense_rank: int = 2,
    dense_object_count: int = 1,
) -> ResourceBudgetEstimate:
    """Enforce budget: return estimate if allowed, else raise typed error.

    Raises the same low-level error types as the composed guards
    (:class:`PauliOperatorBudgetError` / :class:`DenseAllocationError`) wrapped
    as :class:`ResourceBudgetExceededError` with structured detail.

    Parameters
    ----------
    budget_id
        Catalogue budget key.
    n_qubits
        Qubit count.
    max_gib
        Optional GiB cap override.
    include_zz
        Pauli option.
    dense_rank
        Dense rank.
    dense_object_count
        Dense object count.

    Returns
    -------
    ResourceBudgetEstimate
        Estimate when within budget.

    Raises
    ------
    ResourceBudgetExceededError
        When the request exceeds the active budget.
    ValueError
        If identifiers / dimensions are invalid.

    """
    decision = check_resource_budget(
        budget_id,
        n_qubits=n_qubits,
        max_gib=max_gib,
        include_zz=include_zz,
        dense_rank=dense_rank,
        dense_object_count=dense_object_count,
    )
    estimate = estimate_resource_budget(
        budget_id,
        n_qubits=n_qubits,
        max_gib=max_gib,
        include_zz=include_zz,
        dense_rank=dense_rank,
        dense_object_count=dense_object_count,
    )
    if decision.allowed:
        return estimate
    raise ResourceBudgetExceededError(
        decision.reason,
        budget_id=decision.budget_id,
        n_qubits=decision.n_qubits,
        bytes_required=decision.bytes_required,
        budget_bytes=decision.budget_bytes,
    )


class ResourceBudgetExceededError(MemoryError):
    """Typed exceed-budget error for the product resource budget gate."""

    def __init__(
        self,
        message: str,
        *,
        budget_id: str,
        n_qubits: int,
        bytes_required: int,
        budget_bytes: int,
    ) -> None:
        """Store structured exceed-budget fields alongside the message."""
        super().__init__(message)
        self.budget_id = budget_id
        self.n_qubits = n_qubits
        self.bytes_required = bytes_required
        self.budget_bytes = budget_bytes

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready error payload."""
        return {
            "error": "ResourceBudgetExceededError",
            "message": str(self),
            "budget_id": self.budget_id,
            "n_qubits": self.n_qubits,
            "bytes_required": self.bytes_required,
            "budget_bytes": self.budget_bytes,
            "claim_boundary": RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY,
        }


def build_resource_budget_registry() -> dict[str, object]:
    """Build the full serialisable resource budget registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every dimension (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_DIMENSIONS]
    compile_count = sum(1 for row in _CANONICAL_DIMENSIONS if row.family == "compile_pauli")
    dense_count = sum(1 for row in _CANONICAL_DIMENSIONS if row.family == "dense_hilbert")
    return {
        "schema": RESOURCE_BUDGET_GATE_SCHEMA,
        "claim_boundary": RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY,
        "dimension_count": len(rows),
        "compile_pauli_count": compile_count,
        "dense_hilbert_count": dense_count,
        "blank_entry_count": 0,
        "gib_constant": GIB,
        "dimensions": rows,
        "policy_note": (
            "Composes compile_budget.estimate_pauli_operator and "
            "dense_budget.estimate_dense_allocation with explicit product GiB caps; "
            "Full compiler/linalg call-site enforcement remains residual."
        ),
    }


def assert_resource_budget_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers dimensions without blanks.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_resource_budget_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage or blanks fail.

    """
    registry = dict(payload) if payload is not None else build_resource_budget_registry()
    dimensions = registry.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("resource budget registry must contain a non-empty dimensions list")
    seen: set[str] = set()
    blank = 0
    families: set[str] = set()
    for index, row in enumerate(dimensions):
        if not isinstance(row, Mapping):
            raise ValueError(f"dimension row {index} must be a mapping")
        budget_id = row.get("budget_id")
        family = row.get("family")
        max_gib = row.get("default_max_gib")
        if not budget_id or not str(budget_id).strip():
            blank += 1
            continue
        bid = str(budget_id).strip()
        if bid in seen:
            raise ValueError(f"duplicate budget_id in registry: {bid!r}")
        seen.add(bid)
        if family not in {"compile_pauli", "dense_hilbert"}:
            blank += 1
            continue
        families.add(str(family))
        if not isinstance(max_gib, (int, float)) or float(max_gib) <= 0:
            raise ValueError(f"budget {bid!r} has invalid default_max_gib")
    if blank:
        raise ValueError(f"resource budget registry has {blank} blank or invalid entries")
    if families != {"compile_pauli", "dense_hilbert"}:
        raise ValueError(f"resource budget registry must include both families (got={families!r})")
    expected = set(list_budget_dimension_ids())
    if seen != expected:
        raise ValueError(
            f"registry budget set drift (missing={expected - seen!r}, extra={seen - expected!r})"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    dimension_count = registry.get("dimension_count", -1)
    if not isinstance(dimension_count, int) or dimension_count != len(dimensions):
        raise ValueError("dimension_count does not match dimensions list length")
    return registry


__all__ = [
    "RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY",
    "RESOURCE_BUDGET_GATE_SCHEMA",
    "BudgetDimension",
    "BudgetFamily",
    "CheckOutcome",
    "ResourceBudgetDecision",
    "ResourceBudgetEstimate",
    "ResourceBudgetExceededError",
    "assert_resource_budget_integrity",
    "build_resource_budget_registry",
    "check_resource_budget",
    "enforce_resource_budget",
    "estimate_resource_budget",
    "get_budget_dimension",
    "iter_budget_dimensions",
    "list_budget_dimension_ids",
]
