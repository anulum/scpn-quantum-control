# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — governed multi-ecosystem route matrix
"""Fail-closed multi-ecosystem differentiable route matrix and explain API.

This module is the product surface for route-matrix (governed multi-ecosystem support
matrix). It unifies route identifiers across transform, adapter, compiler, and
Rust families and refuses silent degradation: unknown or blank cells raise or
resolve only to explicit ``permanent_boundary`` / ``implementation_path``
statuses.

The surface is pure and deterministic. It does not execute gradients, contact
providers, or promote performance claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

RouteFamily = Literal[
    "transform",
    "adapter",
    "compiler",
    "rust",
    "provider",
    "competitor_boundary",
]
"""High-level family of a governed route identifier."""

RouteClosureStatus = Literal[
    "supported",
    "permanent_boundary",
    "implementation_path",
]
"""Closure status for one governed route cell."""

GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY: Final[str] = (
    "governed multi-ecosystem route matrix only; supported rows are local "
    "conformance or documented adapter evidence, permanent_boundary and "
    "implementation_path rows are fail-closed and never silently promoted to "
    "provider, hardware, compiler-performance, or category-leadership claims"
)
"""Shared claim boundary attached to every matrix row and explanation."""

GOVERNED_ROUTE_MATRIX_SCHEMA: Final[str] = "governed_route_matrix.v1"
"""JSON schema identifier for serialised matrix payloads."""


@dataclass(frozen=True, slots=True)
class RouteCapability:
    """Capability context supplied when explaining a route decision.

    Attributes
    ----------
    ecosystem
        Caller-declared ecosystem label (for example ``native``, ``jax``,
        ``catalyst``). Empty or whitespace-only values fail closed.
    method
        Requested gradient or transform method, or ``auto``.
    finite_shot
        Whether the caller requires finite-shot planning.
    allow_hardware
        Whether policy-gated hardware routes may be considered.

    """

    ecosystem: str
    method: str = "auto"
    finite_shot: bool = False
    allow_hardware: bool = False

    def __post_init__(self) -> None:
        """Validate capability fields."""
        if not self.ecosystem or not self.ecosystem.strip():
            raise ValueError("RouteCapability.ecosystem must be a non-empty string")
        if not self.method or not self.method.strip():
            raise ValueError("RouteCapability.method must be a non-empty string")
        object.__setattr__(self, "ecosystem", self.ecosystem.strip().lower())
        object.__setattr__(self, "method", self.method.strip().lower())


@dataclass(frozen=True, slots=True)
class GovernedRouteRecord:
    """One machine-checked multi-ecosystem route cell.

    Attributes
    ----------
    route_id
        Stable route identifier (taxonomy key).
    family
        Route family bucket.
    closure_status
        Fail-closed support classification.
    summary
        Short human-readable description.
    evidence
        Evidence labels or artefact pointers (not performance claims).
    rejected_alternatives
        Alternative route IDs rejected for the same capability class.
    closure_reason
        Required free-text reason for non-supported rows; empty for supported.
    claim_boundary
        Non-promotional claim boundary string.

    """

    route_id: str
    family: RouteFamily
    closure_status: RouteClosureStatus
    summary: str
    evidence: tuple[str, ...]
    rejected_alternatives: tuple[str, ...]
    closure_reason: str = ""
    claim_boundary: str = GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate route-record invariants."""
        if not self.route_id or not self.route_id.strip():
            raise ValueError("route_id must be non-empty")
        if self.family not in {
            "transform",
            "adapter",
            "compiler",
            "rust",
            "provider",
            "competitor_boundary",
        }:
            raise ValueError(f"unknown route family: {self.family!r}")
        if self.closure_status not in {
            "supported",
            "permanent_boundary",
            "implementation_path",
        }:
            raise ValueError(f"unknown closure_status: {self.closure_status!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.closure_status == "supported":
            if self.closure_reason:
                raise ValueError("supported routes must not carry a closure_reason")
        elif not self.closure_reason or not self.closure_reason.strip():
            raise ValueError(
                "non-supported routes require a non-empty closure_reason "
                f"(route_id={self.route_id!r}, status={self.closure_status!r})"
            )
        if any(not item or not item.strip() for item in self.evidence):
            raise ValueError("evidence labels must be non-empty strings")
        if any(not item or not item.strip() for item in self.rejected_alternatives):
            raise ValueError("rejected_alternatives must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this route record."""
        return {
            "route_id": self.route_id,
            "family": self.family,
            "closure_status": self.closure_status,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "rejected_alternatives": list(self.rejected_alternatives),
            "closure_reason": self.closure_reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    """Deterministic explanation for a route selection under a capability.

    Attributes
    ----------
    route_id
        Requested route identifier.
    capability
        Capability context used for the decision.
    selected
        The resolved route record (never invents unsupported green cells).
    rejected
        Alternative routes considered and rejected with reasons.
    notes
        Additional deterministic notes for operators.

    """

    route_id: str
    capability: RouteCapability
    selected: GovernedRouteRecord
    rejected: tuple[GovernedRouteRecord, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate explanation invariants."""
        if not self.route_id or not self.route_id.strip():
            raise ValueError("route_id must be non-empty")
        selected_id = self.selected.route_id
        allowed_unknown = selected_id.startswith("unknown:")
        if selected_id != self.route_id and not allowed_unknown:
            raise ValueError("selected route_id must match the request or an unknown:* boundary")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this explanation."""
        return {
            "route_id": self.route_id,
            "capability": {
                "ecosystem": self.capability.ecosystem,
                "method": self.capability.method,
                "finite_shot": self.capability.finite_shot,
                "allow_hardware": self.capability.allow_hardware,
            },
            "selected": self.selected.to_dict(),
            "rejected": [row.to_dict() for row in self.rejected],
            "notes": list(self.notes),
            "claim_boundary": GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY,
        }


def _route(
    route_id: str,
    family: RouteFamily,
    closure_status: RouteClosureStatus,
    summary: str,
    *,
    evidence: Sequence[str],
    rejected: Sequence[str] = (),
    closure_reason: str = "",
) -> GovernedRouteRecord:
    """Build one validated catalogue row."""
    return GovernedRouteRecord(
        route_id=route_id,
        family=family,
        closure_status=closure_status,
        summary=summary,
        evidence=tuple(evidence),
        rejected_alternatives=tuple(rejected),
        closure_reason=closure_reason,
    )


# Canonical catalogue: every cell has an explicit closure status (no blanks).
_CANONICAL_ROUTES: Final[tuple[GovernedRouteRecord, ...]] = (
    _route(
        "transform:native.grad_vmap",
        "transform",
        "supported",
        "Native grad(vmap) composition under local transform-algebra audit.",
        evidence=("transform_algebra_audit", "support_matrix_generation"),
        rejected=("transform:unsupported.complex_objective",),
    ),
    _route(
        "transform:native.vmap_grad",
        "transform",
        "supported",
        "Native vmap(grad) composition under local transform-algebra audit.",
        evidence=("transform_algebra_audit", "support_matrix_generation"),
    ),
    _route(
        "transform:unsupported.complex_objective",
        "transform",
        "permanent_boundary",
        "Complex-valued objectives without an explicit Wirtinger contract.",
        evidence=("transform_algebra_unsupported_boundary",),
        closure_reason=(
            "complex-valued objectives require an explicit Wirtinger contract; "
            "silent reverse-mode is refused"
        ),
    ),
    _route(
        "transform:ssgf.latent_finite_difference",
        "transform",
        "supported",
        "Central finite difference on the bounded SSGF softplus latent geometry map.",
        evidence=(
            "geometry_gradient_cost_certificate",
            "geometry_gradient_metamorphic_certificate",
        ),
        rejected=("transform:ssgf.latent_parameter_shift",),
    ),
    _route(
        "transform:ssgf.latent_parameter_shift",
        "transform",
        "permanent_boundary",
        "Circuit parameter-shift applied directly to the SSGF latent z vector.",
        evidence=("nonlinear_latent_map_boundary", "anti_silent_wrong_registry"),
        closure_reason=(
            "z enters Hamiltonian coefficients through the nonlinear softplus W(z) map; "
            "a circuit-angle parameter-shift is not directly dC/dz, so the supported "
            "bounded route remains central finite difference"
        ),
    ),
    _route(
        "adapter:jax.value_and_grad_local",
        "adapter",
        "supported",
        "Bounded local JAX value_and_grad / transform routes on Phase-QNode.",
        evidence=("external_comparison_closure", "jax_bridge"),
        rejected=("adapter:jax.provider_arbitrary_simulator",),
    ),
    _route(
        "adapter:jax.provider_arbitrary_simulator",
        "adapter",
        "implementation_path",
        "Arbitrary provider/native simulator autodiff-through-simulator kernels.",
        evidence=("external_comparison_hard_gap",),
        closure_reason=(
            "provider/native arbitrary simulator autodiff is not implemented; "
            "local statevector routes remain the supported path"
        ),
    ),
    _route(
        "adapter:torch.func_local",
        "adapter",
        "supported",
        "Bounded local torch.func / non-fullgraph compile routes.",
        evidence=("external_comparison_closure", "torch_bridge"),
        rejected=("adapter:torch.fullgraph_compile",),
    ),
    _route(
        "adapter:torch.fullgraph_compile",
        "adapter",
        "implementation_path",
        "Registered fullgraph torch.compile lowering for Phase-QNode.",
        evidence=("torch_maturity_boundary",),
        closure_reason=(
            "fullgraph torch.compile lowering is not registered; non-fullgraph "
            "local routes remain supported"
        ),
    ),
    _route(
        "adapter:pennylane.local_default_qubit",
        "adapter",
        "supported",
        "Bounded local PennyLane default.qubit parity / import routes.",
        evidence=("pennylane_import", "external_comparison_closure"),
        rejected=("adapter:pennylane.hardware_plugin_gradient",),
    ),
    _route(
        "adapter:pennylane.hardware_plugin_gradient",
        "adapter",
        "permanent_boundary",
        "Hardware-plugin gradient execution through PennyLane devices.",
        evidence=("provider_hardware_safety_audit",),
        closure_reason=(
            "hardware-plugin gradients require owner-ticketed evidence chains "
            "and never silently plan as local supported routes"
        ),
    ),
    _route(
        "adapter:l16.local_indicator",
        "adapter",
        "supported",
        "Bounded local exact-simulator L16 indicator evaluation and heuristic safety routing.",
        evidence=("bounded_director_functional_evidence", "bounded_director_codesign_interlock"),
        rejected=("adapter:l16.autonomous_hardware_control",),
    ),
    _route(
        "adapter:l16.autonomous_hardware_control",
        "adapter",
        "permanent_boundary",
        "Autonomous hardware or plant actuation from the L16 weighted heuristic.",
        evidence=(
            "bounded_director_claim_boundary",
            "codesign_safety_policy",
            "control_stack_execution_policy",
        ),
        closure_reason=(
            "weighted indicator composite is not a Lyapunov, PCS, or stability certificate; "
            "owner-ticketed hardware and partner control validation cannot be inferred"
        ),
    ),
    _route(
        "compiler:mlir_enzyme.bounded_kernels",
        "compiler",
        "supported",
        "Bounded MLIR/Enzyme scalar/vector/matrix kernel evidence.",
        evidence=("mlir_enzyme_evidence", "compiler_promotion_batch"),
        rejected=("compiler:catalyst.qjit_vmap",),
    ),
    _route(
        "compiler:catalyst.qjit_vmap",
        "compiler",
        "permanent_boundary",
        "Catalyst qjit + jax.vmap over quantum instructions.",
        evidence=("catalyst_sharp_bits", "competitive_baseline"),
        closure_reason=(
            "Catalyst documents missing batching rules for quantum instructions; "
            "vmap-inside-qjit is a permanent competitor boundary, not a silent gap"
        ),
    ),
    _route(
        "rust:program_ad.static_registry_replay",
        "rust",
        "supported",
        "Bounded Rust Program AD static registry replay (parity-gated families).",
        evidence=("rust_program_ad_parity", "registry_metadata_mirror"),
        rejected=("rust:program_ad.dynamic_axes",),
    ),
    _route(
        "rust:program_ad.dynamic_axes",
        "rust",
        "permanent_boundary",
        "Dynamic axes / dynamic indexing in Rust Program AD replay.",
        evidence=("dynamic_boundary_fail_closed_audit",),
        closure_reason=(
            "dynamic axes and indexing remain typed fail-closed; they are not "
            "silently approximated by static replay"
        ),
    ),
    _route(
        "provider:hardware.gradient_live",
        "provider",
        "permanent_boundary",
        "Live hardware gradient submission.",
        evidence=("no_submit_default", "provider_hardware_safety_audit"),
        closure_reason=(
            "live hardware gradients are owner-ticket gated and never selected "
            "by the default matrix without allow_hardware policy evidence"
        ),
    ),
    _route(
        "competitor:differentiation_interface.silent_wrong_grads",
        "competitor_boundary",
        "permanent_boundary",
        "DifferentiationInterface.jl ReverseDiff compiled-tape silent wrong gradients.",
        evidence=("plan_sota_addendum_8_1", "citation:DifferentiationInterface.jl"),
        closure_reason=(
            "documented competitor failure mode: value-dependent control flow may "
            "yield silently wrong reverse-mode results under compiled tapes; "
            "SCPN refuses silent degradation for the same class"
        ),
    ),
    _route(
        "competitor:catalyst.no_broadcast_adaptive_shots",
        "competitor_boundary",
        "permanent_boundary",
        "Catalyst boundary for adaptive finite-shot trainability without broadcast/vmap.",
        evidence=("finite_shot_trainability_boundary", "catalyst_comparison"),
        closure_reason=(
            "Catalyst comparison rows document no-broadcast/no-vmap limitations "
            "for adaptive finite-shot trainability dry-runs"
        ),
    ),
)


def _catalogue_map() -> dict[str, GovernedRouteRecord]:
    """Return the route_id → record map for the canonical catalogue."""
    mapping = {row.route_id: row for row in _CANONICAL_ROUTES}
    if len(mapping) != len(_CANONICAL_ROUTES):
        raise RuntimeError("duplicate route_id in governed route catalogue")
    return mapping


_ROUTE_BY_ID: Final[Mapping[str, GovernedRouteRecord]] = _catalogue_map()


def list_governed_route_ids() -> tuple[str, ...]:
    """Return all canonical route identifiers in stable catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered route identifiers from the fail-closed catalogue.

    """
    return tuple(row.route_id for row in _CANONICAL_ROUTES)


def get_governed_route(route_id: str) -> GovernedRouteRecord:
    """Return one catalogue row or raise for unknown identifiers.

    Parameters
    ----------
    route_id
        Taxonomy key to look up.

    Returns
    -------
    GovernedRouteRecord
        The matching catalogue row.

    Raises
    ------
    ValueError
        If ``route_id`` is empty/blank or not present in the catalogue.

    """
    if not route_id or not str(route_id).strip():
        raise ValueError("route_id must be a non-empty string")
    key = str(route_id).strip()
    try:
        return _ROUTE_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown governed route_id {key!r}; refuse silent invent-green support "
            f"(known_count={len(_ROUTE_BY_ID)})"
        ) from exc


def iter_governed_routes(
    *,
    family: RouteFamily | None = None,
    closure_status: RouteClosureStatus | None = None,
) -> tuple[GovernedRouteRecord, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    family
        Optional family filter.
    closure_status
        Optional closure-status filter.

    Returns
    -------
    tuple[GovernedRouteRecord, ...]
        Matching rows.

    """
    rows: Iterable[GovernedRouteRecord] = _CANONICAL_ROUTES
    if family is not None:
        rows = (row for row in rows if row.family == family)
    if closure_status is not None:
        rows = (row for row in rows if row.closure_status == closure_status)
    return tuple(rows)


def build_governed_route_matrix() -> dict[str, object]:
    """Build the full serialisable multi-ecosystem route matrix payload.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue cell (no blanks). Counts are
        derived from the catalogue so drift is visible to callers.

    """
    rows = [row.to_dict() for row in _CANONICAL_ROUTES]
    supported = sum(1 for row in _CANONICAL_ROUTES if row.closure_status == "supported")
    permanent = sum(1 for row in _CANONICAL_ROUTES if row.closure_status == "permanent_boundary")
    impl_path = sum(1 for row in _CANONICAL_ROUTES if row.closure_status == "implementation_path")
    return {
        "schema": GOVERNED_ROUTE_MATRIX_SCHEMA,
        "claim_boundary": GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY,
        "route_count": len(rows),
        "supported_count": supported,
        "permanent_boundary_count": permanent,
        "implementation_path_count": impl_path,
        "blank_cell_count": 0,
        "routes": rows,
    }


def _unknown_route_record(route_id: str) -> GovernedRouteRecord:
    """Synthesise a permanent_boundary record for unknown route identifiers."""
    return GovernedRouteRecord(
        route_id=f"unknown:{route_id}",
        family="competitor_boundary",
        closure_status="permanent_boundary",
        summary="Unknown route identifier refused without inventing support.",
        evidence=("governed_route_matrix.unknown_route",),
        rejected_alternatives=(),
        closure_reason=(
            f"route_id {route_id!r} is not in the governed catalogue; "
            "blank or invent-green cells are forbidden"
        ),
    )


def explain_route(
    route_id: str,
    capability: RouteCapability | Mapping[str, object] | None = None,
    *,
    unknown_policy: Literal["raise", "boundary"] = "raise",
) -> RouteExplanation:
    """Explain one route under a capability context without inventing support.

    Parameters
    ----------
    route_id
        Taxonomy key to explain.
    capability
        Capability context as :class:`RouteCapability` or a mapping with keys
        ``ecosystem``, ``method``, ``finite_shot``, ``allow_hardware``.
    unknown_policy
        ``raise`` (default) rejects unknown IDs with :class:`ValueError`.
        ``boundary`` returns a synthetic permanent_boundary explanation so
        operators can inspect fail-closed behaviour without exceptions.

    Returns
    -------
    RouteExplanation
        Deterministic selected route plus rejected alternatives.

    Raises
    ------
    ValueError
        If ``route_id``/capability fields are invalid, or if the route is
        unknown under ``unknown_policy='raise'``.
    TypeError
        If ``capability`` is neither a :class:`RouteCapability` nor a mapping.

    """
    if not route_id or not str(route_id).strip():
        raise ValueError("route_id must be a non-empty string")
    key = str(route_id).strip()
    if capability is None:
        cap = RouteCapability(ecosystem="native", method="auto")
    elif isinstance(capability, RouteCapability):
        cap = capability
    elif isinstance(capability, Mapping):
        cap = RouteCapability(
            ecosystem=str(capability.get("ecosystem", "native")),
            method=str(capability.get("method", "auto")),
            finite_shot=bool(capability.get("finite_shot", False)),
            allow_hardware=bool(capability.get("allow_hardware", False)),
        )
    else:
        raise TypeError(
            f"capability must be RouteCapability, mapping, or None (got {type(capability)!r})"
        )

    record = _ROUTE_BY_ID.get(key)
    notes: list[str] = []
    if record is None:
        if unknown_policy == "raise":
            raise ValueError(
                f"unknown governed route_id {key!r}; refuse silent invent-green support"
            )
        if unknown_policy != "boundary":
            raise ValueError(
                f"unknown_policy must be 'raise' or 'boundary' (got {unknown_policy!r})"
            )
        selected = _unknown_route_record(key)
        notes.append("unknown_policy=boundary synthesised permanent_boundary row")
        return RouteExplanation(
            route_id=key,
            capability=cap,
            selected=selected,
            rejected=(),
            notes=tuple(notes),
        )

    # Hardware policy: never upgrade permanent provider hardware routes when
    # allow_hardware is false (default). When true, still do not invent support —
    # only note that tickets are required.
    if record.route_id == "provider:hardware.gradient_live" and not cap.allow_hardware:
        notes.append("allow_hardware=False keeps provider live gradients boundary-closed")
    elif record.route_id == "provider:hardware.gradient_live" and cap.allow_hardware:
        notes.append(
            "allow_hardware=True does not invent support; owner-ticket evidence still required"
        )

    if cap.finite_shot and record.family == "transform" and record.closure_status == "supported":
        notes.append(
            "finite_shot=True does not silently change transform support; "
            "use finite-shot planner routes for shot budgets"
        )

    rejected: list[GovernedRouteRecord] = []
    for alt_id in record.rejected_alternatives:
        alt = _ROUTE_BY_ID.get(alt_id)
        if alt is not None:
            rejected.append(alt)

    # Same-family alternatives that are not selected, for operator contrast.
    for other in _CANONICAL_ROUTES:
        if (
            other.family == record.family
            and other.route_id != record.route_id
            and other not in rejected
            and other.closure_status != "supported"
        ):
            # Keep rejection set small and deterministic: only first same-family boundary.
            rejected.append(other)
            break

    return RouteExplanation(
        route_id=key,
        capability=cap,
        selected=record,
        rejected=tuple(rejected),
        notes=tuple(notes),
    )


def assert_no_blank_matrix_cells(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the matrix payload contains zero blank cells.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_governed_route_matrix`. When omitted,
        a fresh matrix is built.

    Returns
    -------
    dict[str, object]
        The validated payload.

    Raises
    ------
    ValueError
        If blank cells, missing statuses, or count drift are detected.

    """
    matrix = dict(payload) if payload is not None else build_governed_route_matrix()
    routes = matrix.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("governed route matrix must contain a non-empty routes list")
    blank = 0
    for index, row in enumerate(routes):
        if not isinstance(row, Mapping):
            raise ValueError(f"route row {index} must be a mapping")
        status = row.get("closure_status")
        route_id = row.get("route_id")
        if not route_id:
            blank += 1
            continue
        if status not in {"supported", "permanent_boundary", "implementation_path"}:
            blank += 1
            continue
        if status != "supported" and not row.get("closure_reason"):
            raise ValueError(f"route {route_id!r} is non-supported without closure_reason")
    if blank:
        raise ValueError(f"governed route matrix has {blank} blank or invalid cells; refuse green")
    blank_cell_count = matrix.get("blank_cell_count", -1)
    if not isinstance(blank_cell_count, int) or blank_cell_count != 0:
        raise ValueError("blank_cell_count must be 0")
    route_count = matrix.get("route_count", -1)
    if not isinstance(route_count, int) or route_count != len(routes):
        raise ValueError("route_count does not match routes list length")
    return matrix


__all__ = [
    "GOVERNED_ROUTE_MATRIX_CLAIM_BOUNDARY",
    "GOVERNED_ROUTE_MATRIX_SCHEMA",
    "GovernedRouteRecord",
    "RouteCapability",
    "RouteClosureStatus",
    "RouteExplanation",
    "RouteFamily",
    "assert_no_blank_matrix_cells",
    "build_governed_route_matrix",
    "explain_route",
    "get_governed_route",
    "iter_governed_routes",
    "list_governed_route_ids",
]
