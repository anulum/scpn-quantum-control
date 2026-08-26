# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — circuit-cutting product
"""Fail-closed circuit-cutting product for large synchronisation workloads.

The ambient planner supplies real partition and cut counts. This product adds
bounded hardware-safety shot accounting, synthetic reconstruction certificates, and
explicit decisions around the partition-local simulator. It does not implement
general cut reconstruction, submit hardware jobs, or promote omitted coupling
energy to a full-system observable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from .hardware.circuit_cutting import CircuitCuttingPlan, circuit_cutting_plan
from .hardware_safe_execution import CostModelStatus, get_execution_policy

CuttingPath = Literal[
    "dry_run_plan",
    "synthetic_reconstruction",
    "partition_local_diagnostic",
    "full_system_energy",
    "live_submit",
]
"""Governed circuit-cutting request paths."""

DecisionOutcome = Literal["allowed", "refused"]
"""Structured circuit-cutting decision outcome."""

CIRCUIT_CUTTING_PRODUCT_SCHEMA: Final[str] = "circuit_cutting_product.v2"
"""Schema identifier for the public product registry."""

CIRCUIT_CUTTING_RESOURCE_SCHEMA: Final[str] = "circuit_cutting_resource.v2"
"""Schema identifier for resource certificates."""

CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA: Final[str] = "circuit_cutting_reconstruction.v2"
"""Schema identifier for synthetic reconstruction certificates."""

CIRCUIT_CUTTING_CLAIM_BOUNDARY: Final[str] = (
    "bounded local circuit-cutting planning and synthetic certification only; cut cost "
    "is 4^cuts times shots_per_fragment under a hardware-safe no-submit execution "
    "policy; the ambient runner reports partition-local diagnostics and does not "
    "reconstruct omitted cross-partition energy; no general reconstruction, live QPU "
    "result, hardware advantage, or feasible dense large-N claim"
)
"""Shared non-promotional claim boundary."""


@dataclass(frozen=True, slots=True)
class CuttingSurfaceRow:
    """One frozen ambient circuit-cutting surface."""

    surface_id: str
    authority_pointer: str
    support_posture: str
    summary: str
    full_system_energy: bool
    live_submit: bool = False
    claim_boundary: str = CIRCUIT_CUTTING_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate inventory invariants."""
        if not self.surface_id or not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if not self.authority_pointer or not self.authority_pointer.strip():
            raise ValueError("authority_pointer must be non-empty")
        if self.support_posture not in {
            "bounded_planner",
            "partition_local_simulator",
        }:
            raise ValueError(f"unknown support_posture: {self.support_posture!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.live_submit:
            raise ValueError("circuit-cutting product surfaces cannot submit live jobs")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready inventory row."""
        return {
            "surface_id": self.surface_id,
            "authority_pointer": self.authority_pointer,
            "support_posture": self.support_posture,
            "summary": self.summary,
            "full_system_energy": self.full_system_energy,
            "live_submit": self.live_submit,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class CuttingResourceCertificate:
    """Hardware-safe fragment and shot-cost certificate."""

    schema: str
    policy_id: str
    n_oscillators: int
    n_partitions: int
    partition_sizes: tuple[int, ...]
    n_cuts: int
    fragment_evaluations: int | None
    shots_per_fragment: int
    estimated_total_shots: int | None
    cost_model_status: CostModelStatus
    fits_target: bool
    feasible: bool
    outcome: Literal["allowed_plan", "refused"]
    reason: str
    blockers: tuple[str, ...]
    no_submit: bool = True
    claim_boundary: str = CIRCUIT_CUTTING_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate resource-certificate invariants."""
        if self.schema != CIRCUIT_CUTTING_RESOURCE_SCHEMA:
            raise ValueError(f"unknown resource schema: {self.schema!r}")
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if self.n_oscillators < 2 or self.n_partitions < 1 or self.n_cuts < 0:
            raise ValueError("resource dimensions are invalid")
        if not self.partition_sizes or sum(self.partition_sizes) != self.n_oscillators:
            raise ValueError("partition_sizes must cover all oscillators")
        if self.shots_per_fragment <= 0:
            raise ValueError("shots_per_fragment must be positive")
        if self.fragment_evaluations is None:
            if self.estimated_total_shots is not None:
                raise ValueError("unbounded fragments cannot have a total-shot estimate")
        elif self.fragment_evaluations < 1 or self.estimated_total_shots != (
            self.fragment_evaluations * self.shots_per_fragment
        ):
            raise ValueError("estimated_total_shots must equal fragments times shots")
        if self.feasible != (self.outcome == "allowed_plan"):
            raise ValueError("feasible must agree with outcome")
        if self.feasible and self.blockers:
            raise ValueError("feasible certificates cannot list blockers")
        if not self.feasible and not self.blockers:
            raise ValueError("refused certificates require blockers")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.no_submit:
            raise ValueError("circuit-cutting product must remain no-submit")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready resource certificate."""
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "n_oscillators": self.n_oscillators,
            "n_partitions": self.n_partitions,
            "partition_sizes": list(self.partition_sizes),
            "n_cuts": self.n_cuts,
            "fragment_evaluations": self.fragment_evaluations,
            "shots_per_fragment": self.shots_per_fragment,
            "estimated_total_shots": self.estimated_total_shots,
            "cost_model_status": self.cost_model_status,
            "fits_target": self.fits_target,
            "feasible": self.feasible,
            "outcome": self.outcome,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "no_submit": self.no_submit,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class SyntheticReconstructionCertificate:
    """Error certificate for caller-supplied synthetic observable evidence."""

    schema: str
    observable_id: str
    exact_value: float
    reconstructed_value: float
    absolute_error: float
    declared_error_bound: float
    within_bound: bool
    synthetic_only: bool = True
    hardware_result: bool = False
    claim_boundary: str = CIRCUIT_CUTTING_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate synthetic-certificate invariants."""
        if self.schema != CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA:
            raise ValueError(f"unknown reconstruction schema: {self.schema!r}")
        if not self.observable_id or not self.observable_id.strip():
            raise ValueError("observable_id must be non-empty")
        values = (
            self.exact_value,
            self.reconstructed_value,
            self.absolute_error,
            self.declared_error_bound,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reconstruction certificate values must be finite")
        if self.absolute_error < 0.0 or self.declared_error_bound < 0.0:
            raise ValueError("reconstruction errors and bounds must be non-negative")
        if not math.isclose(
            self.absolute_error,
            abs(self.exact_value - self.reconstructed_value),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("absolute_error must match the supplied observable values")
        if self.within_bound != (self.absolute_error <= self.declared_error_bound):
            raise ValueError("within_bound must agree with the declared error bound")
        if not self.synthetic_only or self.hardware_result:
            raise ValueError("reconstruction certificates must remain synthetic-only")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready reconstruction certificate."""
        return {
            "schema": self.schema,
            "observable_id": self.observable_id,
            "exact_value": self.exact_value,
            "reconstructed_value": self.reconstructed_value,
            "absolute_error": self.absolute_error,
            "declared_error_bound": self.declared_error_bound,
            "within_bound": self.within_bound,
            "synthetic_only": self.synthetic_only,
            "hardware_result": self.hardware_result,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class CuttingPathDecision:
    """Fail-closed decision for one circuit-cutting path."""

    path: CuttingPath
    outcome: DecisionOutcome
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = CIRCUIT_CUTTING_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate path-decision invariants."""
        if self.path not in {
            "dry_run_plan",
            "synthetic_reconstruction",
            "partition_local_diagnostic",
            "full_system_energy",
            "live_submit",
        }:
            raise ValueError(f"unknown path: {self.path!r}")
        if self.outcome not in {"allowed", "refused"}:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.allowed != (self.outcome == "allowed"):
            raise ValueError("allowed must agree with outcome")
        if self.allowed and self.blockers:
            raise ValueError("allowed decisions cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("refused decisions require blockers")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready path decision."""
        return {
            "path": self.path,
            "outcome": self.outcome,
            "allowed": self.allowed,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


_SURFACES: Final[tuple[CuttingSurfaceRow, ...]] = (
    CuttingSurfaceRow(
        surface_id="resource_planner",
        authority_pointer="scpn_quantum_control.hardware.circuit_cutting.circuit_cutting_plan",
        support_posture="bounded_planner",
        summary="Real partition/cut inventory with 4^cuts classical overhead.",
        full_system_energy=False,
    ),
    CuttingSurfaceRow(
        surface_id="partition_local_simulator",
        authority_pointer="scpn_quantum_control.hardware.cutting_runner.run_cutting_simulation",
        support_posture="partition_local_simulator",
        summary=(
            "Local statevector partitions with labelled order-parameter and "
            "partition-local energy diagnostics; not general reconstruction."
        ),
        full_system_energy=False,
    ),
)


def list_cutting_surface_ids() -> tuple[str, ...]:
    """Return frozen ambient surface identifiers."""
    return tuple(row.surface_id for row in _SURFACES)


def get_cutting_surface(surface_id: str) -> CuttingSurfaceRow:
    """Return one surface row, failing closed for blank or unknown identifiers."""
    if not surface_id or not str(surface_id).strip():
        raise ValueError("surface_id must be a non-empty string")
    key = str(surface_id).strip()
    for row in _SURFACES:
        if row.surface_id == key:
            return row
    raise ValueError(f"unknown circuit-cutting surface_id: {key!r}")


def iter_cutting_surfaces(*, support_posture: str | None = None) -> tuple[CuttingSurfaceRow, ...]:
    """Return frozen surfaces, optionally filtered by support posture."""
    rows: Iterable[CuttingSurfaceRow] = _SURFACES
    if support_posture is not None:
        rows = (row for row in rows if row.support_posture == support_posture)
    return tuple(rows)


def build_cutting_resource_certificate(
    coupling: NDArray[np.float64],
    *,
    max_partition_size: int = 16,
    target_qubits: int = 127,
    shots_per_fragment: int = 256,
    policy_id: str = "default_no_submit",
    would_submit: bool = False,
) -> CuttingResourceCertificate:
    """Build a finite fragment×shots certificate under a hardware-safe policy."""
    if not isinstance(shots_per_fragment, int):
        raise TypeError("shots_per_fragment must be an integer")
    if shots_per_fragment <= 0:
        raise ValueError("shots_per_fragment must be positive")
    policy = get_execution_policy(policy_id)
    plan: CircuitCuttingPlan = circuit_cutting_plan(
        coupling,
        max_partition_size=max_partition_size,
        heron_qubits=target_qubits,
    )
    fragments = (
        None if not math.isfinite(plan.classical_overhead) else int(plan.classical_overhead)
    )
    total_shots = None if fragments is None else fragments * shots_per_fragment
    blockers: list[str] = []
    if would_submit:
        blockers.append("live submit is outside the bounded local circuit-cutting boundary")
    if not policy.no_submit:
        blockers.append("bounded circuit cutting requires a hardware-safe no-submit policy")
    if not plan.fits_on_heron:
        blockers.append("largest partition exceeds the target-qubit capacity")
    if fragments is None:
        blockers.append("4^cuts overhead is outside the finite planner bound")
    if shots_per_fragment > policy.max_shots_per_evaluation:
        blockers.append(
            f"shots_per_fragment {shots_per_fragment} exceeds execution-policy maximum "
            f"{policy.max_shots_per_evaluation}"
        )
    if total_shots is not None and total_shots > policy.max_total_shots:
        blockers.append(
            f"estimated_total_shots {total_shots} exceeds execution-policy maximum "
            f"{policy.max_total_shots}"
        )
    feasible = not blockers
    return CuttingResourceCertificate(
        schema=CIRCUIT_CUTTING_RESOURCE_SCHEMA,
        policy_id=policy.policy_id,
        n_oscillators=plan.n_oscillators,
        n_partitions=plan.n_partitions,
        partition_sizes=tuple(plan.partition_sizes),
        n_cuts=plan.n_cuts,
        fragment_evaluations=fragments,
        shots_per_fragment=shots_per_fragment,
        estimated_total_shots=total_shots,
        cost_model_status=policy.cost_model_status,
        fits_target=plan.fits_on_heron,
        feasible=feasible,
        outcome="allowed_plan" if feasible else "refused",
        reason=(
            "bounded no-submit cutting plan allowed; no reconstruction or provider "
            "execution occurred"
            if feasible
            else "circuit-cutting resource plan refused: " + "; ".join(blockers)
        ),
        blockers=tuple(blockers),
    )


def certify_synthetic_reconstruction(
    *,
    observable_id: str,
    exact_value: float,
    reconstructed_value: float,
    declared_error_bound: float,
) -> SyntheticReconstructionCertificate:
    """Certify caller-supplied synthetic observable agreement against a bound."""
    values = (exact_value, reconstructed_value, declared_error_bound)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("synthetic reconstruction inputs must be finite")
    if declared_error_bound < 0.0:
        raise ValueError("declared_error_bound must be non-negative")
    error = abs(float(exact_value) - float(reconstructed_value))
    return SyntheticReconstructionCertificate(
        schema=CIRCUIT_CUTTING_RECONSTRUCTION_SCHEMA,
        observable_id=observable_id,
        exact_value=float(exact_value),
        reconstructed_value=float(reconstructed_value),
        absolute_error=error,
        declared_error_bound=float(declared_error_bound),
        within_bound=error <= declared_error_bound,
    )


def decide_cutting_path(
    path: CuttingPath,
    *,
    resource: CuttingResourceCertificate,
    reconstruction: SyntheticReconstructionCertificate | None = None,
    accept_partition_local_energy: bool = False,
) -> CuttingPathDecision:
    """Decide whether a bounded cutting path may proceed without submission."""
    blockers = list(resource.blockers)
    if path == "live_submit":
        blockers.append("bounded circuit cutting is no-submit and never executes provider jobs")
    elif path == "full_system_energy" and resource.n_partitions > 1:
        blockers.append(
            "partitioned runner omits cross-partition coupling energy; full-system "
            "energy is unavailable"
        )
    elif path == "partition_local_diagnostic" and not accept_partition_local_energy:
        blockers.append("partition-local energy requires explicit caller acceptance")
    elif path == "synthetic_reconstruction":
        if reconstruction is None:
            blockers.append("synthetic reconstruction certificate is required")
        elif not reconstruction.within_bound:
            blockers.append("synthetic reconstruction exceeds its declared error bound")
    elif path not in {
        "dry_run_plan",
        "full_system_energy",
        "partition_local_diagnostic",
    }:
        raise ValueError(f"unknown circuit-cutting path: {path!r}")
    allowed = not blockers
    return CuttingPathDecision(
        path=path,
        outcome="allowed" if allowed else "refused",
        allowed=allowed,
        reason=(
            "bounded local circuit-cutting path allowed; no live submission occurred"
            if allowed
            else "circuit-cutting path refused: " + "; ".join(blockers)
        ),
        blockers=tuple(blockers),
    )


def build_circuit_cutting_product_registry() -> dict[str, object]:
    """Return the versioned, JSON-ready circuit-cutting product registry."""
    return {
        "schema": CIRCUIT_CUTTING_PRODUCT_SCHEMA,
        "surfaces": [row.to_dict() for row in _SURFACES],
        "cost_formula": "fragment_evaluations = 4^n_cuts; total_shots = fragments * shots",
        "reconstruction_evidence": "synthetic_only",
        "live_submit": False,
        "general_reconstruction": False,
        "claim_boundary": CIRCUIT_CUTTING_CLAIM_BOUNDARY,
    }


def assert_circuit_cutting_product_integrity() -> None:
    """Raise when catalogue or registry invariants drift."""
    ids = list_cutting_surface_ids()
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("circuit-cutting surface identifiers must be non-empty and unique")
    if any(row.live_submit for row in _SURFACES):
        raise RuntimeError("circuit-cutting product cannot expose live submission")
    registry = build_circuit_cutting_product_registry()
    if registry["schema"] != CIRCUIT_CUTTING_PRODUCT_SCHEMA:
        raise RuntimeError("circuit-cutting product schema drift")
    if registry["live_submit"] is not False or registry["general_reconstruction"] is not False:
        raise RuntimeError("circuit-cutting product claim boundary drift")


assert_circuit_cutting_product_integrity()
