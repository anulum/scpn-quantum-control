# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SSGF quantum-in-the-loop geometry gradient
"""Governed SSGF quantum-in-the-loop geometry-gradient product.

This module freezes the public SSGF quantum surface, certifies the simulator
cost ``C = 1 - R``, and exposes a bounded central-finite-difference gradient
path over the latent geometry map ``W(z)``.  The latent map uses softplus, so a
parameter-shift rule on circuit angles is not directly a derivative with
respect to ``z``; that route is an explicit fail-closed route-matrix boundary.

All materialised evidence is local simulation.  This surface does not submit
hardware jobs, claim analytic AD, or treat an optional co-design observer as an
operational controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from .governed_route_matrix import get_governed_route
from .ssgf.quantum_costs import compute_quantum_costs
from .ssgf.quantum_gradient import _w_from_z, compute_quantum_gradient, quantum_cost
from .ssgf.quantum_outer_cycle import quantum_outer_cycle

SSGF_GEOMETRY_GRADIENT_SCHEMA: Final[str] = "ssgf_geometry_gradient_product.v1"
SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY: Final[str] = (
    "SSGF quantum-in-the-loop geometry-gradient product for bounded local "
    "statevector simulation; C=1-R certificates, central finite differences "
    "on the nonlinear softplus latent map W(z), phase-periodicity and step-"
    "refinement checks, and functional outer-cycle evidence only; parameter-"
    "shift on latent z, live QPU, analytic-AD, convergence, and advantage "
    "claims are refused"
)
MAX_OSCILLATORS: Final[int] = 6
DEFAULT_EPSILON: Final[float] = 0.01
DEFAULT_REFINEMENT_ATOL: Final[float] = 5e-3
DEFAULT_PERIODICITY_ATOL: Final[float] = 1e-9

GradientMethod = Literal["finite_difference", "parameter_shift"]
SurfaceRole = Literal[
    "latent_geometry",
    "quantum_cost",
    "cost_bundle",
    "gradient",
    "outer_cycle",
    "spectral_observer",
    "hamiltonian_bridge",
    "state_bridge",
]


@dataclass(frozen=True, slots=True)
class SsgfPublicSurfaceRow:
    """One frozen public SSGF quantum surface."""

    surface_id: str
    module_path: str
    symbol: str
    role: SurfaceRole
    hardware_submit_allowed: bool = False
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate inventory invariants."""
        if not self.surface_id.strip():
            raise ValueError("surface_id must be non-empty")
        if not self.module_path.strip() or not self.symbol.strip():
            raise ValueError("module_path and symbol must be non-empty")
        if self.hardware_submit_allowed:
            raise ValueError("SSGF product surfaces cannot allow hardware submission")


@dataclass(frozen=True, slots=True)
class GradientRouteDecision:
    """Governed route decision for a latent-geometry gradient."""

    method: GradientMethod
    route_id: str
    allowed: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class QuantumCostCertificate:
    """Cross-surface certificate for ``C = 1 - R``."""

    n_oscillators: int
    n_parameters: int
    cost: float
    r_global: float
    c_micro: float
    complement_residual: float
    cross_surface_residual: float
    geometry_symmetry_residual: float
    minimum_coupling: float
    certificate_digest: str
    schema: str = SSGF_GEOMETRY_GRADIENT_SCHEMA
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeometryGradientCertificate:
    """Metamorphic certificate for the supported latent finite difference."""

    n_oscillators: int
    n_parameters: int
    method: GradientMethod
    route_id: str
    cost: float
    r_global: float
    gradient: tuple[float, ...]
    refined_gradient: tuple[float, ...]
    gradient_norm: float
    geometry_symmetry_residual: float
    refinement_max_abs_delta: float
    periodic_cost_residual: float
    periodic_gradient_max_abs_delta: float
    n_evaluations: int
    expected_evaluations_per_gradient: int
    certificate_digest: str
    schema: str = SSGF_GEOMETRY_GRADIENT_SCHEMA
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SsgfGeometryObserverRecord:
    """Optional geometric-control and co-design evaluator telemetry record."""

    cost: float
    r_global: float
    gradient_norm: float
    geometry_symmetry_residual: float
    method: GradientMethod
    route_id: str
    operational_control_claim: bool = False
    schema: str = SSGF_GEOMETRY_GRADIENT_SCHEMA
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Refuse promotion of observer telemetry to operational control."""
        if self.operational_control_claim:
            raise ValueError("SSGF observer record cannot claim operational control")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready observer record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OuterCycleEvidence:
    """Functional, non-performance evidence from the ambient outer cycle."""

    n_oscillators: int
    n_parameters: int
    n_iterations: int
    cost_history: tuple[float, ...]
    r_global_history: tuple[float, ...]
    reported_final_cost: float
    reported_final_r_global: float
    cost_delta: float
    geometry_symmetry_residual: float
    minimum_coupling: float
    converged: bool
    evidence_label: str
    evidence_digest: str
    schema: str = SSGF_GEOMETRY_GRADIENT_SCHEMA
    claim_boundary: str = SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready evidence record."""
        return asdict(self)


_SURFACES: Final[tuple[SsgfPublicSurfaceRow, ...]] = (
    SsgfPublicSurfaceRow(
        "latent_geometry",
        "scpn_quantum_control.ssgf.quantum_gradient",
        "_w_from_z",
        "latent_geometry",
    ),
    SsgfPublicSurfaceRow(
        "quantum_cost",
        "scpn_quantum_control.ssgf.quantum_gradient",
        "quantum_cost",
        "quantum_cost",
    ),
    SsgfPublicSurfaceRow(
        "quantum_cost_bundle",
        "scpn_quantum_control.ssgf.quantum_costs",
        "compute_quantum_costs",
        "cost_bundle",
    ),
    SsgfPublicSurfaceRow(
        "quantum_gradient",
        "scpn_quantum_control.ssgf.quantum_gradient",
        "compute_quantum_gradient",
        "gradient",
    ),
    SsgfPublicSurfaceRow(
        "quantum_outer_cycle",
        "scpn_quantum_control.ssgf.quantum_outer_cycle",
        "quantum_outer_cycle",
        "outer_cycle",
    ),
    SsgfPublicSurfaceRow(
        "quantum_spectral",
        "scpn_quantum_control.ssgf.quantum_spectral",
        "spectral_bridge_analysis",
        "spectral_observer",
    ),
    SsgfPublicSurfaceRow(
        "hamiltonian_bridge",
        "scpn_quantum_control.bridge.ssgf_adapter",
        "ssgf_w_to_hamiltonian",
        "hamiltonian_bridge",
    ),
    SsgfPublicSurfaceRow(
        "state_bridge",
        "scpn_quantum_control.bridge.ssgf_adapter",
        "ssgf_state_to_quantum/quantum_to_ssgf_state",
        "state_bridge",
    ),
)

_UNSUITABLE_SCENARIOS: Final[tuple[str, ...]] = (
    "using circuit parameter-shift directly for latent z through nonlinear softplus W(z)",
    "supplying a latent vector whose size is not n_oscillators choose 2",
    "treating finite-difference agreement as analytic or exact automatic differentiation",
    "promoting local statevector evidence to live-QPU or hardware robustness evidence",
    "claiming outer-cycle convergence or advantage from a bounded functional trace",
    "using optional co-design observer telemetry as an operational controller decision",
)


def list_ssgf_public_surfaces() -> tuple[SsgfPublicSurfaceRow, ...]:
    """Return the frozen SSGF quantum surface inventory."""
    return _SURFACES


def ssgf_gradient_unsuitable_scenarios() -> tuple[str, ...]:
    """Return explicit anti-silent-wrong scenarios."""
    return _UNSUITABLE_SCENARIOS


def decide_ssgf_gradient_route(method: str) -> GradientRouteDecision:
    """Resolve the supported FD route or the latent parameter-shift boundary."""
    key = method.strip().lower() if isinstance(method, str) else ""
    if key == "finite_difference":
        route_id = "transform:ssgf.latent_finite_difference"
        row = get_governed_route(route_id)
        if row.closure_status != "supported":
            raise RuntimeError(f"governed route drift: {route_id} is not supported")
        return GradientRouteDecision(
            method="finite_difference",
            route_id=route_id,
            allowed=True,
            reason="central finite difference is supported for bounded local latent-z probes",
            blockers=(),
        )
    if key == "parameter_shift":
        route_id = "transform:ssgf.latent_parameter_shift"
        row = get_governed_route(route_id)
        if row.closure_status != "permanent_boundary":
            raise RuntimeError(f"governed route drift: {route_id} is not a permanent boundary")
        return GradientRouteDecision(
            method="parameter_shift",
            route_id=route_id,
            allowed=False,
            reason="parameter-shift on latent z is not the circuit-angle derivative",
            blockers=(row.closure_reason,),
        )
    raise ValueError("method must be 'finite_difference' or 'parameter_shift'")


def _validate_problem(
    z: NDArray[np.float64],
    n_oscillators: int,
    theta_init: NDArray[np.float64],
    omega: NDArray[np.float64] | None,
    *,
    epsilon: float,
    dt: float,
    trotter_reps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate the bounded product contract before ambient evaluation."""
    if not isinstance(n_oscillators, int) or isinstance(n_oscillators, bool):
        raise TypeError("n_oscillators must be an integer")
    if not 2 <= n_oscillators <= MAX_OSCILLATORS:
        raise ValueError(f"n_oscillators must be in [2, {MAX_OSCILLATORS}]")
    expected = n_oscillators * (n_oscillators - 1) // 2
    latent = np.asarray(z, dtype=np.float64)
    theta = np.asarray(theta_init, dtype=np.float64)
    frequencies = (
        np.zeros(n_oscillators, dtype=np.float64)
        if omega is None
        else np.asarray(omega, dtype=np.float64)
    )
    if latent.shape != (expected,):
        raise ValueError(f"z shape must be ({expected},), got {latent.shape}")
    if theta.shape != (n_oscillators,):
        raise ValueError(f"theta_init shape must be ({n_oscillators},), got {theta.shape}")
    if frequencies.shape != (n_oscillators,):
        raise ValueError(f"omega shape must be ({n_oscillators},), got {frequencies.shape}")
    if not np.all(np.isfinite(latent)):
        raise ValueError("z must contain only finite values")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_init must contain only finite values")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("omega must contain only finite values")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(trotter_reps, int) or isinstance(trotter_reps, bool) or trotter_reps < 1:
        raise ValueError("trotter_reps must be a positive integer")
    return latent, theta, frequencies


def _digest_payload(payload: Mapping[str, object]) -> str:
    """Return canonical SHA-256 for a JSON-ready mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def certify_quantum_cost(
    z: NDArray[np.float64],
    n_oscillators: int,
    theta_init: NDArray[np.float64],
    *,
    omega: NDArray[np.float64] | None = None,
    dt: float = 0.1,
    trotter_reps: int = 3,
    atol: float = 1e-10,
) -> QuantumCostCertificate:
    """Certify ``quantum_cost == c_micro == 1 - R`` across ambient surfaces."""
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    latent, theta, frequencies = _validate_problem(
        z,
        n_oscillators,
        theta_init,
        omega,
        epsilon=DEFAULT_EPSILON,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    geometry = _w_from_z(latent, n_oscillators)
    cost = quantum_cost(geometry, theta, frequencies, dt, trotter_reps)
    bundle = compute_quantum_costs(geometry, theta, frequencies, dt, trotter_reps)
    complement_residual = abs(cost + bundle.r_global - 1.0)
    cross_surface_residual = abs(cost - bundle.c_micro)
    symmetry_residual = float(np.max(np.abs(geometry - geometry.T)))
    values = (cost, bundle.r_global, bundle.c_micro, complement_residual, cross_surface_residual)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ambient SSGF cost certificate contains non-finite values")
    if not 0.0 <= cost <= 1.0 or not 0.0 <= bundle.r_global <= 1.0:
        raise ValueError("ambient SSGF cost/order parameter must be in [0, 1]")
    if complement_residual > atol or cross_surface_residual > atol:
        raise ValueError("ambient SSGF C=1-R cross-surface certificate failed")
    if symmetry_residual > atol or float(np.min(geometry)) < -atol:
        raise ValueError("latent geometry must be symmetric and non-negative")
    payload: dict[str, object] = {
        "schema": SSGF_GEOMETRY_GRADIENT_SCHEMA,
        "n_oscillators": n_oscillators,
        "z": latent.tolist(),
        "theta_init": theta.tolist(),
        "omega": frequencies.tolist(),
        "cost": cost,
        "r_global": bundle.r_global,
        "c_micro": bundle.c_micro,
        "dt": dt,
        "trotter_reps": trotter_reps,
    }
    return QuantumCostCertificate(
        n_oscillators=n_oscillators,
        n_parameters=latent.size,
        cost=float(cost),
        r_global=float(bundle.r_global),
        c_micro=float(bundle.c_micro),
        complement_residual=float(complement_residual),
        cross_surface_residual=float(cross_surface_residual),
        geometry_symmetry_residual=symmetry_residual,
        minimum_coupling=float(np.min(geometry)),
        certificate_digest=_digest_payload(payload),
    )


def certify_geometry_gradient(
    z: NDArray[np.float64],
    n_oscillators: int,
    theta_init: NDArray[np.float64],
    *,
    omega: NDArray[np.float64] | None = None,
    method: str = "finite_difference",
    epsilon: float = DEFAULT_EPSILON,
    dt: float = 0.1,
    trotter_reps: int = 3,
    refinement_atol: float = DEFAULT_REFINEMENT_ATOL,
    periodicity_atol: float = DEFAULT_PERIODICITY_ATOL,
) -> GeometryGradientCertificate:
    """Certify FD step refinement and phase-periodicity metamorphic laws."""
    decision = decide_ssgf_gradient_route(method)
    if not decision.allowed:
        raise ValueError("SSGF gradient route refused: " + "; ".join(decision.blockers))
    if not math.isfinite(refinement_atol) or refinement_atol < 0.0:
        raise ValueError("refinement_atol must be finite and non-negative")
    if not math.isfinite(periodicity_atol) or periodicity_atol < 0.0:
        raise ValueError("periodicity_atol must be finite and non-negative")
    latent, theta, frequencies = _validate_problem(
        z,
        n_oscillators,
        theta_init,
        omega,
        epsilon=epsilon,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    primary = compute_quantum_gradient(
        latent,
        n_oscillators,
        theta_init=theta,
        omega=frequencies,
        epsilon=epsilon,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    refined = compute_quantum_gradient(
        latent,
        n_oscillators,
        theta_init=theta,
        omega=frequencies,
        epsilon=epsilon / 2.0,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    periodic = compute_quantum_gradient(
        latent,
        n_oscillators,
        theta_init=theta + 2.0 * np.pi,
        omega=frequencies,
        epsilon=epsilon,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    expected = 1 + 2 * latent.size
    results = (primary, refined, periodic)
    if any(result.gradient.shape != latent.shape for result in results):
        raise ValueError("ambient SSGF gradient shape drift")
    if any(result.n_evaluations != expected for result in results):
        raise ValueError("ambient SSGF gradient evaluation-count drift")
    if any(
        not np.all(np.isfinite(result.gradient))
        or not math.isfinite(result.cost)
        or not math.isfinite(result.r_global)
        for result in results
    ):
        raise ValueError("ambient SSGF gradient result contains non-finite values")
    refinement_delta = float(np.max(np.abs(primary.gradient - refined.gradient)))
    periodic_cost_residual = abs(primary.cost - periodic.cost)
    periodic_gradient_delta = float(np.max(np.abs(primary.gradient - periodic.gradient)))
    geometry = _w_from_z(latent, n_oscillators)
    geometry_symmetry_residual = float(np.max(np.abs(geometry - geometry.T)))
    if refinement_delta > refinement_atol:
        raise ValueError(
            "SSGF finite-difference step-refinement law failed: "
            f"{refinement_delta:.12g} > {refinement_atol:.12g}"
        )
    if periodic_cost_residual > periodicity_atol or periodic_gradient_delta > periodicity_atol:
        raise ValueError("SSGF 2pi phase-periodicity metamorphic law failed")
    payload: dict[str, object] = {
        "schema": SSGF_GEOMETRY_GRADIENT_SCHEMA,
        "route_id": decision.route_id,
        "n_oscillators": n_oscillators,
        "z": latent.tolist(),
        "theta_init": theta.tolist(),
        "omega": frequencies.tolist(),
        "epsilon": epsilon,
        "cost": primary.cost,
        "r_global": primary.r_global,
        "gradient": primary.gradient.tolist(),
        "refined_gradient": refined.gradient.tolist(),
    }
    return GeometryGradientCertificate(
        n_oscillators=n_oscillators,
        n_parameters=latent.size,
        method="finite_difference",
        route_id=decision.route_id,
        cost=float(primary.cost),
        r_global=float(primary.r_global),
        gradient=tuple(float(value) for value in primary.gradient),
        refined_gradient=tuple(float(value) for value in refined.gradient),
        gradient_norm=float(np.linalg.norm(primary.gradient)),
        geometry_symmetry_residual=geometry_symmetry_residual,
        refinement_max_abs_delta=refinement_delta,
        periodic_cost_residual=float(periodic_cost_residual),
        periodic_gradient_max_abs_delta=periodic_gradient_delta,
        n_evaluations=sum(result.n_evaluations for result in results),
        expected_evaluations_per_gradient=expected,
        certificate_digest=_digest_payload(payload),
    )


def geometry_observer_from_certificate(
    certificate: GeometryGradientCertificate,
) -> SsgfGeometryObserverRecord:
    """Map certified geometry evidence into optional observer telemetry."""
    return SsgfGeometryObserverRecord(
        cost=certificate.cost,
        r_global=certificate.r_global,
        gradient_norm=certificate.gradient_norm,
        geometry_symmetry_residual=certificate.geometry_symmetry_residual,
        method=certificate.method,
        route_id=certificate.route_id,
    )


def materialise_outer_cycle_evidence(
    *,
    n_oscillators: int = 2,
    z_init: NDArray[np.float64] | None = None,
    theta_init: NDArray[np.float64] | None = None,
    learning_rate: float = 0.1,
    max_iterations: int = 3,
    convergence_threshold: float = 1e-4,
    dt: float = 0.1,
    trotter_reps: int = 1,
    seed: int = 70,
) -> OuterCycleEvidence:
    """Run a deterministic pure-quantum functional outer-cycle trace."""
    if not 2 <= n_oscillators <= MAX_OSCILLATORS:
        raise ValueError(f"n_oscillators must be in [2, {MAX_OSCILLATORS}]")
    n_parameters = n_oscillators * (n_oscillators - 1) // 2
    latent = (
        np.zeros(n_parameters, dtype=np.float64)
        if z_init is None
        else np.asarray(z_init, dtype=np.float64)
    )
    theta = (
        np.linspace(0.1, 0.7, n_oscillators, dtype=np.float64)
        if theta_init is None
        else np.asarray(theta_init, dtype=np.float64)
    )
    _validate_problem(
        latent,
        n_oscillators,
        theta,
        None,
        epsilon=DEFAULT_EPSILON,
        dt=dt,
        trotter_reps=trotter_reps,
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(convergence_threshold) or convergence_threshold < 0.0:
        raise ValueError("convergence_threshold must be finite and non-negative")
    result = quantum_outer_cycle(
        n_osc=n_oscillators,
        z_init=latent,
        theta_init=theta,
        alpha=1.0,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        dt=dt,
        trotter_reps=trotter_reps,
        seed=seed,
    )
    costs = tuple(float(value) for value in result.cost_history)
    orders = tuple(float(value) for value in result.r_global_history)
    if not costs or len(costs) != len(orders) or result.n_iterations != len(costs):
        raise ValueError("ambient outer-cycle history contract failed")
    if not all(math.isfinite(value) for value in (*costs, *orders)):
        raise ValueError("ambient outer-cycle evidence contains non-finite values")
    if any(not 0.0 <= value <= 1.0 for value in orders):
        raise ValueError("ambient outer-cycle order parameter must be in [0, 1]")
    symmetry_residual = float(np.max(np.abs(result.W_optimised - result.W_optimised.T)))
    minimum_coupling = float(np.min(result.W_optimised))
    if symmetry_residual > 1e-10 or minimum_coupling < -1e-10:
        raise ValueError("ambient outer-cycle geometry contract failed")
    payload: dict[str, object] = {
        "schema": SSGF_GEOMETRY_GRADIENT_SCHEMA,
        "n_oscillators": n_oscillators,
        "z_init": latent.tolist(),
        "theta_init": theta.tolist(),
        "cost_history": list(costs),
        "r_global_history": list(orders),
        "reported_final_cost": result.final_cost,
        "reported_final_r_global": result.final_r_global,
        "seed": seed,
    }
    return OuterCycleEvidence(
        n_oscillators=n_oscillators,
        n_parameters=n_parameters,
        n_iterations=result.n_iterations,
        cost_history=costs,
        r_global_history=orders,
        reported_final_cost=float(result.final_cost),
        reported_final_r_global=float(result.final_r_global),
        cost_delta=float(costs[-1] - costs[0]),
        geometry_symmetry_residual=symmetry_residual,
        minimum_coupling=minimum_coupling,
        converged=bool(result.converged),
        evidence_label="functional_non_isolated_local_simulation",
        evidence_digest=_digest_payload(payload),
    )


def build_ssgf_geometry_gradient_registry() -> dict[str, object]:
    """Build the versioned product registry."""
    fd = decide_ssgf_gradient_route("finite_difference")
    shift = decide_ssgf_gradient_route("parameter_shift")
    return {
        "schema": SSGF_GEOMETRY_GRADIENT_SCHEMA,
        "claim_boundary": SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY,
        "max_oscillators": MAX_OSCILLATORS,
        "hardware_submit_allowed": False,
        "analytic_ad_claim_allowed": False,
        "parameter_shift_on_latent_z_allowed": False,
        "surface_count": len(_SURFACES),
        "blank_entry_count": 0,
        "surfaces": [asdict(row) for row in _SURFACES],
        "gradient_routes": [asdict(fd), asdict(shift)],
        "unsuitable_scenarios": list(_UNSUITABLE_SCENARIOS),
        "composition": {
            "geometric_control": "SsgfGeometryObserverRecord geometry observer",
            "codesign": "optional evaluator telemetry only",
            "metamorphic_verification": "step-refinement and 2pi phase-periodicity laws",
            "unsuitable_scenarios": "unsupported z dimensions and latent parameter-shift fail closed",
        },
    }


def assert_ssgf_geometry_gradient_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert inventory, policy, and governed route integrity."""
    registry = dict(payload) if payload is not None else build_ssgf_geometry_gradient_registry()
    surfaces = registry.get("surfaces")
    routes = registry.get("gradient_routes")
    scenarios = registry.get("unsuitable_scenarios")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("SSGF registry requires non-empty surfaces")
    if not isinstance(routes, list) or len(routes) != 2:
        raise ValueError("SSGF registry requires exactly two gradient routes")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("SSGF registry requires unsuitable scenarios")
    surface_ids: set[str] = set()
    for row in surfaces:
        if not isinstance(row, Mapping):
            raise ValueError("SSGF surface rows must be mappings")
        surface_id = str(row.get("surface_id", "")).strip()
        if not surface_id:
            raise ValueError("SSGF registry contains a blank surface_id")
        if surface_id in surface_ids:
            raise ValueError(f"duplicate SSGF surface_id: {surface_id!r}")
        surface_ids.add(surface_id)
        if row.get("hardware_submit_allowed") is not False:
            raise ValueError("SSGF surfaces must keep hardware_submit_allowed=False")
    expected_surfaces = {row.surface_id for row in _SURFACES}
    if surface_ids != expected_surfaces:
        raise ValueError("SSGF surface inventory drift")
    by_method = {str(row.get("method")): row for row in routes if isinstance(row, Mapping)}
    if set(by_method) != {"finite_difference", "parameter_shift"}:
        raise ValueError("SSGF gradient route method set drift")
    if by_method["finite_difference"].get("allowed") is not True:
        raise ValueError("SSGF finite-difference route must remain supported")
    if by_method["parameter_shift"].get("allowed") is not False:
        raise ValueError("SSGF latent parameter-shift route must remain fail closed")
    for policy in (
        "hardware_submit_allowed",
        "analytic_ad_claim_allowed",
        "parameter_shift_on_latent_z_allowed",
    ):
        if registry.get(policy) is not False:
            raise ValueError(f"SSGF policy {policy} must remain False")
    if registry.get("surface_count") != len(surfaces):
        raise ValueError("SSGF surface_count drift")
    if registry.get("blank_entry_count") != 0:
        raise ValueError("SSGF blank_entry_count must be zero")
    return registry


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_PERIODICITY_ATOL",
    "DEFAULT_REFINEMENT_ATOL",
    "MAX_OSCILLATORS",
    "SSGF_GEOMETRY_GRADIENT_CLAIM_BOUNDARY",
    "SSGF_GEOMETRY_GRADIENT_SCHEMA",
    "GeometryGradientCertificate",
    "GradientRouteDecision",
    "OuterCycleEvidence",
    "QuantumCostCertificate",
    "SsgfGeometryObserverRecord",
    "SsgfPublicSurfaceRow",
    "assert_ssgf_geometry_gradient_integrity",
    "build_ssgf_geometry_gradient_registry",
    "certify_geometry_gradient",
    "certify_quantum_cost",
    "decide_ssgf_gradient_route",
    "geometry_observer_from_certificate",
    "list_ssgf_public_surfaces",
    "materialise_outer_cycle_evidence",
    "ssgf_gradient_unsuitable_scenarios",
]
