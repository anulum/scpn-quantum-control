# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetry-sector mitigation compiler contract
"""Planning contract for symmetry- and sector-aware error mitigation.

This module does not mutate circuits and does not submit hardware jobs. It is a
fail-closed planner that decides whether existing mitigation primitives such as
parity post-selection, symmetry expansion, and GUESS symmetry-decay correction
are eligible for a given Kuramoto/XY experiment descriptor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

MitigationPrimitive = Literal[
    "parity_postselection",
    "symmetry_expansion",
    "guess_symmetry_decay",
]
PlanStatus = Literal["eligible", "blocked"]


@dataclass(frozen=True, slots=True)
class SymmetrySectorProblem:
    """Minimal descriptor for sector-aware mitigation planning."""

    n_qubits: int
    coupling_matrix: tuple[tuple[float, ...], ...]
    omega: tuple[float, ...]
    initial_state: str
    measurement_basis: Literal["z", "x", "y", "xyz", "counts"]
    has_raw_counts: bool
    has_noise_scaled_symmetry_observables: bool = False


@dataclass(frozen=True, slots=True)
class SymmetrySectorPlan:
    """Mitigation plan with explicit blockers and claim boundary."""

    status: PlanStatus
    expected_parity: int | None
    primitives: tuple[MitigationPrimitive, ...]
    blockers: tuple[str, ...]
    required_evidence: tuple[str, ...]
    benchmark_gates: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable plan."""
        return asdict(self)


def _validate_problem(
    problem: SymmetrySectorProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Validate a problem descriptor and return arrays plus blockers."""
    blockers: list[str] = []
    if problem.n_qubits <= 0:
        blockers.append("n_qubits must be positive")
    coupling = np.asarray(problem.coupling_matrix, dtype=float)
    omega = np.asarray(problem.omega, dtype=float)
    if coupling.shape != (problem.n_qubits, problem.n_qubits):
        blockers.append("coupling_matrix must be square with shape n_qubits x n_qubits")
    elif not np.allclose(coupling, coupling.T, atol=1e-12):
        blockers.append("coupling_matrix must be symmetric for XY parity-sector planning")
    if coupling.size and not np.all(np.isfinite(coupling)):
        blockers.append("coupling_matrix must be finite")
    if omega.shape != (problem.n_qubits,):
        blockers.append("omega length must match n_qubits")
    elif not np.all(np.isfinite(omega)):
        blockers.append("omega must be finite")
    if len(problem.initial_state) != problem.n_qubits:
        blockers.append("initial_state length must match n_qubits")
    elif any(bit not in {"0", "1"} for bit in problem.initial_state):
        blockers.append("initial_state must be a computational-basis bitstring")
    if not problem.has_raw_counts:
        blockers.append("raw measurement counts are required before mitigation planning")
    return coupling, omega, blockers


def _initial_parity(initial_state: str) -> int:
    """Return computational-basis parity for a validated initial state."""
    return initial_state.count("1") % 2


def plan_symmetry_sector_mitigation(problem: SymmetrySectorProblem) -> SymmetrySectorPlan:
    """Plan eligible sector-aware mitigation primitives for a problem.

    The planner is conservative by construction. It only enables parity-aware
    primitives for validated symmetric XY-style inputs with raw counts, and it
    only enables GUESS when noise-scaled symmetry observables are explicitly
    present.
    """
    _coupling, _omega, blockers = _validate_problem(problem)
    primitives: list[MitigationPrimitive] = []
    expected_parity: int | None = None
    if not blockers:
        expected_parity = _initial_parity(problem.initial_state)
        if problem.measurement_basis in {"z", "counts"}:
            primitives.append("parity_postselection")
        if problem.measurement_basis in {"x", "y", "xyz", "counts"}:
            primitives.append("symmetry_expansion")
        if problem.has_noise_scaled_symmetry_observables:
            primitives.append("guess_symmetry_decay")
        else:
            blockers.append("GUESS requires noise-scaled symmetry observables")
    status: PlanStatus = "blocked" if blockers else "eligible"
    return SymmetrySectorPlan(
        status=status,
        expected_parity=expected_parity,
        primitives=tuple(primitives) if status == "eligible" else (),
        blockers=tuple(blockers),
        required_evidence=(
            "validated symmetric XY coupling matrix",
            "finite omega vector",
            "computational-basis initial state",
            "raw counts for every mitigated circuit",
            "noise-scaled symmetry observable table for GUESS",
        ),
        benchmark_gates=(
            "scpn-bench sync-benchmark-gate",
            "hardware result-pack verifier for any raw-count replay claim",
            "focused mitigation regression tests before circuit-path integration",
        ),
        claim_boundary=(
            "Planner output is an eligibility contract only. It does not mutate "
            "circuits, submit hardware jobs, prove improved hardware performance, "
            "or broaden DLA/GUESS claims without benchmark and raw-count evidence."
        ),
    )
