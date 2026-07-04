# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — handbook worked workflow
"""Worked Kuramoto workflow using the public facade and no credentials.

The workflow is intentionally small and deterministic. It exercises the public
``oscillatools`` facade across the handbook path: accelerated RK4 integration,
frequency-locking diagnostics, linear stability, phase-coherence clustering,
mean-field critical-coupling theory, and projected synchronising-coupling design.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

import oscillatools as kuramoto

FloatArray: TypeAlias = NDArray[np.float64]

DT = 0.02
TRAJECTORY_STEPS = 80
DESIGN_STEPS = 40


@dataclass(frozen=True)
class WorkflowSummary:
    """Serializable summary of the worked Kuramoto workflow."""

    oscillator_count: int
    integrator_tier: str
    initial_order_parameter: float
    final_order_parameter: float
    designed_final_order_parameter: float
    frequency_synchronisation_index: float
    locked_fraction: float
    spectral_gap: float
    is_linearly_stable: bool
    gaussian_critical_coupling: float
    mean_pairwise_coupling: float
    designed_mean_pairwise_coupling: float
    cluster_count: int
    cluster_sizes: tuple[int, ...]
    leading_coherence_eigenvalue: float
    design_iterations: int
    design_converged: bool
    design_cost_history: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-stable representation of the workflow summary."""

        return asdict(self)


def build_problem() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the deterministic six-oscillator handbook problem.

    The topology is a sparse undirected chain with two weak long-range links.
    Frequencies are centred near zero so a short local run can show both
    phase-coherence diagnostics and the effect of the coupling-design pass.
    """

    theta0 = np.array([0.0, 0.28, 0.62, 1.05, 1.52, 2.05], dtype=np.float64)
    omega = np.array([-0.24, -0.10, -0.03, 0.05, 0.14, 0.22], dtype=np.float64)
    coupling = np.array(
        [
            [0.0, 0.38, 0.12, 0.0, 0.0, 0.05],
            [0.38, 0.0, 0.34, 0.08, 0.0, 0.0],
            [0.12, 0.34, 0.0, 0.32, 0.08, 0.0],
            [0.0, 0.08, 0.32, 0.0, 0.35, 0.12],
            [0.0, 0.0, 0.08, 0.35, 0.0, 0.37],
            [0.05, 0.0, 0.0, 0.12, 0.37, 0.0],
        ],
        dtype=np.float64,
    )
    return theta0, omega, coupling


def _mean_pairwise_coupling(coupling: FloatArray) -> float:
    """Return the mean off-diagonal coupling strength."""

    count = coupling.shape[0]
    off_diagonal = coupling[~np.eye(count, dtype=np.bool_)]
    return float(np.mean(off_diagonal))


def _rounded(values: NDArray[np.float64]) -> tuple[float, ...]:
    """Return rounded floats for stable JSON output."""

    return tuple(float(np.round(value, 12)) for value in values)


def run_workflow() -> WorkflowSummary:
    """Run the handbook workflow and return deterministic diagnostics."""

    theta0, omega, coupling = build_problem()
    trajectory = kuramoto.kuramoto_rk4_trajectory(theta0, omega, coupling, DT, TRAJECTORY_STEPS)
    final_theta = trajectory[-1]
    frequency = kuramoto.frequency_order_diagnostics(trajectory, dt=DT, tolerance=0.08)
    stability = kuramoto.stability_spectrum(final_theta, coupling)
    coherence = kuramoto.coherence_matrix(final_theta)
    clusters = kuramoto.cluster_partition(coherence, threshold=0.70)
    coherence_values, _ = kuramoto.coherence_spectrum(coherence)
    designed = kuramoto.design_synchronising_coupling(
        theta0,
        omega,
        coupling,
        DT,
        DESIGN_STEPS,
        max_iterations=6,
        learning_rate=1.5,
    )
    designed_trajectory = kuramoto.kuramoto_rk4_trajectory(
        theta0,
        omega,
        designed.coupling,
        DT,
        TRAJECTORY_STEPS,
    )
    integrator_tier = kuramoto.last_kuramoto_rk4_trajectory_tier_used() or "unknown"
    return WorkflowSummary(
        oscillator_count=int(theta0.size),
        integrator_tier=integrator_tier,
        initial_order_parameter=float(kuramoto.order_parameter(theta0)),
        final_order_parameter=float(kuramoto.order_parameter(final_theta)),
        designed_final_order_parameter=float(kuramoto.order_parameter(designed_trajectory[-1])),
        frequency_synchronisation_index=float(frequency.synchronisation_index),
        locked_fraction=float(frequency.locked_fraction),
        spectral_gap=float(stability.spectral_gap),
        is_linearly_stable=bool(stability.is_linearly_stable),
        gaussian_critical_coupling=float(
            kuramoto.gaussian_critical_coupling(float(np.std(omega)))
        ),
        mean_pairwise_coupling=_mean_pairwise_coupling(coupling),
        designed_mean_pairwise_coupling=_mean_pairwise_coupling(designed.coupling),
        cluster_count=int(clusters.count),
        cluster_sizes=tuple(int(size) for size in clusters.sizes),
        leading_coherence_eigenvalue=float(coherence_values[0]),
        design_iterations=int(designed.iterations),
        design_converged=bool(designed.converged),
        design_cost_history=_rounded(np.asarray(designed.cost_history, dtype=np.float64)),
    )


def main() -> None:
    """Print the worked workflow summary as stable JSON."""

    print(json.dumps(run_workflow().to_json_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
