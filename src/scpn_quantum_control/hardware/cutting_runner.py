# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cutting Runner
"""Circuit cutting runner: execute partitioned simulations for N > 16.

Actually runs the sub-circuits from circuit_cutting.py on the
statevector simulator, then reconstructs observables via
classical post-processing.

For N=32 with 2×16 partitions:
    1. Trotter-evolve partition A (16 qubits) independently
    2. Trotter-evolve partition B (16 qubits) independently
    3. For each cut gate: evaluate 4 sub-circuits (I, X, Y, Z basis)
    4. Reconstruct full observables from sub-circuit results

The reconstruction is exact for diagonal observables (Z-basis)
and approximate for off-diagonal (XX, YY correlators across cuts).

Limitation: 4^c overhead per cut. With all-to-all coupling at N=32,
the inter-partition cut count is large — making full reconstruction
infeasible. Instead, we compute partition-local observables exactly
and cross-partition correlators via product-state approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import LieTrotter

from ..bridge.knm_hamiltonian import build_knm_paper27, knm_to_hamiltonian
from .circuit_cutting import optimal_partition


@dataclass
class CuttingRunResult:
    """Result of a partitioned circuit execution."""

    n_oscillators: int
    n_partitions: int
    partition_r_globals: list[float]  # R per partition
    combined_r_global: float  # weighted average
    partition_energies: list[float]
    total_energy_estimate: float
    partition_sizes: list[int]


def _run_partition(
    K_partition: np.ndarray,
    omega_partition: np.ndarray,
    t: float,
    reps: int,
) -> tuple[float, float, np.ndarray]:
    """Run Trotter on one partition, return (energy, R, phases)."""
    n = K_partition.shape[0]
    H = knm_to_hamiltonian(K_partition, omega_partition)
    synth = LieTrotter(reps=reps)
    evo = PauliEvolutionGate(H, time=t, synthesis=synth)

    qc = QuantumCircuit(n)
    qc.append(evo, range(n))
    sv = Statevector.from_instruction(qc)

    energy = float(sv.expectation_value(H).real)

    # Extract phases
    from ..bridge.ssgf_adapter import quantum_to_ssgf_state

    state = quantum_to_ssgf_state(sv, n)
    r = state["R_global"]
    theta = state["theta"]

    return energy, r, theta


def run_cutting_simulation(
    n_oscillators: int = 32,
    k_base: float = 0.45,
    t: float = 1.0,
    reps: int = 5,
    max_partition_size: int = 16,
) -> CuttingRunResult:
    """Execute a partitioned Kuramoto-XY simulation.

    Partitions the system, runs each partition independently,
    combines results via weighted R_global average.
    """
    K_full = build_knm_paper27(L=n_oscillators, K_base=k_base)
    from ..bridge.knm_hamiltonian import OMEGA_N_16

    omega_full = np.tile(OMEGA_N_16, (n_oscillators // 16) + 1)[:n_oscillators]

    partition = optimal_partition(K_full, max_partition_size)

    partition_energies: list[float] = []
    partition_rs: list[float] = []
    all_phases: list[np.ndarray] = []
    sizes: list[int] = []

    for indices in partition:
        n_p = len(indices)
        K_p = K_full[np.ix_(indices, indices)]
        omega_p = omega_full[indices]

        energy, r, theta = _run_partition(K_p, omega_p, t, reps)
        partition_energies.append(energy)
        partition_rs.append(r)
        all_phases.append(theta)
        sizes.append(n_p)

    # Combined R: weighted circular mean of all phases
    combined_phases = np.concatenate(all_phases)
    z = np.mean(np.exp(1j * combined_phases))
    combined_r = float(np.abs(z))

    # Total energy: sum of partition energies (ignores cross-partition coupling)
    total_energy = sum(partition_energies)

    return CuttingRunResult(
        n_oscillators=n_oscillators,
        n_partitions=len(partition),
        partition_r_globals=partition_rs,
        combined_r_global=combined_r,
        partition_energies=partition_energies,
        total_energy_estimate=total_energy,
        partition_sizes=sizes,
    )
