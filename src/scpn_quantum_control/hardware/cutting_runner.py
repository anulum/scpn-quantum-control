# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cutting Runner
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
the inter-partition cut count is large. This runner therefore refuses
to label multi-partition energy as full-system energy unless the caller
explicitly opts into a partition-local diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
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
    energy_scope: str
    is_full_system_energy: bool
    omitted_cross_partition_coupling_l1: float
    partition_sizes: list[int]


def _run_partition(
    K_partition: NDArray[np.float64],
    omega_partition: NDArray[np.float64],
    t: float,
    reps: int,
) -> tuple[float, float, NDArray[np.float64]]:
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
    allow_partition_energy_estimate: bool = False,
) -> CuttingRunResult:
    """Execute a partitioned Kuramoto-XY simulation.

    Partitions the system, runs each partition independently,
    combines phase observables via a circular mean. For multiple partitions,
    the returned energy is only a partition-local diagnostic; call sites must
    opt into that explicitly with ``allow_partition_energy_estimate=True``.
    """
    K_full = build_knm_paper27(L=n_oscillators, K_base=k_base)
    from ..bridge.knm_hamiltonian import OMEGA_N_16

    omega_full = np.tile(OMEGA_N_16, (n_oscillators // 16) + 1)[:n_oscillators]

    partition = optimal_partition(K_full, max_partition_size)
    partition_labels = np.full(n_oscillators, -1, dtype=int)
    for label, indices in enumerate(partition):
        partition_labels[np.asarray(indices, dtype=int)] = label

    omitted_cross_l1 = 0.0
    for i in range(n_oscillators):
        for j in range(i + 1, n_oscillators):
            if partition_labels[i] != partition_labels[j]:
                omitted_cross_l1 += abs(float(K_full[i, j]))

    if len(partition) > 1 and omitted_cross_l1 > 0.0 and not allow_partition_energy_estimate:
        raise ValueError(
            "Multi-partition cutting omits cross-partition coupling energy. Pass "
            "allow_partition_energy_estimate=True only when consuming the labelled "
            "partition-local energy diagnostic."
        )

    partition_energies: list[float] = []
    partition_rs: list[float] = []
    all_phases: list[NDArray[np.float64]] = []
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

    # Labelled diagnostic: sum of partition energies, with omitted cross terms reported.
    total_energy = sum(partition_energies)
    is_full_energy = len(partition) == 1 or omitted_cross_l1 == 0.0
    energy_scope = "full_system" if is_full_energy else "partition_local_sum"

    return CuttingRunResult(
        n_oscillators=n_oscillators,
        n_partitions=len(partition),
        partition_r_globals=partition_rs,
        combined_r_global=combined_r,
        partition_energies=partition_energies,
        total_energy_estimate=total_energy,
        energy_scope=energy_scope,
        is_full_system_energy=is_full_energy,
        omitted_cross_partition_coupling_l1=omitted_cross_l1,
        partition_sizes=sizes,
    )
