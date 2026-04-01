# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Cross Domain Transfer
"""Cross-domain VQE transfer learning.

Tests whether VQE parameters optimized for one physical system
transfer to a different physical system with similar coupling topology.

This is WIDE OPEN in the literature. All existing VQE transfer work
stays within the same domain (molecule→molecule, graph→graph). Nobody
has demonstrated cross-physics-domain parameter transfer.

Hypothesis: if the VQE energy landscape is determined by the coupling
TOPOLOGY (graph structure) rather than the physical interpretation
of the Hamiltonian, then parameters learned on System A should
warm-start optimization on System B — even if A is a photosynthetic
complex and B is a power grid.

Prior art:
    Galda et al. (2021): QAOA parameter transfer between regular graphs.
    Tseng et al. (arXiv:2501.01507): Formal VQC transfer learning framework.
    NONE: cross-physics-domain VQE transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian


@dataclass
class TransferResult:
    """Result of VQE parameter transfer between two systems."""

    source_system: str
    target_system: str
    random_init_energy: float
    transfer_init_energy: float
    random_init_iters: int
    transfer_init_iters: int
    exact_energy: float
    speedup: float
    energy_improvement: float


@dataclass
class PhysicalSystem:
    """A physical system defined by its coupling matrix and frequencies."""

    name: str
    K: np.ndarray
    omega: np.ndarray


def build_systems(n_qubits: int = 4) -> list[PhysicalSystem]:
    """Build a set of physical systems for transfer learning experiments.

    All systems use the same number of oscillators but different coupling
    structures and frequencies, modeling different physics domains.
    """
    from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    systems = []

    # System 1: SCPN K_nm (neural oscillators)
    K_scpn = build_knm_paper27(L=n_qubits)
    omega_scpn = OMEGA_N_16[:n_qubits]
    systems.append(PhysicalSystem("scpn_neural", K_scpn, omega_scpn))

    # System 2: Nearest-neighbor chain (condensed matter)
    K_nn = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits - 1):
        K_nn[i, i + 1] = K_nn[i + 1, i] = 0.5
    omega_nn = np.linspace(0.5, 2.0, n_qubits)
    systems.append(PhysicalSystem("nearest_neighbor", K_nn, omega_nn))

    # System 3: All-to-all uniform (mean-field)
    K_all = np.ones((n_qubits, n_qubits)) * 0.3
    np.fill_diagonal(K_all, 0.0)
    omega_all = np.ones(n_qubits) * 1.5
    systems.append(PhysicalSystem("mean_field", K_all, omega_all))

    # System 4: Power-law decay (gravitational/Coulomb analogy)
    K_pl = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            K_pl[i, j] = K_pl[j, i] = 1.0 / (abs(i - j) ** 1.5)
    omega_pl = np.array([0.8 + 0.3 * i for i in range(n_qubits)])
    systems.append(PhysicalSystem("power_law", K_pl, omega_pl))

    return systems


def _vqe_optimize(
    ansatz,
    hamiltonian: SparsePauliOp,
    init_params: np.ndarray,
    maxiter: int = 200,
) -> tuple[float, int, np.ndarray]:
    """COBYLA VQE optimization. Returns (energy, n_evals, optimal_params)."""
    history: list[float] = []

    def cost(params):
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        e = float(sv.expectation_value(hamiltonian).real)
        history.append(e)
        return e

    res = minimize(cost, init_params, method="COBYLA", options={"maxiter": maxiter})
    return float(res.fun), res.nfev, res.x


def transfer_experiment(
    source: PhysicalSystem,
    target: PhysicalSystem,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 42,
) -> TransferResult:
    """Run VQE transfer learning between source and target systems.

    1. Optimize VQE on source system (random init)
    2. Use source's optimal params as warm start for target
    3. Compare against random init on target

    Both use the K_nm-informed ansatz built from the TARGET system's K.
    """
    rng = np.random.default_rng(seed)

    # Source optimization
    source_ansatz = knm_to_ansatz(source.K, reps=reps)
    source_H = knm_to_hamiltonian(source.K, source.omega)
    source_init = rng.uniform(-np.pi, np.pi, source_ansatz.num_parameters)
    _, _, source_params = _vqe_optimize(source_ansatz, source_H, source_init, maxiter)

    # Target setup
    target_ansatz = knm_to_ansatz(target.K, reps=reps)
    target_H = knm_to_hamiltonian(target.K, target.omega)
    n_target_params = target_ansatz.num_parameters

    # Exact ground state energy
    from ..hardware.classical import classical_exact_diag

    exact = classical_exact_diag(target.K.shape[0], K=target.K, omega=target.omega)
    exact_energy = exact["ground_energy"]

    # Transfer: map source params to target (truncate/pad if sizes differ)
    transfer_params = np.zeros(n_target_params)
    n_copy = min(len(source_params), n_target_params)
    transfer_params[:n_copy] = source_params[:n_copy]

    # Random init baseline
    random_init = rng.uniform(-np.pi, np.pi, n_target_params)
    random_energy, random_iters, _ = _vqe_optimize(target_ansatz, target_H, random_init, maxiter)

    # Transfer init
    transfer_energy, transfer_iters, _ = _vqe_optimize(
        target_ansatz, target_H, transfer_params, maxiter
    )

    speedup = random_iters / max(transfer_iters, 1)
    improvement = random_energy - transfer_energy

    return TransferResult(
        source_system=source.name,
        target_system=target.name,
        random_init_energy=random_energy,
        transfer_init_energy=transfer_energy,
        random_init_iters=random_iters,
        transfer_init_iters=transfer_iters,
        exact_energy=exact_energy,
        speedup=speedup,
        energy_improvement=improvement,
    )


def run_transfer_matrix(
    n_qubits: int = 4,
    reps: int = 2,
    maxiter: int = 100,
    seed: int = 42,
) -> list[TransferResult]:
    """Run all-pairs transfer learning between physical systems.

    Returns a list of TransferResult for each (source, target) pair
    where source != target.
    """
    systems = build_systems(n_qubits)
    results = []
    for source in systems:
        for target in systems:
            if source.name == target.name:
                continue
            r = transfer_experiment(source, target, reps, maxiter, seed)
            results.append(r)
    return results


def summarize_transfer(results: list[TransferResult]) -> dict:
    """Summarize transfer matrix: which pairs show positive transfer?"""
    positive = [r for r in results if r.energy_improvement > 0]
    negative = [r for r in results if r.energy_improvement <= 0]

    return {
        "n_pairs": len(results),
        "n_positive_transfer": len(positive),
        "n_negative_transfer": len(negative),
        "best_transfer": max(results, key=lambda r: r.speedup).source_system
        + " → "
        + max(results, key=lambda r: r.speedup).target_system
        if results
        else None,
        "best_speedup": max(r.speedup for r in results) if results else 0.0,
        "mean_speedup": float(np.mean([r.speedup for r in results])) if results else 0.0,
    }
