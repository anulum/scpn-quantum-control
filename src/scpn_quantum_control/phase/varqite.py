# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Varqite
"""Variational Quantum Imaginary Time Evolution (VarQITE).

VarQITE (McArdle et al., npj Quantum Information 5, 75 (2019))
finds the ground state via imaginary time evolution:

    |ψ(τ)> = e^{-Hτ} |ψ(0)> / ||e^{-Hτ} |ψ(0)>||

As τ → ∞, |ψ(τ)> → ground state (for non-orthogonal initial state).

The variational version uses McLachlan's principle:

    min ||d|ψ>/dτ + (H - <H>)|ψ>||²

giving the equation of motion:
    A × dθ/dτ = C

where A_ij = Re(<∂_i ψ|∂_j ψ>) and C_i = -Re(<∂_i ψ|(H - <H>)|ψ>).

Advantages over COBYLA VQE:
    - Guaranteed convergence to ground state (no local minima)
    - No optimizer hyperparameters
    - Natural stopping criterion (dθ/dτ → 0)

For the Kuramoto-XY system, VarQITE finds the maximum synchronisation
configuration without the barren plateau problem of VQE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian
from ..hardware.classical import classical_exact_diag


@dataclass
class VarQITEResult:
    """VarQITE ground state result."""

    energy: float
    exact_energy: float
    relative_error_pct: float
    n_steps: int
    energy_history: list[float]
    converged: bool
    optimal_params: np.ndarray


def _varqite_matrices(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    H_op,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VarQITE A matrix and C vector.

    A_ij = Re(<∂_i ψ|∂_j ψ>)
    C_i = -Re(<∂_i ψ|(H - <H>)|ψ>)
    """
    n_params = len(params)
    sv_0 = Statevector.from_instruction(ansatz.assign_parameters(params))
    psi_0 = sv_0.data

    H_mat = H_op.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()
    e_mean = float(np.real(psi_0.conj() @ H_mat @ psi_0))
    H_shifted = H_mat - e_mean * np.eye(len(psi_0))

    dpsi: np.ndarray = np.zeros((n_params, len(psi_0)), dtype=complex)
    for k in range(n_params):
        p_plus = params.copy()
        p_plus[k] += epsilon
        p_minus = params.copy()
        p_minus[k] -= epsilon
        psi_plus = Statevector.from_instruction(ansatz.assign_parameters(p_plus)).data
        psi_minus = Statevector.from_instruction(ansatz.assign_parameters(p_minus)).data
        dpsi[k] = (psi_plus - psi_minus) / (2.0 * epsilon)

    A = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            A[i, j] = float(np.real(np.dot(dpsi[i].conj(), dpsi[j])))

    H_psi = H_shifted @ psi_0
    C = np.zeros(n_params)
    for i in range(n_params):
        C[i] = -float(np.real(np.dot(dpsi[i].conj(), H_psi)))

    return A, C


def varqite_ground_state(
    K: np.ndarray,
    omega: np.ndarray,
    tau_total: float = 3.0,
    n_steps: int = 30,
    ansatz_reps: int = 2,
    convergence_threshold: float = 1e-5,
    seed: int | None = None,
) -> VarQITEResult:
    """Find ground state via VarQITE.

    Args:
        K: coupling matrix
        omega: natural frequencies
        tau_total: total imaginary time
        n_steps: number of ITE steps
        ansatz_reps: ansatz repetitions
        convergence_threshold: stop when |ΔE| < threshold
        seed: random seed
    """
    n = K.shape[0]
    H_op = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=ansatz_reps)
    n_params = ansatz.num_parameters

    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, size=n_params)
    dtau = tau_total / n_steps

    energy_history: list[float] = []
    converged = False

    for step in range(n_steps):
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        energy = float(sv.expectation_value(H_op).real)
        energy_history.append(energy)

        if step > 0 and abs(energy_history[-1] - energy_history[-2]) < convergence_threshold:
            converged = True
            break

        A, C = _varqite_matrices(ansatz, params, H_op)
        reg = 1e-6 * np.eye(n_params)
        dtheta = np.linalg.solve(A + reg, C) * dtau
        params = params + dtheta

    # Final energy
    sv_final = Statevector.from_instruction(ansatz.assign_parameters(params))
    final_energy = float(sv_final.expectation_value(H_op).real)
    energy_history.append(final_energy)

    exact = classical_exact_diag(n, K=K, omega=omega)
    exact_e = exact["ground_energy"]
    rel_err = abs(final_energy - exact_e) / max(abs(exact_e), 1e-15) * 100

    return VarQITEResult(
        energy=final_energy,
        exact_energy=exact_e,
        relative_error_pct=rel_err,
        n_steps=len(energy_history) - 1,
        energy_history=energy_history,
        converged=converged,
        optimal_params=params,
    )
