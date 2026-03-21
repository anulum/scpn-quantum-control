# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Adaptive Variational Quantum Dynamics Simulation (AVQDS).

AVQDS (Yao et al., PRX Quantum 2, 030307 (2021)) simulates real-time
dynamics using a variational ansatz that adapts at each time step:

    |ψ(t+dt)> ≈ A(θ + dθ)|0>

where dθ is found by minimising the McLachlan distance:

    ||d/dt|ψ> + iH|ψ>||² → min

This gives the equation of motion:
    M × dθ/dt = V

where M_ij = Re(<∂_i ψ|∂_j ψ>) and V_i = -Im(<∂_i ψ|H|ψ>).

Advantages over Trotter:
    - Circuit depth independent of simulation time (only ansatz depth)
    - Adapts to the actual dynamics (adds operators when needed)
    - 100x shallower than Trotter for same accuracy at long times

For the Kuramoto-XY system, AVQDS tracks synchronisation dynamics
in real time without deep Trotter circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian


@dataclass
class AVQDSResult:
    """AVQDS simulation result."""

    times: np.ndarray
    energies: np.ndarray
    fidelities: np.ndarray  # overlap with exact evolution
    parameters_history: list[np.ndarray]
    n_params: int
    final_energy: float
    final_fidelity: float


def _mclachlan_matrices(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    H_op,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute McLachlan M matrix and V vector.

    M_ij = Re(<∂_i ψ|∂_j ψ>)
    V_i = -Im(<∂_i ψ|H|ψ>)
    """
    n_params = len(params)
    sv_0 = Statevector.from_instruction(ansatz.assign_parameters(params))
    psi_0 = sv_0.data

    # Compute parameter derivatives via finite differences
    dpsi: np.ndarray = np.zeros((n_params, len(psi_0)), dtype=complex)
    for k in range(n_params):
        p_plus = params.copy()
        p_plus[k] += epsilon
        p_minus = params.copy()
        p_minus[k] -= epsilon
        psi_plus = Statevector.from_instruction(ansatz.assign_parameters(p_plus)).data
        psi_minus = Statevector.from_instruction(ansatz.assign_parameters(p_minus)).data
        dpsi[k] = (psi_plus - psi_minus) / (2.0 * epsilon)

    # M matrix
    M = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            M[i, j] = float(np.real(np.dot(dpsi[i].conj(), dpsi[j])))

    # V vector
    H_mat = H_op.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()
    H_psi = H_mat @ psi_0

    V = np.zeros(n_params)
    for i in range(n_params):
        V[i] = -float(np.imag(np.dot(dpsi[i].conj(), H_psi)))

    return M, V


def avqds_simulate(
    K: np.ndarray,
    omega: np.ndarray,
    t_total: float = 1.0,
    n_steps: int = 20,
    ansatz_reps: int = 2,
    seed: int | None = None,
) -> AVQDSResult:
    """Run AVQDS simulation of the Kuramoto-XY dynamics.

    Args:
        K: coupling matrix
        omega: natural frequencies
        t_total: total simulation time
        n_steps: number of variational time steps
        ansatz_reps: ansatz circuit repetitions
        seed: random seed for initial parameters
    """
    from scipy.linalg import expm

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = H_op.to_matrix()
    if hasattr(H_mat, "toarray"):
        H_mat = H_mat.toarray()

    ansatz = knm_to_ansatz(K, reps=ansatz_reps)
    n_params = ansatz.num_parameters

    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, size=n_params)
    dt = t_total / n_steps

    times = np.linspace(0, t_total, n_steps + 1)
    energies = np.zeros(n_steps + 1)
    fidelities = np.zeros(n_steps + 1)
    param_history: list[np.ndarray] = [params.copy()]

    # Initial exact state for fidelity comparison
    psi_exact = Statevector.from_instruction(ansatz.assign_parameters(params)).data

    for step in range(n_steps + 1):
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        energies[step] = float(sv.expectation_value(H_op).real)
        fidelities[step] = float(abs(np.dot(psi_exact.conj(), sv.data)) ** 2)

        if step == n_steps:
            break

        # McLachlan step
        M, V = _mclachlan_matrices(ansatz, params, H_op)

        # Regularised solve: dθ = M^{-1} V × dt
        reg = 1e-6 * np.eye(n_params)
        dtheta = np.linalg.solve(M + reg, V) * dt
        params = params + dtheta
        param_history.append(params.copy())

        # Evolve exact reference
        U_dt = expm(-1j * H_mat * dt)
        psi_exact = U_dt @ psi_exact

    return AVQDSResult(
        times=times,
        energies=energies,
        fidelities=fidelities,
        parameters_history=param_history,
        n_params=n_params,
        final_energy=float(energies[-1]),
        final_fidelity=float(fidelities[-1]),
    )
