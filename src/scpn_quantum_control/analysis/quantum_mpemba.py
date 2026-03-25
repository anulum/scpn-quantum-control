# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum Mpemba effect in synchronization dynamics.

The Mpemba effect: a system prepared far from equilibrium
relaxes to equilibrium FASTER than one prepared close to it.
The quantum version was observed experimentally (Nature Comms 2024)
and studied at Ising criticality (Wei et al. 2025).

Nobody has asked: does a state prepared far from synchronization
reach the synchronized state faster than one prepared close to it?

Protocol:
1. Prepare initial states at different "distances" from the
   synchronized ground state (measured by 1 - fidelity)
2. Evolve under Lindblad dynamics: H(K) + depolarizing noise
3. Track the approach to the steady state (measured by R or fidelity)
4. Mpemba effect: if initial distance d_far > d_close but
   relaxation time t_far < t_close

The Lindblad equation:
    dρ/dt = -i[H, ρ] + γ Σ_k (L_k ρ L_k† - {L_k†L_k, ρ}/2)

where L_k are depolarizing jump operators on each qubit.

Prior art: QME at Ising criticality, QME in spin chains.
QME + synchronization: completely empty intersection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class MpembaResult:
    """Result of quantum Mpemba experiment."""

    times: np.ndarray
    fidelity_near: np.ndarray  # F(ρ(t), ρ_ss) for "near" initial state
    fidelity_far: np.ndarray  # F(ρ(t), ρ_ss) for "far" initial state
    R_near: np.ndarray  # order parameter for near
    R_far: np.ndarray  # order parameter for far
    initial_distance_near: float  # 1 - F(ρ_near, ρ_ss)
    initial_distance_far: float  # 1 - F(ρ_far, ρ_ss)
    has_mpemba: bool  # far state reaches equilibrium first
    crossing_time: float | None  # time when far overtakes near


def _lindblad_superoperator(H: np.ndarray, gamma: float, n_qubits: int) -> np.ndarray:
    """Build the Lindblad superoperator L such that dvec(ρ)/dt = L vec(ρ).

    Depolarizing noise: L_k = sqrt(γ) * σ_k^(q) for σ ∈ {X,Y,Z}, q ∈ qubits.
    """
    dim = H.shape[0]

    # Hamiltonian part: -i(H⊗I - I⊗H^T)
    I_d = np.eye(dim)
    L_total = -1j * (np.kron(H, I_d) - np.kron(I_d, H.T))

    # Amplitude damping: L_k = sqrt(gamma) * |0><1| on qubit k
    # Drives each qubit toward |0⟩. The NESS depends on H vs gamma.
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)

    for q in range(n_qubits):
        L_k = np.eye(1, dtype=complex)
        for j in range(n_qubits):
            L_k = np.kron(L_k, lowering if j == q else np.eye(2))
        L_k *= np.sqrt(gamma)

        # Dissipator: L⊗L* - (L†L⊗I + I⊗L^T L*)/2
        L_total += (
            np.kron(L_k, L_k.conj())
            - 0.5 * np.kron(L_k.conj().T @ L_k, I_d)
            - 0.5 * np.kron(I_d, L_k.T @ L_k.conj())
        )

    result: np.ndarray = L_total
    return result


def _fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """State fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))²."""
    sqrt_rho = _matrix_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = _matrix_sqrt(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)  # clamp numerical negatives
    sqrt_diag = np.diag(np.sqrt(eigvals))
    result: np.ndarray = eigvecs @ sqrt_diag @ eigvecs.conj().T
    return result


def _R_from_density_matrix(rho: np.ndarray, n_qubits: int) -> float:
    """Order parameter R from density matrix."""
    phases = np.zeros(n_qubits)

    for k in range(n_qubits):
        # Build single-qubit X and Y operators
        for pauli_idx, pauli in enumerate(
            [
                np.array([[0, 1], [1, 0]], dtype=complex),  # X
                np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
            ]
        ):
            op = np.eye(1)
            for j in range(n_qubits):
                op = np.kron(op, pauli if j == k else np.eye(2))
            exp_val = float(np.real(np.trace(rho @ op)))
            if pauli_idx == 0:
                ex = exp_val
            else:
                ey = exp_val
        phases[k] = np.arctan2(ey, ex)

    return float(abs(np.mean(np.exp(1j * phases))))


def mpemba_experiment(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
    gamma: float = 0.1,
    t_max: float = 5.0,
    n_steps: int = 30,
) -> MpembaResult:
    """Run quantum Mpemba experiment for synchronization.

    Two initial states:
    - "near": ground state of H(K_base*0.8) — close to NESS
    - "far": |+⟩^n (equal superposition) — far from NESS

    Both evolve under Lindblad(H(K_base), amplitude damping γ).
    Mpemba effect: "far" state reaches NESS faster than "near".
    """
    n = len(omega)
    dim = 2**n
    K = K_base * K_topology

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = H_op.to_matrix()

    # Steady state: evolve |0⟩ for long time under Lindblad
    L_super = _lindblad_superoperator(H_mat, gamma, n)

    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[0] = 1.0
    rho_init = np.outer(psi_0, psi_0.conj())
    vec_ss = expm(L_super * 50.0) @ rho_init.flatten()
    rho_ss = vec_ss.reshape(dim, dim)
    rho_ss = (rho_ss + rho_ss.conj().T) / 2  # ensure Hermitian
    rho_ss /= np.trace(rho_ss)

    # Near initial state: ground state of H(K_base*0.8) — close to NESS
    K_near = (K_base * 0.8) * K_topology
    H_near = knm_to_hamiltonian(K_near, omega).to_matrix()
    eigvals, eigvecs = np.linalg.eigh(H_near)
    psi_near = np.ascontiguousarray(eigvecs[:, 0])
    rho_near = np.outer(psi_near, psi_near.conj())

    # Far initial state: |+⟩^n (equal superposition — all phases randomized)
    psi_plus = np.ones(dim, dtype=complex) / np.sqrt(dim)
    rho_far = np.outer(psi_plus, psi_plus.conj())

    # Initial distances
    d_near = 1.0 - _fidelity(rho_near, rho_ss)
    d_far = 1.0 - _fidelity(rho_far, rho_ss)

    # Time evolution
    dt = t_max / n_steps
    times = np.linspace(0, t_max, n_steps + 1)
    fid_near = np.zeros(n_steps + 1)
    fid_far = np.zeros(n_steps + 1)
    R_near_arr = np.zeros(n_steps + 1)
    R_far_arr = np.zeros(n_steps + 1)

    fid_near[0] = _fidelity(rho_near, rho_ss)
    fid_far[0] = _fidelity(rho_far, rho_ss)
    R_near_arr[0] = _R_from_density_matrix(rho_near, n)
    R_far_arr[0] = _R_from_density_matrix(rho_far, n)

    # Propagator for one time step
    U_dt = expm(L_super * dt)

    vec_near = rho_near.flatten()
    vec_far = rho_far.flatten()

    for step in range(n_steps):
        vec_near = U_dt @ vec_near
        vec_far = U_dt @ vec_far

        rho_n = vec_near.reshape(dim, dim)
        rho_f = vec_far.reshape(dim, dim)

        fid_near[step + 1] = _fidelity(rho_n, rho_ss)
        fid_far[step + 1] = _fidelity(rho_f, rho_ss)
        R_near_arr[step + 1] = _R_from_density_matrix(rho_n, n)
        R_far_arr[step + 1] = _R_from_density_matrix(rho_f, n)

    # Detect Mpemba crossing: far state fidelity overtakes near
    crossing_t = None
    has_mpemba = False
    for i in range(1, n_steps + 1):
        if fid_far[i] > fid_near[i] + 1e-6:
            crossing_t = float(times[i])
            has_mpemba = True
            break

    return MpembaResult(
        times=times,
        fidelity_near=fid_near,
        fidelity_far=fid_far,
        R_near=R_near_arr,
        R_far=R_far_arr,
        initial_distance_near=d_near,
        initial_distance_far=d_far,
        has_mpemba=has_mpemba,
        crossing_time=crossing_t,
    )
