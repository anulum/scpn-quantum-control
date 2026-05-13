# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor Jump Method for Open Systems
"""Monte Carlo Wave Function (MCWF) method for open Kuramoto-XY systems.

The trajectory solver uses sparse Hamiltonians, sparse jump operators, and
state-vector propagation. It avoids the O(4^n) density-matrix bottleneck and
the dense Hamiltonian allocations used by exact diagonalisation, but it is not
a Matrix Product State implementation. Memory still scales as O(2^n) in the
trajectory state vector.

For the Kuramoto-XY system:

  1. Evolve |ψ⟩ under effective non-Hermitian Hamiltonian H_eff = H - iΣ L†L/2
  2. At each timestep, with probability dp = 1 - ⟨ψ|ψ⟩, apply a quantum jump
  3. Average over many trajectories to recover the density matrix ρ

Reference: Causer et al., Nature Comms (2025);
Dalibard et al., PRL 68, 580 (1992).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from ..bridge.knm_hamiltonian import knm_to_sparse_matrix
from ..dense_budget import require_dense_allocation


def _build_effective_hamiltonian(
    H: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    """Build H_eff = H - (i/2) Σ L†L."""
    H_eff: np.ndarray = H.copy().astype(np.complex128)
    for L in lindblad_ops:
        H_eff -= 0.5j * (L.conj().T @ L)
    return H_eff


def _validate_mcwf_inputs(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_amp: float,
    gamma_deph: float,
    t_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    K_arr = np.asarray(K, dtype=np.float64)
    omega_arr = np.asarray(omega, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError("K must be a square two-dimensional coupling matrix.")
    if omega_arr.ndim != 1 or omega_arr.shape[0] != K_arr.shape[0]:
        raise ValueError("omega must be a one-dimensional vector matching K.")
    if not np.all(np.isfinite(K_arr)) or not np.all(np.isfinite(omega_arr)):
        raise ValueError("K and omega must contain only finite values.")
    if not np.isfinite(gamma_amp) or gamma_amp < 0:
        raise ValueError("gamma_amp must be finite and non-negative.")
    if not np.isfinite(gamma_deph) or gamma_deph < 0:
        raise ValueError("gamma_deph must be finite and non-negative.")
    if not np.isfinite(t_max) or t_max < 0:
        raise ValueError("t_max must be finite and non-negative.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    return K_arr, omega_arr


def _single_qubit_sparse(pauli: str, qubit: int, n: int) -> sparse.csr_matrix:
    matrices = {
        "X": sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
        "Y": sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
        "Z": sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128)),
        "+": sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128)),
        "-": sparse.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128)),
    }
    op = sparse.identity(1, dtype=np.complex128, format="csr")
    ident = sparse.identity(2, dtype=np.complex128, format="csr")
    for i in range(n):
        op = sparse.kron(op, matrices[pauli] if i == qubit else ident, format="csr")
    return op


def _build_sparse_lindblad_ops(
    n: int,
    gamma_amp: float,
    gamma_deph: float,
) -> list[sparse.csr_matrix]:
    ops: list[sparse.csr_matrix] = []
    for i in range(n):
        if gamma_amp > 0:
            ops.append(np.sqrt(gamma_amp) * _single_qubit_sparse("-", i, n))
        if gamma_deph > 0:
            ops.append(np.sqrt(gamma_deph / 2) * _single_qubit_sparse("Z", i, n))
    return ops


def _initial_product_state(omega: np.ndarray) -> np.ndarray:
    psi = np.array([1.0 + 0.0j], dtype=np.complex128)
    for omega_i in omega:
        angle = float(omega_i) % (2 * np.pi)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        qubit_state = np.array([c, s], dtype=np.complex128)
        psi = np.asarray(np.kron(psi, qubit_state), dtype=np.complex128)
    return psi


def mcwf_trajectory(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_amp: float = 0.05,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.05,
    seed: int | None = None,
    *,
    max_dense_gib: float | None = None,
) -> dict:
    """Run a single MCWF trajectory.

    Parameters
    ----------
    K, omega : coupling matrix and frequencies
    gamma_amp : amplitude damping rate per qubit
    gamma_deph : dephasing rate per qubit
    t_max : total time
    dt : timestep
    seed : RNG seed

    Returns
    -------
    dict with keys: times, R, psi_final, n_jumps
    """
    K, omega = _validate_mcwf_inputs(K, omega, gamma_amp, gamma_deph, t_max, dt)
    n = K.shape[0]
    dim = 2**n
    require_dense_allocation(
        n,
        rank=1,
        object_count=4,
        max_gib=max_dense_gib,
        label="MCWF statevector workspace",
    )
    rng = np.random.default_rng(seed)

    H = knm_to_sparse_matrix(K, omega).astype(np.complex128)
    L_ops = _build_sparse_lindblad_ops(n, gamma_amp, gamma_deph)
    anti_hermitian = sparse.csr_matrix((dim, dim), dtype=np.complex128)
    for L in L_ops:
        anti_hermitian = anti_hermitian + (L.getH() @ L)
    H_eff = H - 0.5j * anti_hermitian

    # Initial state: product of Ry-rotated qubits
    psi = _initial_product_state(omega)

    n_steps = max(1, int(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    R_history = np.zeros(n_steps + 1)
    R_history[0] = _order_param_vec(psi, n)

    step_generator = -1j * H_eff * dt

    n_jumps = 0
    for step in range(1, n_steps + 1):
        # Evolve under H_eff
        psi_new = expm_multiply(step_generator, psi)
        norm_sq = float(np.real(np.dot(psi_new.conj(), psi_new)))

        # Jump probability
        dp = float(np.clip(1 - norm_sq, 0.0, 1.0))
        r = rng.uniform()

        if r < dp and len(L_ops) > 0:
            # Quantum jump: choose which L_k fires
            probs = np.array([float(np.real(np.dot((L @ psi).conj(), L @ psi))) for L in L_ops])
            total = probs.sum()
            if total > 1e-15:
                probs /= total
                k = rng.choice(len(L_ops), p=probs)
                psi = L_ops[k] @ psi
                psi /= np.linalg.norm(psi)
                n_jumps += 1
            else:
                psi = psi_new / np.sqrt(norm_sq) if norm_sq > 1e-15 else psi_new
        else:
            # No jump: renormalise
            psi = psi_new / np.sqrt(norm_sq) if norm_sq > 1e-15 else psi_new

        R_history[step] = _order_param_vec(psi, n)

    return {
        "times": times,
        "R": R_history,
        "psi_final": psi,
        "n_jumps": n_jumps,
    }


def mcwf_ensemble(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_amp: float = 0.05,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.05,
    n_trajectories: int = 50,
    seed: int | None = None,
    *,
    max_dense_gib: float | None = None,
) -> dict:
    """Run an ensemble of MCWF trajectories and average.

    Returns dict with keys: times, R_mean, R_std, R_trajectories, total_jumps
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)

    all_R = []
    total_jumps = 0

    for i in range(n_trajectories):
        traj = mcwf_trajectory(
            K,
            omega,
            gamma_amp,
            gamma_deph,
            t_max,
            dt,
            seed=int(seeds[i]),
            max_dense_gib=max_dense_gib,
        )
        all_R.append(traj["R"])
        total_jumps += traj["n_jumps"]

    R_array = np.array(all_R)
    return {
        "times": np.linspace(0, t_max, max(1, int(t_max / dt)) + 1),
        "R_mean": np.mean(R_array, axis=0),
        "R_std": np.std(R_array, axis=0),
        "R_trajectories": R_array,
        "total_jumps": total_jumps,
        "n_trajectories": n_trajectories,
    }


def _order_param_vec(psi: np.ndarray, n: int) -> float:
    """Order parameter R from state vector. Uses Rust (851×) when available."""
    # Rust fast path
    try:
        import scpn_quantum_engine as eng

        return float(
            eng.order_param_from_statevector(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n,
            )
        )
    except (ImportError, Exception):
        pass

    z = 0.0 + 0.0j
    dim = 2**n
    for i in range(n):
        # <X_i> = Re(Σ_k ψ*_k ψ_{k^(1<<i)}) * 2 for single-qubit X
        exp_x = 0.0
        exp_y = 0.0
        for k in range(dim):
            k_flip = k ^ (1 << i)
            exp_x += float(np.real(psi[k].conj() * psi[k_flip]))
            exp_y += float(np.imag(psi[k_flip].conj() * psi[k]))
        z += exp_x + 1j * exp_y
    z /= n
    return float(abs(z))
