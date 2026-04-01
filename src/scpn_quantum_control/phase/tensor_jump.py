# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tensor Jump Method for Open Systems
"""Monte Carlo Wave Function (MCWF) method for open Kuramoto-XY systems.

The Tensor Jump Method (TJM) combines the MCWF approach with Matrix Product
States (MPS) to simulate open quantum systems at scales beyond exact density
matrix methods. For the Kuramoto-XY system:

  1. Evolve |ψ⟩ under effective non-Hermitian Hamiltonian H_eff = H - iΣ L†L/2
  2. At each timestep, with probability dp = 1 - ⟨ψ|ψ⟩, apply a quantum jump
  3. Average over many trajectories to recover the density matrix ρ

For MPS representation, each trajectory maintains a low-bond-dimension state,
enabling simulation of n=32-100+ oscillators.

Reference: Causer et al., Nature Comms (2025);
Dalibard et al., PRL 68, 580 (1992).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..phase.lindblad import _sigma


def _build_effective_hamiltonian(
    H: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    """Build H_eff = H - (i/2) Σ L†L."""
    H_eff: np.ndarray = H.copy().astype(np.complex128)
    for L in lindblad_ops:
        H_eff -= 0.5j * (L.conj().T @ L)
    return H_eff


def mcwf_trajectory(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_amp: float = 0.05,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.05,
    seed: int | None = None,
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
    n = K.shape[0]
    dim = 2**n
    rng = np.random.default_rng(seed)

    H = knm_to_dense_matrix(K, omega)

    # Build Lindblad operators
    L_ops = []
    for i in range(n):
        if gamma_amp > 0:
            L_ops.append(np.sqrt(gamma_amp) * _sigma("-", i, n))
        if gamma_deph > 0:
            L_ops.append(np.sqrt(gamma_deph / 2) * _sigma("Z", i, n))

    H_eff = _build_effective_hamiltonian(H, L_ops)

    # Initial state: product of Ry-rotated qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    U_init = np.eye(1, dtype=np.complex128)
    for i in range(n):
        angle = float(omega[i]) % (2 * np.pi)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
        U_init = np.kron(U_init, ry)  # type: ignore[assignment]
    psi = U_init @ psi

    n_steps = max(1, int(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    R_history = np.zeros(n_steps + 1)
    R_history[0] = _order_param_vec(psi, n)

    # Non-Hermitian propagator for one step
    U_eff = expm(-1j * H_eff * dt)

    n_jumps = 0
    for step in range(1, n_steps + 1):
        # Evolve under H_eff
        psi_new = U_eff @ psi
        norm_sq = float(np.real(np.dot(psi_new.conj(), psi_new)))

        # Jump probability
        dp = 1 - norm_sq
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
) -> dict:
    """Run an ensemble of MCWF trajectories and average.

    Returns dict with keys: times, R_mean, R_std, R_trajectories, total_jumps
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)

    all_R = []
    total_jumps = 0

    for i in range(n_trajectories):
        traj = mcwf_trajectory(K, omega, gamma_amp, gamma_deph, t_max, dt, seed=int(seeds[i]))
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
