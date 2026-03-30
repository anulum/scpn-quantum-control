# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Enaqt
"""Environment-Assisted Quantum Transport (ENAQT) optimisation.

ENAQT (Plenio & Huelga, NJP 10, 113019 (2008)) shows that optimal
quantum transport occurs at intermediate noise — neither purely
coherent nor fully classical. For the Kuramoto-XY system:

    - Zero noise: quantum interference can trap population (Anderson)
    - Optimal noise: dephasing breaks interference, enables transport
    - High noise: classical random walk, slow diffusion

The ENAQT optimal dephasing rate γ* maximises the Kuramoto order
parameter R(γ) after evolution time t:

    γ* = argmax_γ R(γ, t)

This connects to the SCPN consciousness threshold: the BKT transition
temperature may correspond to the ENAQT optimal noise level.

Method: scan dephasing rate γ, for each:
    1. Evolve state under H + noise (Lindblad dephasing channel)
    2. Compute R_global from noisy state
    3. Find the γ that maximises R
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class ENAQTResult:
    """ENAQT noise optimisation result."""

    optimal_gamma: float  # dephasing rate that maximises R
    optimal_r: float  # R_global at optimal γ
    gamma_values: np.ndarray
    r_values: np.ndarray
    coherent_r: float  # R at γ=0
    classical_r: float  # R at large γ
    enhancement: float  # optimal_r / coherent_r


def _lindblad_evolve(
    rho: np.ndarray,
    H_mat: np.ndarray,
    gamma: float,
    dt: float,
    n_qubits: int,
) -> np.ndarray:
    """One step of Lindblad master equation with Z-dephasing.

    dρ/dt = -i[H, ρ] + γ Σ_k (Z_k ρ Z_k - ρ)

    Uses first-order Euler for simplicity (dt should be small).
    """
    commutator = -1j * (H_mat @ rho - rho @ H_mat)

    # Dephasing: Z_k ρ Z_k - ρ for each qubit
    dephasing = np.zeros_like(rho)
    for k in range(n_qubits):
        Z_k = np.eye(1, dtype=complex)
        for q in range(n_qubits):
            if q == k:
                Z_k = np.kron(Z_k, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                Z_k = np.kron(Z_k, np.eye(2, dtype=complex))
        dephasing += Z_k @ rho @ Z_k - rho

    drho = commutator + gamma * dephasing
    rho_new = rho + dt * drho

    # Enforce trace = 1 and Hermiticity
    rho_new = (rho_new + rho_new.conj().T) / 2.0
    rho_new /= np.trace(rho_new)
    result: np.ndarray = rho_new
    return result


def _r_from_density_matrix(rho: np.ndarray, n_qubits: int) -> float:
    """Extract R_global from density matrix via X,Y expectations."""
    phases = np.zeros(n_qubits)
    for k in range(n_qubits):
        X_k = np.eye(1, dtype=complex)
        Y_k = np.eye(1, dtype=complex)
        for q in range(n_qubits):
            if q == k:
                X_k = np.kron(X_k, np.array([[0, 1], [1, 0]], dtype=complex))
                Y_k = np.kron(Y_k, np.array([[0, -1j], [1j, 0]], dtype=complex))
            else:
                X_k = np.kron(X_k, np.eye(2, dtype=complex))
                Y_k = np.kron(Y_k, np.eye(2, dtype=complex))
        exp_x = float(np.real(np.trace(rho @ X_k)))
        exp_y = float(np.real(np.trace(rho @ Y_k)))
        phases[k] = np.arctan2(exp_y, exp_x)

    z = np.mean(np.exp(1j * phases))
    return float(np.abs(z))


def enaqt_scan(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_range: np.ndarray | None = None,
    t_evolve: float = 1.0,
    n_steps: int = 50,
) -> ENAQTResult:
    """Scan dephasing rate to find ENAQT optimum.

    Args:
        K: coupling matrix
        omega: natural frequencies
        gamma_range: dephasing rates to scan
        t_evolve: evolution time
        n_steps: Lindblad time steps
    """
    n = K.shape[0]
    if gamma_range is None:
        gamma_range = np.logspace(-3, 1, 20)

    H_op = knm_to_hamiltonian(K, omega)
    H_raw = H_op.to_matrix()
    H_mat = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)

    dim = 2**n
    # Initial state: |+>^n (equal superposition for transport)
    psi_init = np.ones(dim, dtype=complex) / np.sqrt(dim)
    rho_init = np.outer(psi_init, psi_init.conj())

    dt = t_evolve / n_steps
    r_values = np.zeros(len(gamma_range))

    for idx, gamma in enumerate(gamma_range):
        rho = rho_init.copy()
        for _step in range(n_steps):
            rho = _lindblad_evolve(rho, H_mat, gamma, dt, n)
        r_values[idx] = _r_from_density_matrix(rho, n)

    best_idx = int(np.argmax(r_values))
    coherent_r = r_values[0] if len(r_values) > 0 else 0.0
    classical_r = r_values[-1] if len(r_values) > 0 else 0.0
    optimal_r = r_values[best_idx]
    enhancement = optimal_r / max(coherent_r, 1e-15)

    return ENAQTResult(
        optimal_gamma=float(gamma_range[best_idx]),
        optimal_r=optimal_r,
        gamma_values=gamma_range,
        r_values=r_values,
        coherent_r=coherent_r,
        classical_r=classical_r,
        enhancement=enhancement,
    )
