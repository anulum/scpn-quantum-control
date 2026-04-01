# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Lindblad Ness
"""Non-equilibrium steady state (NESS) of driven-dissipative Kuramoto-XY.

Computes the NESS under Lindblad dynamics: unitary H(K) + amplitude
damping. The NESS is what you'd see on real quantum hardware after
the system equilibrates with its environment.

Compares NESS order parameter R_NESS to ideal ground state R_ideal
across coupling strength K. The gap between them quantifies how
much noise destroys synchronization.

Prior art: NESS of driven-dissipative quantum sync studied for
2-3 oscillators (Roulet, Entropy 2024). Many-body heterogeneous-
frequency NESS is missing from the literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from .quantum_mpemba import _R_from_density_matrix


@dataclass
class NESSResult:
    """NESS computation result at a single coupling strength."""

    K_base: float
    rho_ness: np.ndarray
    R_ness: float
    R_ideal: float
    purity: float  # Tr(ρ²) — 1 for pure, 1/d for maximally mixed
    gap_R: float  # R_ideal - R_ness


@dataclass
class NESSScanResult:
    """NESS scan across coupling strength."""

    k_values: np.ndarray
    R_ness: np.ndarray
    R_ideal: np.ndarray
    purity: np.ndarray
    gap_R: np.ndarray  # R_ideal - R_ness at each K
    noise_resilience: float  # K where gap_R is minimized


def _lindblad_superoperator(H: np.ndarray, gamma: float, n_qubits: int) -> np.ndarray:
    """Lindblad superoperator with amplitude damping."""
    dim = H.shape[0]
    I_d = np.eye(dim)
    L_total: np.ndarray = -1j * (np.kron(H, I_d) - np.kron(I_d, H.T))

    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    for q in range(n_qubits):
        L_k = np.eye(1, dtype=complex)
        for j in range(n_qubits):
            L_k = np.kron(L_k, lowering if j == q else np.eye(2))
        L_k *= np.sqrt(gamma)

        L_total += (
            np.kron(L_k, L_k.conj())
            - 0.5 * np.kron(L_k.conj().T @ L_k, I_d)
            - 0.5 * np.kron(I_d, L_k.T @ L_k.conj())
        )
    return L_total


def _R_from_statevector(psi: np.ndarray, n: int) -> float:
    """R from pure state."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(np.ascontiguousarray(psi))
    phases = np.zeros(n)
    for k in range(n):
        x_str = ["I"] * n
        x_str[k] = "X"
        y_str = ["I"] * n
        y_str[k] = "Y"
        ex = float(sv.expectation_value(SparsePauliOp("".join(reversed(x_str)))).real)
        ey = float(sv.expectation_value(SparsePauliOp("".join(reversed(y_str)))).real)
        phases[k] = np.arctan2(ey, ex)
    return float(abs(np.mean(np.exp(1j * phases))))


def compute_ness(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
    gamma: float = 0.1,
    t_relax: float = 50.0,
) -> NESSResult:
    """Compute NESS at a single coupling strength."""
    n = len(omega)
    dim = 2**n
    K = K_base * K_topology

    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega)
    L_super = _lindblad_superoperator(H_mat, gamma, n)

    # NESS: evolve |0⟩ for long time
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[0] = 1.0
    rho_init = np.outer(psi_0, psi_0.conj())
    vec_ss = expm(L_super * t_relax) @ rho_init.flatten()
    rho_ness = vec_ss.reshape(dim, dim)
    rho_ness = (rho_ness + rho_ness.conj().T) / 2
    rho_ness /= np.trace(rho_ness)

    R_ness = _R_from_density_matrix(rho_ness, n)
    purity = float(np.real(np.trace(rho_ness @ rho_ness)))

    # Ideal: ground state R
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi_gs = eigenvectors[:, 0]
    R_ideal = _R_from_statevector(psi_gs, n)

    return NESSResult(
        K_base=K_base,
        rho_ness=rho_ness,
        R_ness=R_ness,
        R_ideal=R_ideal,
        purity=purity,
        gap_R=R_ideal - R_ness,
    )


def ness_vs_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    gamma: float = 0.1,
) -> NESSScanResult:
    """Scan NESS across coupling strength."""
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15)

    n_k = len(k_range)
    R_ness = np.zeros(n_k)
    R_ideal = np.zeros(n_k)
    purity = np.zeros(n_k)
    gap = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        result = compute_ness(omega, K_topology, float(kb), gamma)
        R_ness[idx] = result.R_ness
        R_ideal[idx] = result.R_ideal
        purity[idx] = result.purity
        gap[idx] = result.gap_R

    # Noise resilience: K where gap is minimized
    resilience_k = float(k_range[int(np.argmin(np.abs(gap)))])

    return NESSScanResult(
        k_values=k_range,
        R_ness=R_ness,
        R_ideal=R_ideal,
        purity=purity,
        gap_R=gap,
        noise_resilience=resilience_k,
    )
