# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Adiabatic Preparation
"""Adiabatic state preparation for the synchronization ground state.

Start from the trivial ground state of H(K=0) (product state) and
slowly ramp K from 0 to K_target. Track:
1. Instantaneous fidelity F(t) = |⟨ψ(t)|ψ_gs(K(t))⟩|²
2. Spectral gap Δ(K) along the path
3. Minimum gap → controls adiabatic speed limit

At the BKT critical point K_c, the gap closes as
Δ ~ exp(-b/√(K-K_c)), an essential singularity.
This is qualitatively different from power-law gap closings
at standard (2nd-order) QPTs:
- 2nd order: Δ ~ |K-K_c|^(zν) → adiabatic time T ~ 1/Δ² ~ L^(2zν)
- BKT: Δ ~ exp(-b/√(K-K_c)) → T ~ exp(2b/√(K-K_c))
  → EXPONENTIALLY slow adiabatic preparation at K_c

Nobody has studied adiabatic preparation fidelity specifically
at BKT transitions with heterogeneous frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


@dataclass
class AdiabaticResult:
    """Adiabatic preparation result."""

    times: np.ndarray
    K_schedule: np.ndarray  # K(t) at each time
    fidelity: np.ndarray  # |⟨ψ(t)|ψ_gs(K(t))⟩|² at each time
    gap: np.ndarray  # spectral gap at each K(t)
    final_fidelity: float
    min_gap: float
    min_gap_K: float  # K value where gap is smallest


def adiabatic_ramp(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_target: float,
    T_total: float = 10.0,
    n_steps: int = 50,
) -> AdiabaticResult:
    """Adiabatic preparation: ramp K from 0 to K_target over time T_total.

    Linear schedule: K(t) = K_target * t / T_total.
    """
    dt = T_total / n_steps

    # Initial state: ground state of H(K≈0) — product state
    # Use small K to avoid degeneracy at K=0
    K_init = 0.01 * K_topology
    H_init = knm_to_dense_matrix(K_init, omega)
    eigvals_init, eigvecs_init = np.linalg.eigh(H_init)
    psi = np.ascontiguousarray(eigvecs_init[:, 0]).astype(complex)

    times = np.linspace(0, T_total, n_steps + 1)
    K_schedule = K_target * times / T_total
    fidelity = np.zeros(n_steps + 1)
    gaps = np.zeros(n_steps + 1)

    # Initial fidelity and gap
    fidelity[0] = 1.0
    gaps[0] = float(eigvals_init[1] - eigvals_init[0])

    for step in range(n_steps):
        # Hamiltonian at midpoint
        K_mid = K_target * (times[step] + dt / 2) / T_total
        K_mat = K_mid * K_topology
        H_mat = knm_to_dense_matrix(K_mat, omega)

        # Evolve: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩
        U = expm(-1j * H_mat * dt)
        psi = U @ psi

        # Current ground state and gap
        K_now = K_schedule[step + 1]
        K_now_mat = K_now * K_topology
        H_now = knm_to_dense_matrix(K_now_mat, omega)
        eigvals, eigvecs = np.linalg.eigh(H_now)
        psi_gs = eigvecs[:, 0]
        gaps[step + 1] = float(eigvals[1] - eigvals[0])

        # Fidelity with instantaneous ground state
        overlap = abs(np.vdot(psi_gs, psi)) ** 2
        fidelity[step + 1] = float(overlap)

    min_gap_idx = int(np.argmin(gaps))

    return AdiabaticResult(
        times=times,
        K_schedule=K_schedule,
        fidelity=fidelity,
        gap=gaps,
        final_fidelity=float(fidelity[-1]),
        min_gap=float(gaps[min_gap_idx]),
        min_gap_K=float(K_schedule[min_gap_idx]),
    )


def adiabatic_time_scaling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_target: float,
    T_values: np.ndarray | None = None,
    n_steps_per_T: int = 40,
) -> dict[str, list[float]]:
    """Scan adiabatic time vs final fidelity.

    For BKT: fidelity should improve exponentially slowly with T
    (compared to power-law for 2nd-order QPTs).
    """
    if T_values is None:
        T_values = np.array([1.0, 2.0, 5.0, 10.0, 20.0])

    results: dict[str, list[float]] = {
        "T_total": [],
        "final_fidelity": [],
        "min_gap": [],
    }

    for T in T_values:
        ar = adiabatic_ramp(omega, K_topology, K_target, float(T), n_steps_per_T)
        results["T_total"].append(float(T))
        results["final_fidelity"].append(ar.final_fidelity)
        results["min_gap"].append(ar.min_gap)

    return results
