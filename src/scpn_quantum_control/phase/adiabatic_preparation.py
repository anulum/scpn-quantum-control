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

In finite-size scans near a Berezinskii-Kosterlitz-Thouless transition,
the spectral gap can become very small compared with conventional
second-order critical paths. This module measures the finite-size
instantaneous gap and fidelity along a chosen dense exact path; it does not
prove an asymptotic BKT scaling law by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation


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
    *,
    max_dense_gib: float | None = None,
) -> AdiabaticResult:
    """Adiabatic preparation: ramp K from 0 to K_target over time T_total.

    Linear schedule: K(t) = K_target * t / T_total.
    """
    n = len(omega)
    require_dense_allocation(
        n,
        rank=2,
        object_count=5,
        max_gib=max_dense_gib,
        label="adiabatic dense eigensolver/evolution workspace",
    )
    require_dense_allocation(
        n,
        rank=1,
        object_count=3,
        max_gib=max_dense_gib,
        label="adiabatic dense state workspace",
    )
    dt = T_total / n_steps

    # Initial state: ground state of H(K≈0) — product state
    # Use small K to avoid degeneracy at K=0
    K_init = 0.01 * K_topology
    H_init = knm_to_dense_matrix(K_init, omega, max_dense_gib=max_dense_gib)
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
        H_mat = knm_to_dense_matrix(K_mat, omega, max_dense_gib=max_dense_gib)

        # Evolve: |ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩
        U = expm(-1j * H_mat * dt)
        psi = U @ psi

        # Current ground state and gap
        K_now = K_schedule[step + 1]
        K_now_mat = K_now * K_topology
        H_now = knm_to_dense_matrix(K_now_mat, omega, max_dense_gib=max_dense_gib)
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
    *,
    max_dense_gib: float | None = None,
) -> dict[str, list[float]]:
    """Scan adiabatic time vs final fidelity.

    Near very small finite-size gaps, fidelity can improve slowly with total
    ramp time. This scan reports the observed finite-size trend for the given
    system and schedule; it is not an asymptotic BKT scaling proof.
    """
    if T_values is None:
        T_values = np.array([1.0, 2.0, 5.0, 10.0, 20.0])

    results: dict[str, list[float]] = {
        "T_total": [],
        "final_fidelity": [],
        "min_gap": [],
    }

    for T in T_values:
        ar = adiabatic_ramp(
            omega,
            K_topology,
            K_target,
            float(T),
            n_steps_per_T,
            max_dense_gib=max_dense_gib,
        )
        results["T_total"].append(float(T))
        results["final_fidelity"].append(ar.final_fidelity)
        results["min_gap"].append(ar.min_gap)

    return results
