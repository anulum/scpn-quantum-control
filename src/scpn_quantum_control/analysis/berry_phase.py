# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Berry Phase
"""Finite-size Berry diagnostics for exact Kuramoto-XY ground-state scans.

The Berry (geometric) phase γ = -Im ∮ ⟨ψ(λ)|∂_λ ψ(λ)⟩ dλ measures
the geometry of the ground-state manifold in parameter space. Along the
one-dimensional open K scan used here, the accumulated connection is
gauge-dependent; the fidelity and fidelity susceptibility are the primary
gauge-invariant diagnostics.

Near a finite-size avoided crossing or transition proxy, the ground state can
change character rapidly. This module reports exact dense finite-size
diagnostics for that behaviour; it does not by itself prove an asymptotic BKT
singularity or an exhaustive literature boundary.

We compute:
1. Berry connection A(K) = -Im⟨ψ(K)|∂_K ψ(K)⟩
   (approximated as -Im⟨ψ(K)|ψ(K+dK)⟩/dK)
2. Berry curvature F(K) = dA/dK (derivative of connection)
3. Accumulated phase γ(K) = ∫_0^K A(K') dK'
4. Fidelity susceptibility χ_F = -2 ln|⟨ψ(K)|ψ(K+dK)⟩|/dK²
   (diverges at K_c, related to QFI)

Prior art includes geometric-phase probes, fidelity susceptibility, and
quantum-synchronisation diagnostics. This module applies those finite-size
diagnostics to the Kuramoto-XY Hamiltonian implemented in this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation, require_dense_eigensolver_workspace


@dataclass
class BerryPhaseResult:
    """Berry phase analysis across coupling strength."""

    k_values: np.ndarray
    berry_connection: np.ndarray  # A(K) at each midpoint
    berry_curvature: np.ndarray  # F(K) = dA/dK
    accumulated_phase: np.ndarray  # γ(K) = cumulative integral
    fidelity: np.ndarray  # |⟨ψ(K)|ψ(K+dK)⟩| at each step
    fidelity_susceptibility: np.ndarray  # χ_F at each midpoint
    spectral_gap: np.ndarray  # gap at each K
    curvature_peak_k: float | None  # K where |F| is maximum


def _ground_state(
    K: np.ndarray,
    omega: np.ndarray,
    *,
    max_dense_gib: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return (ground state vector, spectral gap)."""
    n = len(omega)
    require_dense_eigensolver_workspace(
        n,
        max_gib=max_dense_gib,
        label="Berry phase dense eigensolver workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0 = np.ascontiguousarray(eigenvectors[:, 0])
    gap = float(eigenvalues[1] - eigenvalues[0])
    return psi0, gap


def _fix_gauge(psi: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    """Fix the U(1) gauge freedom by maximizing Re⟨psi_ref|psi⟩.

    Eigenvectors have arbitrary overall phase. This aligns
    consecutive ground states so the Berry connection is smooth.
    """
    overlap = np.vdot(psi_ref, psi)
    if abs(overlap) < 1e-15:
        return psi
    phase = overlap / abs(overlap)
    result: np.ndarray = psi / phase
    return result


def berry_phase_scan(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
    *,
    max_dense_gib: float | None = None,
) -> BerryPhaseResult:
    """Compute Berry connection, curvature, and fidelity across K.

    K_topology: normalized coupling matrix (max=1), scaled by k_range values.
    max_dense_gib: optional GiB budget for dense eigensolver and retained states.
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 30)

    n = len(omega)
    n_k = len(k_range)
    require_dense_allocation(
        n,
        rank=1,
        object_count=max(n_k, 1),
        max_gib=max_dense_gib,
        label="Berry phase retained ground-state workspace",
    )
    gaps = np.zeros(n_k)

    # Compute all ground states
    states = []
    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        psi, gap = _ground_state(K, omega, max_dense_gib=max_dense_gib)
        gaps[idx] = gap
        states.append(psi)

    # Gauge-fix: align each state to the previous one
    for i in range(1, n_k):
        states[i] = _fix_gauge(states[i], states[i - 1])

    # Berry connection: A(K_i) ≈ -Im⟨ψ_i|ψ_{i+1}⟩ / dK
    n_mid = n_k - 1
    connection = np.zeros(n_mid)
    fidelity = np.zeros(n_mid)
    fid_susceptibility = np.zeros(n_mid)

    for i in range(n_mid):
        dk = k_range[i + 1] - k_range[i]
        overlap = np.vdot(states[i], states[i + 1])
        fidelity[i] = abs(overlap)
        connection[i] = -np.imag(np.log(overlap)) / dk if abs(overlap) > 1e-15 else 0.0

        # Fidelity susceptibility: χ_F = -2 ln|F| / dK²
        if fidelity[i] > 1e-15 and dk > 1e-15:
            fid_susceptibility[i] = -2.0 * np.log(fidelity[i]) / dk**2
        else:
            fid_susceptibility[i] = 0.0

    # Berry curvature: F = dA/dK (finite difference)
    if n_mid > 1:
        curvature: np.ndarray = np.gradient(connection, k_range[:n_mid])
    else:
        curvature = np.zeros(n_mid)

    # Accumulated phase: γ(K) = integral of A
    k_mid = (k_range[:-1] + k_range[1:]) / 2
    dk_arr = np.diff(k_range)
    accumulated: np.ndarray = np.cumsum(connection * dk_arr)

    # Peak curvature location
    peak_k = None
    if n_mid > 0:
        peak_idx = int(np.argmax(np.abs(curvature)))
        peak_k = float(k_mid[peak_idx])

    return BerryPhaseResult(
        k_values=k_mid,
        berry_connection=connection,
        berry_curvature=curvature,
        accumulated_phase=accumulated,
        fidelity=fidelity,
        fidelity_susceptibility=fid_susceptibility,
        spectral_gap=gaps[:n_mid],
        curvature_peak_k=peak_k,
    )
