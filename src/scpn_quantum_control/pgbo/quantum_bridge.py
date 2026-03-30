# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Bridge
"""Quantum PGBO: phase-geometry bridge operator.

The classical PGBO computes the metric tensor h_μν that maps
phase differences to geometric distances. In the quantum model,
h_μν is extracted from the quantum geometric tensor (QGT):

    Q_μν = <∂_μ ψ|∂_ν ψ> - <∂_μ ψ|ψ><ψ|∂_ν ψ>

where ∂_μ = ∂/∂K_μ (derivative w.r.t. coupling parameters).

The real part of Q_μν is the Fubini-Study metric (quantum distance),
the imaginary part is the Berry curvature (geometric phase).

For the Kuramoto-XY Hamiltonian parameterised by K_ij:
    h_μν^quantum = Re(Q_μν)  — quantum metric
    F_μν^quantum = -2 Im(Q_μν) — Berry curvature

The PGBO tensor connects phase dynamics to geometry:
    - Large h_μν: small parameter change → large state change (sensitive)
    - Large F_μν: parameter loop → geometric phase (topological)

Computed via parameter-shift: ∂|ψ>/∂K_ij ≈ (|ψ(K+ε)> - |ψ(K-ε)>) / 2ε
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hardware.classical import classical_exact_diag


@dataclass
class PGBOResult:
    """Quantum PGBO tensor result."""

    metric_tensor: np.ndarray  # h_μν = Re(Q_μν), n_params × n_params
    berry_curvature: np.ndarray  # F_μν = -2 Im(Q_μν)
    metric_determinant: float  # det(h) — volume element
    total_curvature: float  # Σ |F_μν|
    n_parameters: int
    parameter_labels: list[str]


def _ground_state(K: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Get ground state vector."""
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    result: np.ndarray = np.ascontiguousarray(exact["ground_state"])
    return result


def compute_pgbo_tensor(
    K: np.ndarray,
    omega: np.ndarray,
    epsilon: float = 0.005,
) -> PGBOResult:
    """Compute the quantum geometric tensor Q_μν for K_ij parameters.

    Parameters are the upper-triangle entries of K.
    """
    n = K.shape[0]
    psi_0 = _ground_state(K, omega)

    # Parameter list: upper triangle of K
    params: list[tuple[int, int]] = []
    labels: list[str] = []
    for i in range(n):
        for j in range(i + 1, n):
            params.append((i, j))
            labels.append(f"K_{i}{j}")

    n_params = len(params)
    dpsi: np.ndarray = np.zeros((n_params, len(psi_0)), dtype=complex)

    # Compute ∂|ψ>/∂K_ij via finite differences
    for mu, (i, j) in enumerate(params):
        K_plus = K.copy()
        K_plus[i, j] += epsilon
        K_plus[j, i] += epsilon
        K_minus = K.copy()
        K_minus[i, j] -= epsilon
        K_minus[j, i] -= epsilon

        psi_plus = _ground_state(K_plus, omega)
        psi_minus = _ground_state(K_minus, omega)

        # Fix phase ambiguity: align with reference
        psi_plus *= np.exp(-1j * np.angle(np.dot(psi_0.conj(), psi_plus)))
        psi_minus *= np.exp(-1j * np.angle(np.dot(psi_0.conj(), psi_minus)))

        dpsi[mu] = (psi_plus - psi_minus) / (2.0 * epsilon)

    # Quantum geometric tensor: Q_μν = <∂_μ|∂_ν> - <∂_μ|ψ><ψ|∂_ν>
    Q: np.ndarray = np.zeros((n_params, n_params), dtype=complex)
    for mu in range(n_params):
        for nu in range(n_params):
            inner = np.dot(dpsi[mu].conj(), dpsi[nu])
            proj = np.dot(dpsi[mu].conj(), psi_0) * np.dot(psi_0.conj(), dpsi[nu])
            Q[mu, nu] = inner - proj

    # Extract metric and curvature
    metric = np.real(Q)
    curvature = -2.0 * np.imag(Q)

    # Make metric symmetric (should be by construction, enforce numerically)
    metric = (metric + metric.T) / 2.0

    det_metric = float(np.linalg.det(metric)) if n_params > 0 else 0.0
    total_curv = float(np.sum(np.abs(curvature)))

    return PGBOResult(
        metric_tensor=metric,
        berry_curvature=curvature,
        metric_determinant=det_metric,
        total_curvature=total_curv,
        n_parameters=n_params,
        parameter_labels=labels,
    )
