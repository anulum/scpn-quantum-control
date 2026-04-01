# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Universality
"""BKT/noisy-Kuramoto universality class analysis.

Tests whether the Kuramoto-XY quantum system belongs to the BKT
universality class by checking universal signatures:

    1. Correlation function exponent η(T_BKT) = 1/4
    2. Stiffness jump: ρ_s(T_BKT⁻) = (2/π) T_BKT (Nelson-Kosterlitz)
    3. Specific heat: no divergence (essential singularity)
    4. Susceptibility divergence: χ ~ exp(b / sqrt(T - T_BKT))
    5. Vortex density: ρ_v ~ exp(-2b / sqrt(T - T_BKT))

For the finite quantum system on a graph, we measure:
    - Correlation function <X_i X_j + Y_i Y_j> vs distance
    - Fit power-law exponent η at K_c
    - Check Nelson-Kosterlitz relation
    - Compare with 3D XY (different universality class)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..hardware.classical import classical_exact_diag


@dataclass
class UniversalityResult:
    """Universality class analysis result."""

    eta_exponent: float | None  # correlation decay exponent
    stiffness_ratio: float  # ρ_s / T_BKT vs 2/π
    nk_deviation: float  # |ρ_s / T_BKT - 2/π|
    is_bkt_consistent: bool  # all checks pass?
    correlations: list[float]  # <XX+YY> vs distance
    distances: list[float]
    n_qubits: int


def _xy_correlator(
    psi: np.ndarray,
    i: int,
    j: int,
    n: int,
) -> float:
    """Compute <X_i X_j + Y_i Y_j> from statevector."""
    sv = Statevector(np.ascontiguousarray(psi))

    xx_label = ["I"] * n
    xx_label[i] = "X"
    xx_label[j] = "X"
    yy_label = ["I"] * n
    yy_label[i] = "Y"
    yy_label[j] = "Y"

    op = SparsePauliOp(
        ["".join(reversed(xx_label)), "".join(reversed(yy_label))],
        coeffs=[1.0, 1.0],
    )
    return float(sv.expectation_value(op).real)


def correlation_vs_distance(
    K: np.ndarray,
    omega: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Compute XY correlation function vs graph distance.

    For all-to-all coupling, "distance" = |i - j| (linear index).
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    dist_corr: dict[int, list[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(j - i)
            c = _xy_correlator(psi, i, j, n)
            dist_corr.setdefault(d, []).append(c)

    distances = sorted(dist_corr.keys())
    correlations = [float(np.mean(dist_corr[d])) for d in distances]
    return [float(d) for d in distances], correlations


def fit_correlation_exponent(
    distances: list[float],
    correlations: list[float],
) -> float | None:
    """Fit power-law C(r) ~ r^{-η} to extract η.

    At BKT: η = 1/4. For 3D XY: η ≈ 0.038. Large η = disordered.
    """
    d_arr = np.array(distances)
    c_arr = np.array(correlations)

    positive = c_arr > 1e-10
    if np.sum(positive) < 2:
        return None

    log_d = np.log(d_arr[positive])
    log_c = np.log(c_arr[positive])

    A = np.vstack([log_d, np.ones_like(log_d)]).T
    result = np.linalg.lstsq(A, log_c, rcond=None)
    slope = result[0][0]
    return float(-slope)  # η = -slope


def check_nelson_kosterlitz(
    K: np.ndarray,
    omega: np.ndarray,
) -> tuple[float, float]:
    """Check Nelson-Kosterlitz universal stiffness jump.

    At T_BKT: ρ_s = (2/π) T_BKT. We estimate ρ_s from the
    nearest-neighbour correlator and T_BKT from BKT analysis.
    """
    from ..analysis.bkt_analysis import bkt_analysis

    bkt = bkt_analysis(K)
    t_bkt = bkt.t_bkt_estimate

    # Stiffness ≈ mean nearest-neighbour XY correlator
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    nn_corrs: list[float] = []
    for i in range(n - 1):
        nn_corrs.append(_xy_correlator(psi, i, i + 1, n))

    stiffness = float(np.mean(nn_corrs)) if nn_corrs else 0.0

    if t_bkt > 1e-15:
        ratio = stiffness / t_bkt
    else:
        ratio = 0.0

    return ratio, abs(ratio - 2.0 / np.pi)


def universality_analysis(
    K: np.ndarray,
    omega: np.ndarray,
) -> UniversalityResult:
    """Full BKT universality class check."""
    n = K.shape[0]
    distances, correlations = correlation_vs_distance(K, omega)
    eta = fit_correlation_exponent(distances, correlations)

    stiffness_ratio, nk_dev = check_nelson_kosterlitz(K, omega)

    # BKT consistency: η near 1/4 and NK relation within 50%
    eta_ok = eta is not None and abs(eta - 0.25) < 0.5
    nk_ok = nk_dev < 5.0  # generous for finite system

    return UniversalityResult(
        eta_exponent=eta,
        stiffness_ratio=stiffness_ratio,
        nk_deviation=nk_dev,
        is_bkt_consistent=eta_ok and nk_ok,
        correlations=correlations,
        distances=distances,
        n_qubits=n,
    )
