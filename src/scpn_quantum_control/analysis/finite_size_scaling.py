# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Finite Size Scaling
"""Finite-size scaling for K_c extraction from small quantum systems.

Extracts the thermodynamic-limit K_c from small-N exact diag data.
For BKT transitions, the gap minimum K_c(N) shifts with N due to
logarithmic corrections:

    K_c(N) = K_c(∞) + a / (log N)²

(standard BKT FSS ansatz, Nomura-Kitazawa 2002).

This module:
1. Computes K_c(N) from gap minimum for N = 2, 3, 4, 5 qubits
2. Fits the BKT FSS ansatz to extrapolate K_c(∞)
3. Also fits power-law K_c(N) = K_c(∞) + b/N^ν for comparison

Methods: Nomura-Kitazawa level spectroscopy (2002),
Hasenbusch-Pinn log-correction extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, knm_to_dense_matrix


@dataclass
class FSSResult:
    """Finite-size scaling result."""

    system_sizes: list[int]
    k_c_values: list[float]  # K_c(N) from gap minimum
    gap_min_values: list[float]  # minimum gap at each N
    k_c_extrapolated_bkt: float | None  # from BKT ansatz fit
    k_c_extrapolated_power: float | None  # from power-law fit


def _find_kc_from_gap(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray,
) -> tuple[float, float]:
    """Find K_c(N) = K_base where spectral gap is minimized.

    Returns (k_c, min_gap).
    """
    gaps = np.zeros(len(k_range))
    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topology
        H = knm_to_dense_matrix(K, omega)
        eigvals = np.linalg.eigvalsh(H)
        gaps[idx] = float(eigvals[1] - eigvals[0])

    min_idx = int(np.argmin(gaps))
    return float(k_range[min_idx]), float(gaps[min_idx])


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    result: np.ndarray = T
    return result


def finite_size_scaling(
    system_sizes: list[int] | None = None,
    k_range: np.ndarray | None = None,
) -> FSSResult:
    """Extract K_c from multiple system sizes and extrapolate.

    Uses ring topology with Paper 27 natural frequencies.
    """
    if system_sizes is None:
        system_sizes = [2, 3, 4]
    if k_range is None:
        k_range = np.linspace(0.3, 6.0, 20)

    k_c_list: list[float] = []
    gap_min_list: list[float] = []

    for n in system_sizes:
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        kc, gmin = _find_kc_from_gap(omega, T, k_range)
        k_c_list.append(kc)
        gap_min_list.append(gmin)

    # Fit BKT ansatz: K_c(N) = K_c(∞) + a / (log N)²
    k_c_bkt = _fit_bkt_ansatz(system_sizes, k_c_list)

    # Fit power-law: K_c(N) = K_c(∞) + b / N^ν
    k_c_power = _fit_power_ansatz(system_sizes, k_c_list)

    return FSSResult(
        system_sizes=system_sizes,
        k_c_values=k_c_list,
        gap_min_values=gap_min_list,
        k_c_extrapolated_bkt=k_c_bkt,
        k_c_extrapolated_power=k_c_power,
    )


def _fit_bkt_ansatz(sizes: list[int], k_c_vals: list[float]) -> float | None:
    """Fit K_c(N) = K_c(∞) + a / (log N)²."""
    if len(sizes) < 2:
        return None
    log_n_sq = np.array([1.0 / max(np.log(n) ** 2, 0.01) for n in sizes])
    k_c = np.array(k_c_vals)
    # Linear fit: k_c = K_c_inf + a * (1/log²N)
    A = np.column_stack([np.ones(len(sizes)), log_n_sq])
    try:
        result = np.linalg.lstsq(A, k_c, rcond=None)
        return float(result[0][0])
    except np.linalg.LinAlgError:
        return None


def _fit_power_ansatz(sizes: list[int], k_c_vals: list[float]) -> float | None:
    """Fit K_c(N) = K_c(∞) + b / N."""
    if len(sizes) < 2:
        return None
    inv_n = np.array([1.0 / n for n in sizes])
    k_c = np.array(k_c_vals)
    A = np.column_stack([np.ones(len(sizes)), inv_n])
    try:
        result = np.linalg.lstsq(A, k_c, rcond=None)
        return float(result[0][0])
    except np.linalg.LinAlgError:
        return None
