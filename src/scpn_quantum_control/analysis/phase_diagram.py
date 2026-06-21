# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Diagram
"""Quantum Kuramoto phase diagram: K_c vs effective temperature.

Maps the synchronization-decoherence phase boundary for the
XY-Kuramoto model on a finite graph. Three regimes:

    1. Incoherent (K < K_c): no synchronization, R → 0
    2. Partially synchronised (K ≈ K_c): BKT transition, vortex unbinding
    3. Fully synchronised (K >> K_c): R → 1, entanglement saturated

The effective temperature T_eff combines frequency heterogeneity
(classical thermal noise analog) and quantum decoherence (T1, T2).
The critical coupling K_c(T_eff) defines the phase boundary.

For classical Kuramoto on complete graph:
    K_c = 2 / (π · g(0))
where g(0) is the frequency distribution density at ω = 0.

For finite graph with Laplacian spectrum:
    K_c = Δω / λ_2(L)
where Δω is the frequency spread and λ_2 is the Fiedler value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bkt_analysis import fiedler_eigenvalue


@dataclass
class PhaseBoundary:
    """Phase boundary point: K_c at given T_eff."""

    k_critical: float
    t_eff: float
    regime: str  # "incoherent", "bkt", "synchronized"


@dataclass
class PhaseDiagramResult:
    """Full phase diagram scan result."""

    k_values: NDArray[np.float64]
    t_eff_values: NDArray[np.float64]
    order_parameter: NDArray[np.float64]  # R(K, T_eff) matrix
    k_critical_curve: NDArray[np.float64]  # K_c(T_eff) boundary
    t_eff_grid: NDArray[np.float64]
    bkt_temperature: float
    classical_k_c: float
    quantum_k_c: float  # K_c with decoherence correction


def critical_coupling_finite_graph(
    omega: NDArray[np.float64],
    fiedler: float,
) -> float:
    """Critical coupling for synchronization on finite graph.

    K_c = Δω / λ_2 where Δω = max(ω) - min(ω) and λ_2 is the Fiedler value.
    """
    if not np.isfinite(fiedler):
        raise ValueError("fiedler must be finite")
    if fiedler < 0.0:
        raise ValueError("fiedler must be non-negative")
    delta_omega = float(np.max(omega) - np.min(omega))
    if delta_omega <= 1e-15:
        return 0.0
    if fiedler <= 1e-15:
        return float("inf")
    return delta_omega / fiedler


def critical_coupling_mean_field(omega: NDArray[np.float64]) -> float:
    """Classical Kuramoto K_c for complete graph: K_c = 2 / (π · g(0)).

    Approximates g(0) from the empirical frequency distribution using
    kernel density estimation with Scott's bandwidth.
    """
    n = len(omega)
    if n < 2:
        return 0.0
    std = float(np.std(omega))
    if std < 1e-15:
        return 0.0
    # KDE at ω=0 with Gaussian kernel, Scott bandwidth
    bw = 1.06 * std * n ** (-0.2)
    mean_omega = float(np.mean(omega))
    g0 = float(np.sum(np.exp(-0.5 * ((omega - mean_omega) / bw) ** 2))) / (
        n * bw * np.sqrt(2 * np.pi)
    )
    return float(2.0 / (np.pi * max(g0, 1e-15)))


def decoherence_temperature(t1: float, t2: float) -> float:
    """Effective temperature contribution from quantum decoherence.

    Maps T1/T2 decoherence rates to an effective thermal noise scale.
    T_decoherence ~ ħ / (k_B · T2) in natural units → 1/T2.
    """
    if np.isnan(t1):
        raise ValueError("t1 must be finite or infinite")
    if np.isnan(t2):
        raise ValueError("t2 must be finite or infinite")
    if t1 <= 0.0:
        raise ValueError("t1 must be positive")
    if t2 <= 0.0:
        raise ValueError("t2 must be positive")
    gamma_1 = 0.0 if np.isinf(t1) else 1.0 / t1
    gamma_2 = 0.0 if np.isinf(t2) else 1.0 / t2
    return float(gamma_1 + gamma_2) / 2.0


def effective_temperature(
    omega: NDArray[np.float64],
    t1: float = np.inf,
    t2: float = np.inf,
) -> float:
    """Combined effective temperature: frequency disorder + decoherence.

    T_eff = σ_ω (classical noise) + T_decoherence (quantum noise).
    """
    t_classical = float(np.std(omega))
    t_quantum = decoherence_temperature(t1, t2)
    return t_classical + t_quantum


def order_parameter_steady_state(
    K_coupling: float,
    k_critical: float,
) -> float:
    """Steady-state Kuramoto order parameter R for K > K_c.

    R = sqrt(1 - K_c / K) for K > K_c, else R = 0.
    Mean-field result; Strogatz 2000.
    """
    if K_coupling <= k_critical or k_critical <= 0:
        return 0.0
    return float(np.sqrt(1.0 - k_critical / K_coupling))


def compute_phase_diagram(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    k_range: tuple[float, float] = (0.01, 2.0),
    n_k: int = 50,
    t2_range: tuple[float, float] = (1.0, 1000.0),
    n_t: int = 40,
    t1_factor: float = 2.0,
) -> PhaseDiagramResult:
    """Compute the K vs T_eff phase diagram.

    Scans coupling strength K_base and decoherence time T2 to map
    the synchronization phase boundary.

    Args:
        K: coupling matrix (used for Fiedler value)
        omega: natural frequencies
        k_range: (min, max) coupling strength scan
        n_k: number of coupling points
        t2_range: (min, max) T2 decoherence time scan
        n_t: number of temperature points
        t1_factor: T1 = t1_factor × T2
    """
    fiedler = fiedler_eigenvalue(K)
    k_c_classical = critical_coupling_finite_graph(omega, fiedler)

    k_values = np.linspace(k_range[0], k_range[1], n_k, dtype=np.float64)
    t2_values = np.logspace(np.log10(t2_range[0]), np.log10(t2_range[1]), n_t)

    t_eff_grid = np.array(
        [effective_temperature(omega, t1=t2 * t1_factor, t2=t2) for t2 in t2_values]
    )

    # R(K, T_eff) matrix
    R_matrix = np.zeros((n_k, n_t))
    k_critical_curve = np.zeros(n_t)

    for j, _t2 in enumerate(t2_values):
        t_eff = t_eff_grid[j]
        # Decoherence increases effective K_c
        k_c_eff = k_c_classical * (1.0 + t_eff / max(float(np.std(omega)), 1e-15))
        k_critical_curve[j] = k_c_eff
        for i, k_val in enumerate(k_values):
            R_matrix[i, j] = order_parameter_steady_state(k_val, k_c_eff)

    # BKT temperature from existing analysis
    from .bkt_analysis import estimate_t_bkt

    t_bkt = estimate_t_bkt(K)

    # Quantum K_c includes decoherence at typical T2
    t_eff_typical = effective_temperature(omega, t1=200.0, t2=100.0)
    k_c_quantum = k_c_classical * (1.0 + t_eff_typical / max(float(np.std(omega)), 1e-15))

    return PhaseDiagramResult(
        k_values=k_values,
        t_eff_values=t_eff_grid,
        order_parameter=R_matrix,
        k_critical_curve=k_critical_curve,
        t_eff_grid=t_eff_grid,
        bkt_temperature=t_bkt,
        classical_k_c=k_c_classical,
        quantum_k_c=k_c_quantum,
    )
