# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Persistent Homology
"""Persistent homology of oscillator phase configurations.

THIS is the correct way to compute p_h1. Not from BKT universals
acting on the order parameter, but from direct topological analysis
of the phase field.

For n oscillators with phases theta_i, build a distance matrix:
    d_ij = 1 - cos(theta_j - theta_i)  (phase distance in [0, 2])

Apply Vietoris-Rips persistent homology via ripser. Count H1
(1-dimensional cycles = vortices) that persist above a threshold.

p_h1 = (number of persistent H1 features) / (maximum possible H1)

The maximum possible H1 for n points is n(n-1)/2 - (n-1) = (n-1)(n-2)/2
(complete graph minus spanning tree).

The key question: at what temperature does p_h1 = 0.72 emerge
from MC phase configurations on the K_nm graph?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ripser import ripser  # type: ignore[import-untyped]

    _RIPSER_AVAILABLE = True
except ImportError:
    _RIPSER_AVAILABLE = False


@dataclass
class PersistenceResult:
    """Persistent homology result for a phase configuration."""

    n_h0: int  # connected components
    n_h1: int  # 1-cycles (vortices)
    p_h1: float  # H1 fraction
    persistence_h1: list[float]  # lifetimes of H1 features
    n_oscillators: int


def phase_distance_matrix(theta: np.ndarray) -> np.ndarray:
    """Build phase distance matrix: d_ij = 1 - cos(theta_j - theta_i)."""
    n = len(theta)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - np.cos(theta[j] - theta[i])
            D[i, j] = D[j, i] = d
    result: np.ndarray = D
    return result


def compute_persistence(
    theta: np.ndarray,
    persistence_threshold: float = 0.1,
) -> PersistenceResult:
    """Compute persistent homology of a phase configuration.

    Args:
        theta: oscillator phases (n,)
        persistence_threshold: minimum lifetime to count as persistent
    """
    if not _RIPSER_AVAILABLE:
        raise ImportError("ripser not installed: pip install ripser")

    n = len(theta)
    D = phase_distance_matrix(theta)
    result = ripser(D, maxdim=1, distance_matrix=True)

    # H0: connected components (always n-1 finite bars + 1 infinite)
    h0_dgm = result["dgms"][0]
    n_h0 = len(h0_dgm)

    # H1: 1-cycles (vortices)
    h1_dgm = result["dgms"][1]
    persistence_h1 = [
        float(death - birth) for birth, death in h1_dgm if death - birth > persistence_threshold
    ]
    n_h1 = len(persistence_h1)

    # Maximum possible H1 for n points: (n-1)(n-2)/2
    max_h1 = max((n - 1) * (n - 2) // 2, 1)
    p_h1 = n_h1 / max_h1

    return PersistenceResult(
        n_h0=n_h0,
        n_h1=n_h1,
        p_h1=float(p_h1),
        persistence_h1=persistence_h1,
        n_oscillators=n,
    )


def p_h1_vs_temperature(
    K: np.ndarray,
    t_range: tuple[float, float] = (0.01, 0.5),
    n_temps: int = 20,
    n_thermalize: int = 5000,
    n_samples: int = 50,
    persistence_threshold: float = 0.1,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Measure p_h1 from persistent homology across temperatures.

    At each temperature, run MC, sample phase configurations,
    compute persistent homology, average p_h1.

    THIS is the correct way to find where p_h1 = 0.72.
    """
    if not _RIPSER_AVAILABLE:
        raise ImportError("ripser not installed: pip install ripser")

    from .monte_carlo_xy import _mc_sweep

    n = K.shape[0]
    temps = np.linspace(t_range[0], t_range[1], n_temps)

    results: dict[str, list[float]] = {
        "temperature": [],
        "p_h1_mean": [],
        "p_h1_std": [],
        "n_h1_mean": [],
    }

    for temp in temps:
        beta = 1.0 / max(temp, 1e-15)
        rng = np.random.default_rng(seed + int(temp * 1000))
        theta = np.asarray(rng.uniform(0, 2 * np.pi, n))

        # Thermalise
        for _ in range(n_thermalize):
            theta = _mc_sweep(theta, K, beta, rng)

        # Sample and measure
        p_h1_samples: list[float] = []
        n_h1_samples: list[float] = []
        for _ in range(n_samples):
            for _ in range(10):  # decorrelation sweeps
                theta = _mc_sweep(theta, K, beta, rng)
            pr = compute_persistence(theta, persistence_threshold)
            p_h1_samples.append(pr.p_h1)
            n_h1_samples.append(float(pr.n_h1))

        results["temperature"].append(float(temp))
        results["p_h1_mean"].append(float(np.mean(p_h1_samples)))
        results["p_h1_std"].append(float(np.std(p_h1_samples)))
        results["n_h1_mean"].append(float(np.mean(n_h1_samples)))

    return results
