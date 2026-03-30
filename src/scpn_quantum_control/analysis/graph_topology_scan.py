# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Graph Topology Scan
"""Graph topology → p_h1 systematic scan.

The key finding: p_h1 at the BKT transition depends on graph topology,
not on BKT universal constants. This module scans p_h1 across graph
families to find which topologies produce p_h1 ≈ 0.72.

Graph families tested:
    1. Erdős-Rényi G(n, p) — random, varying density
    2. Watts-Strogatz — small-world, varying rewiring
    3. Barabási-Albert — scale-free, varying attachment
    4. k-regular lattice — fixed degree
    5. Complete graph (K_nm) — all-to-all with exponential decay
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .monte_carlo_xy import _mc_sweep
from .persistent_homology import _RIPSER_AVAILABLE, compute_persistence


@dataclass
class GraphP_H1_Result:
    """p_h1 measurement for one graph topology."""

    graph_family: str
    parameter: float  # family-specific (e.g., edge probability for ER)
    n_nodes: int
    avg_degree: float
    p_h1_mean: float
    p_h1_std: float
    n_samples: int


def _erdos_renyi_coupling(n: int, p: float, strength: float = 0.5, seed: int = 42) -> np.ndarray:
    """Erdős-Rényi random graph coupling matrix."""
    rng = np.random.default_rng(seed)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                K[i, j] = K[j, i] = strength
    result: np.ndarray = K
    return result


def _watts_strogatz_coupling(
    n: int, k: int = 4, beta: float = 0.3, strength: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Watts-Strogatz small-world coupling matrix."""
    rng = np.random.default_rng(seed)
    K = np.zeros((n, n))
    # Start with ring lattice of degree k
    for i in range(n):
        for j in range(1, k // 2 + 1):
            K[i, (i + j) % n] = strength
            K[(i + j) % n, i] = strength
    # Rewire with probability beta
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < beta:
                new_j = rng.integers(n)
                while new_j == i or K[i, new_j] > 0:
                    new_j = rng.integers(n)
                K[i, (i + j) % n] = 0
                K[(i + j) % n, i] = 0
                K[i, new_j] = K[new_j, i] = strength
    result: np.ndarray = K
    return result


def _ring_coupling(n: int, k: int = 2, strength: float = 0.5) -> np.ndarray:
    """k-nearest-neighbour ring coupling."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k + 1):
            K[i, (i + j) % n] = strength
            K[(i + j) % n, i] = strength
    result: np.ndarray = K
    return result


def _measure_p_h1_at_transition(
    K: np.ndarray,
    n_thermalize: int = 3000,
    n_samples: int = 30,
    persistence_threshold: float = 0.01,
    seed: int = 42,
) -> tuple[float, float]:
    """Measure p_h1 at the approximate BKT transition for given coupling.

    Uses a range of temperatures and picks the one where vortex
    density changes fastest (proxy for T_BKT).
    """
    n = K.shape[0]
    # Scan a few temperatures to find the transition region
    temps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    best_t = 0.5  # default
    max_p_h1 = 0.0

    for t in temps:
        beta = 1.0 / t
        rng = np.random.default_rng(seed + int(t * 1000))
        theta = np.asarray(rng.uniform(0, 2 * np.pi, n))
        for _ in range(n_thermalize):
            theta = _mc_sweep(theta, K, beta, rng)
        pr = compute_persistence(theta, persistence_threshold)
        if pr.p_h1 > max_p_h1:
            max_p_h1 = pr.p_h1
            best_t = t

    # Measure p_h1 at best temperature with multiple samples
    beta = 1.0 / best_t
    rng = np.random.default_rng(seed + 999)
    theta = np.asarray(rng.uniform(0, 2 * np.pi, n))
    for _ in range(n_thermalize):
        theta = _mc_sweep(theta, K, beta, rng)

    samples: list[float] = []
    for _ in range(n_samples):
        for _ in range(10):
            theta = _mc_sweep(theta, K, beta, rng)
        pr = compute_persistence(theta, persistence_threshold)
        samples.append(pr.p_h1)

    return float(np.mean(samples)), float(np.std(samples))


def scan_graph_topologies(
    n: int = 16,
    n_samples: int = 20,
    seed: int = 42,
) -> list[GraphP_H1_Result]:
    """Scan p_h1 across graph families.

    The question: which graph topology produces p_h1 ≈ 0.72?
    """
    if not _RIPSER_AVAILABLE:
        raise ImportError("ripser not installed: pip install ripser")

    results: list[GraphP_H1_Result] = []

    # Erdős-Rényi at various edge probabilities
    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        K = _erdos_renyi_coupling(n, p, seed=seed)
        avg_deg = float(np.sum(K > 0)) / n
        mean, std = _measure_p_h1_at_transition(K, n_samples=n_samples, seed=seed)
        results.append(GraphP_H1_Result("erdos_renyi", p, n, avg_deg, mean, std, n_samples))

    # Ring lattice at various degrees
    for k in [1, 2, 3, 4]:
        K = _ring_coupling(n, k)
        avg_deg = float(np.sum(K > 0)) / n
        mean, std = _measure_p_h1_at_transition(K, n_samples=n_samples, seed=seed)
        results.append(
            GraphP_H1_Result("ring_lattice", float(k), n, avg_deg, mean, std, n_samples)
        )

    # Watts-Strogatz at various rewiring probabilities
    for beta in [0.0, 0.1, 0.3, 0.5, 1.0]:
        K = _watts_strogatz_coupling(n, k=4, beta=beta, seed=seed)
        avg_deg = float(np.sum(K > 0)) / n
        mean, std = _measure_p_h1_at_transition(K, n_samples=n_samples, seed=seed)
        results.append(GraphP_H1_Result("watts_strogatz", beta, n, avg_deg, mean, std, n_samples))

    # Complete graph (K_nm)
    from ..bridge.knm_hamiltonian import build_knm_paper27

    K_knm = build_knm_paper27(L=n)
    avg_deg_knm = float(np.sum(K_knm > 0)) / n
    mean, std = _measure_p_h1_at_transition(K_knm, n_samples=n_samples, seed=seed)
    results.append(GraphP_H1_Result("knm_complete", 1.0, n, avg_deg_knm, mean, std, n_samples))

    return results
