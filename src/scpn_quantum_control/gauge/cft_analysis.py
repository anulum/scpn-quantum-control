# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cft Analysis
"""CFT central charge extraction at the XY critical point.

At the BKT transition, the 2D XY model flows to a free boson CFT
with central charge c=1. The entanglement entropy of a subsystem
of length l in a system of length L at criticality follows
(Calabrese & Cardy 2004):

    S(l) = (c/3) × log((L/π) sin(πl/L)) + const

Extracting c from entanglement data is a numerical test of whether
the system is at the XY critical point.

For c=1: confirmed experimentally by cold-atom quantum simulation
(Science, Jan 2026) and theoretically exact for U(1) symmetry.

This module:
    1. Scans K to find the coupling where entanglement entropy peaks
    2. Extracts c at that K via Calabrese-Cardy fit
    3. Reports deviation from c=1 as a criticality diagnostic
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.entanglement_spectrum import (
    entanglement_analysis,
    entropy_vs_subsystem_size,
    fit_cft_central_charge,
)
from ..bridge.knm_hamiltonian import build_knm_paper27


@dataclass
class CFTResult:
    """CFT central charge extraction result."""

    central_charge: float | None
    k_critical: float  # K where entropy peaks
    peak_entropy: float  # S(n/2) at K_c
    deviation_from_c1: float | None  # |c - 1|
    entropy_vs_k: list[float]
    k_values: list[float]


def find_critical_coupling(
    omega: np.ndarray,
    k_range: tuple[float, float] = (0.01, 5.0),
    n_points: int = 30,
) -> tuple[float, list[float], list[float]]:
    """Find K where half-chain entanglement entropy is maximised.

    At the critical point, entanglement is maximal (log divergence).
    """
    n = len(omega)
    k_values = np.linspace(k_range[0], k_range[1], n_points)
    entropies: list[float] = []

    for kb in k_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        analysis = entanglement_analysis(K, omega)
        entropies.append(analysis.half_chain_entropy)

    peak_idx = int(np.argmax(entropies))
    k_c = float(k_values[peak_idx])
    return k_c, list(k_values.astype(float)), entropies


def extract_central_charge(
    K: np.ndarray,
    omega: np.ndarray,
) -> float | None:
    """Extract CFT central charge c from entanglement scaling at given K.

    Uses Calabrese-Cardy formula with chord length correction.
    """
    n = K.shape[0]
    s_vs_l = entropy_vs_subsystem_size(K, omega)
    if len(s_vs_l) < 3:
        return None
    sizes = np.arange(1, n // 2 + 1, dtype=float)
    return fit_cft_central_charge(sizes, np.array(s_vs_l), n)


def cft_analysis(
    omega: np.ndarray,
    k_range: tuple[float, float] = (0.01, 5.0),
    n_points: int = 30,
) -> CFTResult:
    """Full CFT central charge extraction.

    1. Scan K to find critical coupling (max entropy)
    2. Extract c at K_c
    3. Report deviation from c=1
    """
    n = len(omega)
    k_c, k_vals, entropies = find_critical_coupling(omega, k_range, n_points)

    K_c = build_knm_paper27(L=n, K_base=k_c)
    c = extract_central_charge(K_c, omega)

    peak_entropy = max(entropies)
    deviation = abs(c - 1.0) if c is not None else None

    return CFTResult(
        central_charge=c,
        k_critical=k_c,
        peak_entropy=peak_entropy,
        deviation_from_c1=deviation,
        entropy_vs_k=entropies,
        k_values=k_vals,
    )
