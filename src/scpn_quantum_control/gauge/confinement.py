# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Confinement
"""Confinement-deconfinement transition in the U(1) Kuramoto gauge theory.

In lattice gauge theory, the Wilson loop expectation value distinguishes
confinement from deconfinement:

    Confined:    <W(C)> ~ exp(-σ × Area(C))    — string tension σ > 0
    Deconfined:  <W(C)> ~ exp(-μ × Perimeter(C)) — perimeter law

For the 2D XY model / U(1) gauge theory:
    - Below BKT (T < T_BKT): deconfined (free charges, perimeter law)
    - Above BKT (T > T_BKT): confined (bound pairs, area law)

Note: 2D U(1) pure gauge is always confining (no deconfined phase).
The XY model with matter fields has a genuine confinement transition.
For our Kuramoto system, the coupling K plays the role of inverse
temperature, so:
    - K > K_c: synchronized = ordered = deconfined (perimeter)
    - K < K_c: desynchronized = disordered = confined (area)

The string tension σ(K) is extracted from the ratio of Wilson loops
of different sizes:
    σ = -log(<W(C2)> / <W(C1)>) / (A2 - A1)

where A is the "area" enclosed by the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import build_knm_paper27
from .wilson_loop import compute_wilson_loops


@dataclass
class ConfinementResult:
    """Confinement analysis result."""

    string_tension: float | None  # σ extracted from Wilson loop ratio
    is_confined: bool  # σ > threshold
    wilson_triangle_avg: float  # average |W| for triangles
    wilson_square_avg: float  # average |W| for squares
    confinement_ratio: float  # |W_square| / |W_triangle|
    n_qubits: int


def _average_wilson_by_length(
    K: np.ndarray,
    omega: np.ndarray,
    length: int,
) -> float:
    """Average Wilson loop magnitude for loops of given length."""
    results = compute_wilson_loops(K, omega, max_length=length, max_loops=50)
    loops_at_length = [r for r in results if r.loop_length == length]
    if not loops_at_length:
        return 0.0
    return float(np.mean([r.magnitude for r in loops_at_length]))


def extract_string_tension(
    w_small: float,
    w_large: float,
    area_small: float = 0.5,
    area_large: float = 1.0,
) -> float | None:
    """Extract string tension from Wilson loop ratio.

    σ = -log(W_large / W_small) / (A_large - A_small)

    For triangles: A ≈ sqrt(3)/4 ≈ 0.43
    For squares: A = 1.0 (unit plaquette)
    """
    if w_small < 1e-15 or w_large < 1e-15:
        return None
    ratio = w_large / w_small
    if ratio <= 0:
        return None
    delta_a = area_large - area_small
    if abs(delta_a) < 1e-15:
        return None
    return float(-np.log(ratio) / delta_a)


def confinement_analysis(
    K: np.ndarray,
    omega: np.ndarray,
) -> ConfinementResult:
    """Full confinement-deconfinement analysis.

    Computes Wilson loops for triangles (length 3) and squares (length 4),
    extracts string tension from their ratio.
    """
    n = K.shape[0]
    w_tri = _average_wilson_by_length(K, omega, 3)
    w_sq = _average_wilson_by_length(K, omega, 4)

    # Triangle area ≈ sqrt(3)/4, square area = 1
    sigma = extract_string_tension(w_tri, w_sq, area_small=0.433, area_large=1.0)

    # Confinement ratio: W_square/W_triangle < 1 means area-law decay
    ratio = w_sq / max(w_tri, 1e-15)

    confined = sigma is not None and sigma > 0.01

    return ConfinementResult(
        string_tension=sigma,
        is_confined=confined,
        wilson_triangle_avg=w_tri,
        wilson_square_avg=w_sq,
        confinement_ratio=ratio,
        n_qubits=n,
    )


def confinement_vs_coupling(
    omega: np.ndarray,
    k_values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Scan confinement across coupling strength.

    At K_c, the string tension should vanish (deconfinement transition).
    """
    if k_values is None:
        k_values = np.linspace(0.1, 3.0, 15)

    n = len(omega)
    results: dict[str, list[float]] = {
        "k_base": [],
        "string_tension": [],
        "wilson_triangle": [],
        "wilson_square": [],
        "confinement_ratio": [],
    }

    for kb in k_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        cr = confinement_analysis(K, omega)
        results["k_base"].append(float(kb))
        results["string_tension"].append(
            cr.string_tension if cr.string_tension is not None else 0.0
        )
        results["wilson_triangle"].append(cr.wilson_triangle_avg)
        results["wilson_square"].append(cr.wilson_square_avg)
        results["confinement_ratio"].append(cr.confinement_ratio)

    return results
