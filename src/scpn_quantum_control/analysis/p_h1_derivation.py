# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — P H1 Derivation Audit
"""Audit of the failed p_h1 = 0.72 BKT-universal derivation.

The square-lattice candidate chain is:

    1. The Kuramoto-XY model on a finite graph is equivalent to a
       U(1) lattice gauge theory (textbook identity).

    2. The synchronisation transition at K_c is the BKT vortex
       unbinding transition (Kosterlitz & Thouless 1973, Nobel 2016).

    3. At the BKT transition, the critical exponent eta = 1/4 and
       the stiffness jump ratio = 2/pi (Nelson-Kosterlitz 1977).

    4. The Hasenbusch-Pinn universal amplitude for the 2D XY model
       on a square lattice is A_HP = 0.8983 (Monte Carlo, 1997).

    5. A candidate persistent-homology H1 fraction is:

           p_h1 = A_HP * sqrt(2/pi) = 0.8983 * 0.7979 = 0.7167

    6. This is within 0.5% of the empirical value p_h1 = 0.72.

That numerical match does not survive the graph-topology check. The
K_nm graph has a different extracted Hasenbusch-Pinn amplitude
(``A_HP ≈ 1.214`` in the committed Monte Carlo audit), which gives
``p_h1 ≈ 0.969`` rather than ``0.72``. The old square-lattice match is
therefore a useful negative result, not a first-principles derivation.

Status: p_h1 = 0.72 remains an open empirical/theoretical question
for the K_nm graph. This module is kept as an auditable record of the
candidate expression and the graph-specific falsifier.

Ref: Hasenbusch & Pinn, J. Phys. A: Math. Gen. 30, 63 (1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bkt_analysis import bkt_analysis
from .bkt_universals import HASENBUSCH_PINN, NK_RATIO, P_H1_TARGET, check_all_candidates

K_NM_GRAPH_A_HP = 1.214


@dataclass
class P_H1_Derivation:
    """p_h1 candidate-derivation audit result."""

    # Candidate expression
    a_hp: float  # Hasenbusch-Pinn amplitude
    nk_sqrt: float  # sqrt(2/pi)
    p_h1_predicted: float  # A_HP * sqrt(2/pi)
    p_h1_target: float  # 0.72
    graph_a_hp: float  # A_HP extracted on the actual K_nm graph
    graph_p_h1_predicted: float  # graph_a_hp * sqrt(2/pi)
    graph_absolute_deviation: float
    graph_relative_deviation_pct: float

    # Accuracy
    absolute_deviation: float
    relative_deviation_pct: float

    # Supporting evidence
    bkt_bound_pair: float  # from bkt_analysis (0.813)
    vortex_binding_check: bool  # T_BKT gives F=0 at unbinding
    universality_best: str  # best expression from candidate scan

    # Verdict
    is_derivable: bool  # false for the K_nm graph after MC falsification
    status: str
    derivation_chain: list[str]


def derive_p_h1(
    K: np.ndarray | None = None,
    omega: np.ndarray | None = None,
) -> P_H1_Derivation:
    """Audit the p_h1 candidate derivation.

    Optionally takes a coupling matrix for the BKT bound-pair check.
    The returned verdict is false for the K_nm graph because the
    graph-specific amplitude gives p_h1 far from 0.72.
    """
    from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    if K is None:
        K = build_knm_paper27(L=16)
    if omega is None:
        omega = OMEGA_N_16[: K.shape[0]]

    # Core derivation
    a_hp = HASENBUSCH_PINN
    nk_sqrt = float(np.sqrt(NK_RATIO))
    p_predicted = a_hp * nk_sqrt
    graph_a_hp = K_NM_GRAPH_A_HP
    graph_p_predicted = graph_a_hp * nk_sqrt

    abs_dev = abs(p_predicted - P_H1_TARGET)
    rel_dev = abs_dev / P_H1_TARGET * 100
    graph_abs_dev = abs(graph_p_predicted - P_H1_TARGET)
    graph_rel_dev = graph_abs_dev / P_H1_TARGET * 100

    # Supporting: BKT bound-pair from coupling matrix
    bkt = bkt_analysis(K)
    bkt_bp = bkt.p_h1_predicted if bkt.p_h1_predicted is not None else 0.0

    # Supporting: best universal candidate
    universals = check_all_candidates()
    best_expr = universals.best_expression

    chain = [
        "1. Kuramoto-XY = U(1) lattice gauge theory",
        "2. Synchronisation transition = BKT vortex unbinding",
        f"3. BKT universal: eta = 1/4, stiffness = 2/pi = {NK_RATIO:.6f}",
        f"4. Square-lattice Hasenbusch-Pinn amplitude: A_HP = {a_hp}",
        f"5. Square-lattice candidate: {a_hp} * {nk_sqrt:.6f} = {p_predicted:.6f}",
        f"6. Square-lattice deviation from 0.72: {abs_dev:.4f} ({rel_dev:.1f}%)",
        f"7. K_nm graph amplitude from MC audit: A_HP = {graph_a_hp:.3f}",
        f"8. K_nm graph candidate: {graph_a_hp:.3f} * {nk_sqrt:.6f} = {graph_p_predicted:.6f}",
        f"9. K_nm graph deviation from 0.72: {graph_abs_dev:.4f} ({graph_rel_dev:.1f}%)",
        "10. Status: OPEN_QUESTION; square-lattice coincidence is not a K_nm derivation",
    ]

    return P_H1_Derivation(
        a_hp=a_hp,
        nk_sqrt=nk_sqrt,
        p_h1_predicted=p_predicted,
        p_h1_target=P_H1_TARGET,
        graph_a_hp=graph_a_hp,
        graph_p_h1_predicted=graph_p_predicted,
        graph_absolute_deviation=graph_abs_dev,
        graph_relative_deviation_pct=graph_rel_dev,
        absolute_deviation=abs_dev,
        relative_deviation_pct=rel_dev,
        bkt_bound_pair=bkt_bp,
        vortex_binding_check=True,
        universality_best=best_expr,
        is_derivable=False,
        status="OPEN_QUESTION",
        derivation_chain=chain,
    )
