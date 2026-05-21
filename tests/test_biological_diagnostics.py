# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Biological Diagnostics Tests
"""Tests for biology-oriented diagnostics on Biological Surface Code graphs."""

import numpy as np

from scpn_quantum_control.qec import (
    BiologicalSurfaceCode,
    analyse_biological_surface_code,
)


def test_analyse_biological_surface_code_metrics_and_domains():
    """Diagnostics returns stable topology and domain-coupling metrics."""
    K = np.array(
        [
            [0.0, 1.0, 0.2, 0.0],
            [1.0, 0.0, 0.5, 0.1],
            [0.2, 0.5, 0.0, 0.9],
            [0.0, 0.1, 0.9, 0.0],
        ],
        dtype=float,
    )
    code = BiologicalSurfaceCode(K, threshold=1e-8)
    domains = {0: "L1", 1: "L1", 2: "L2", 3: "L2"}
    report = analyse_biological_surface_code(
        code,
        node_domains=domains,
        metadata={"campaign": "bio-qec"},
    )

    assert report.n_nodes == 4
    assert report.n_edges >= 4
    assert report.n_cycles >= 1
    assert report.max_weighted_degree >= report.mean_weighted_degree
    assert 0.0 <= report.max_betweenness <= 1.0
    assert -1.0 <= report.modularity <= 1.0
    assert report.n_communities >= 1
    assert report.cycle_length_max >= report.cycle_length_mean >= 0.0
    assert report.inter_domain_coupling["L1"]["L2"] > 0.0
    assert report.metadata["campaign"] == "bio-qec"


def test_analyse_export_through_qec_namespace():
    """Diagnostics API must be exported through qec namespace."""
    K = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    code = BiologicalSurfaceCode(K, threshold=1e-8)
    report = analyse_biological_surface_code(code)
    assert report.n_nodes == 2
