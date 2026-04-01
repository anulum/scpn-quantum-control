# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Percolation Edge Cases
"""Cover edge cases in percolation.py: lines 122, 198, 238."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.crypto.percolation import (
    active_channel_graph,
    best_entanglement_path,
    key_rate_per_channel,
    robustness_targeted_removal,
)


def test_key_rate_moderate_concurrence():
    """Moderate concurrence exercises the entropy calculation (line 122-124)."""
    conc = np.array([[0.0, 0.5], [0.5, 0.0]])
    rates = key_rate_per_channel(conc)
    assert rates[0, 1] > 0.0
    assert rates[0, 1] == rates[1, 0]


def test_key_rate_clamped_entropy_low():
    """Concurrence below threshold → skipped entirely (line 118)."""
    conc = np.array([[0.0, 1e-20], [1e-20, 0.0]])
    rates = key_rate_per_channel(conc)
    assert np.all(rates == 0.0)


def test_key_rate_high_concurrence():
    """C near 1.0 → e near 0 → clamped branch (line 121)."""
    conc = np.array([[0.0, 0.999], [0.999, 0.0]])
    rates = key_rate_per_channel(conc)
    assert rates[0, 1] >= 0.0


def test_robustness_targeted_fully_connected():
    """Fully connected graph survives many removals before disconnect (line 198)."""
    # 3-node complete graph: removing strongest first
    K = np.array([[0.0, 0.9, 0.8], [0.9, 0.0, 0.7], [0.8, 0.7, 0.0]])
    result = robustness_targeted_removal(K)
    # With 3 edges, removing 1 or 2 should disconnect
    assert result["edges_to_disconnect"] >= 1
    assert "fraction" in result


def test_robustness_targeted_chain_falls_through():
    """Chain graph (n=2) with 1 edge: removing it disconnects immediately."""
    K = np.array([[0.0, 0.1], [0.1, 0.0]])
    result = robustness_targeted_removal(K)
    assert result["edges_to_disconnect"] == 1
    assert result["fraction"] == 1.0


def test_best_path_revisit_skip():
    """Graph with multiple paths forces visited-node skip (line 237-238)."""
    # 4-node graph with multiple routes to force heap re-visits
    K = np.array(
        [
            [0.0, 0.9, 0.5, 0.0],
            [0.9, 0.0, 0.8, 0.3],
            [0.5, 0.8, 0.0, 0.7],
            [0.0, 0.3, 0.7, 0.0],
        ]
    )
    result = best_entanglement_path(K, source=0, target=3)
    assert len(result["path"]) >= 2
    assert result["path"][0] == 0
    assert result["path"][-1] == 3
    assert result["bottleneck"] > 0


def test_best_path_boundary_validation():
    """Source/target out of range raises ValueError (line 224)."""
    K = np.array([[0.0, 0.5], [0.5, 0.0]])
    with pytest.raises(ValueError, match="out of range"):
        best_entanglement_path(K, source=0, target=5)


def test_best_path_same_source_target():
    K = np.array([[0.0, 0.5], [0.5, 0.0]])
    result = best_entanglement_path(K, source=0, target=0)
    assert result["path"] == [0]


def test_best_path_direct_connection():
    K = np.array([[0.0, 0.9], [0.9, 0.0]])
    result = best_entanglement_path(K, source=0, target=1)
    assert result["path"] == [0, 1]
    assert result["bottleneck"] == pytest.approx(0.9)


def test_best_path_3_nodes():
    K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.8], [0.1, 0.8, 0]])
    result = best_entanglement_path(K, source=0, target=2)
    assert result["path"][0] == 0
    assert result["path"][-1] == 2


def test_active_channel_graph_4_nodes():
    K = build_knm_paper27(L=4)
    channels = active_channel_graph(K, threshold=0.05)
    assert len(channels) > 0
