# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for graph topology → p_h1 scan."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.graph_topology_scan import (
    GraphP_H1_Result,
    _erdos_renyi_coupling,
    _ring_coupling,
    _watts_strogatz_coupling,
)


class TestGraphGenerators:
    def test_erdos_renyi_symmetric(self):
        K = _erdos_renyi_coupling(16, 0.5)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_erdos_renyi_density(self):
        K = _erdos_renyi_coupling(16, 1.0)
        assert np.all(K[np.eye(16) == 0] > 0)

    def test_erdos_renyi_empty(self):
        K = _erdos_renyi_coupling(16, 0.0)
        assert np.sum(K) == 0.0

    def test_ring_degree(self):
        K = _ring_coupling(16, k=2)
        degrees = np.sum(K > 0, axis=1)
        assert np.all(degrees == 4)

    def test_ring_shape(self):
        K = _ring_coupling(8, k=1)
        assert K.shape == (8, 8)

    def test_watts_strogatz_shape(self):
        K = _watts_strogatz_coupling(16, k=4, beta=0.3)
        assert K.shape == (16, 16)

    def test_watts_strogatz_symmetric(self):
        K = _watts_strogatz_coupling(16, k=4, beta=0.5)
        np.testing.assert_allclose(K, K.T, atol=1e-12)


class TestGraphP_H1_Result:
    def test_dataclass_fields(self):
        r = GraphP_H1_Result(
            graph_family="test",
            parameter=0.5,
            n_nodes=16,
            avg_degree=4.0,
            p_h1_mean=0.01,
            p_h1_std=0.005,
            n_samples=10,
        )
        assert r.graph_family == "test"
        assert 0 <= r.p_h1_mean <= 1.0
