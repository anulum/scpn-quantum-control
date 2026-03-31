# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Graph Topology Scan
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


# ---------------------------------------------------------------------------
# Graph generator invariants
# ---------------------------------------------------------------------------


class TestGraphInvariants:
    def test_ring_zero_diagonal(self):
        K = _ring_coupling(8, k=1)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_erdos_renyi_zero_diagonal(self):
        K = _erdos_renyi_coupling(8, 0.5)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_watts_strogatz_zero_diagonal(self):
        K = _watts_strogatz_coupling(8, k=2, beta=0.3)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_ring_non_negative(self):
        K = _ring_coupling(16, k=2)
        assert np.all(K >= 0)

    def test_erdos_renyi_non_negative(self):
        K = _erdos_renyi_coupling(16, 0.3)
        assert np.all(K >= 0)


# ---------------------------------------------------------------------------
# Pipeline: graph → coupling → Hamiltonian wiring
# ---------------------------------------------------------------------------


class TestGraphPipeline:
    def test_ring_to_hamiltonian(self):
        """Ring topology coupling feeds into knm_to_hamiltonian without error."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            knm_to_hamiltonian,
        )

        K = _ring_coupling(4, k=1)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == 4

    def test_erdos_renyi_to_hamiltonian(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            knm_to_hamiltonian,
        )

        K = _erdos_renyi_coupling(4, 0.8)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == 4
