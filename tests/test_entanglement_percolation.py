# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Entanglement Percolation
"""Tests for entanglement percolation threshold analysis."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.entanglement_percolation import (
    PercolationScanResult,
    concurrence_map_exact,
    fiedler_eigenvalue,
    percolation_scan,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestConcurrenceMapExact:
    def test_product_state_zero_concurrence(self):
        """Product state |00⟩ has zero concurrence."""
        psi = np.array([1.0, 0.0, 0.0, 0.0])
        cmap = concurrence_map_exact(psi, 2)
        assert cmap[0, 1] < 1e-6

    def test_bell_state_max_concurrence(self):
        """Bell state (|00⟩+|11⟩)/√2 has concurrence ≈ 1."""
        psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        cmap = concurrence_map_exact(psi, 2)
        assert cmap[0, 1] > 0.9

    def test_symmetric(self):
        psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        cmap = concurrence_map_exact(psi, 2)
        assert abs(cmap[0, 1] - cmap[1, 0]) < 1e-10

    def test_3qubit_ghz(self):
        """GHZ state has pairwise concurrence 0 (multipartite only)."""
        psi = np.zeros(8)
        psi[0] = psi[7] = 1.0 / np.sqrt(2)
        cmap = concurrence_map_exact(psi, 3)
        # GHZ: pairwise reduced states are maximally mixed → C = 0
        assert np.max(cmap) < 0.1

    def test_3qubit_w_state(self):
        """W state has nonzero pairwise concurrence."""
        psi = np.zeros(8)
        psi[1] = psi[2] = psi[4] = 1.0 / np.sqrt(3)
        cmap = concurrence_map_exact(psi, 3)
        assert cmap[0, 1] > 0.3


class TestFiedlerEigenvalue:
    def test_disconnected_graph(self):
        adj = np.zeros((3, 3))
        assert fiedler_eigenvalue(adj) < 1e-10

    def test_connected_graph(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        assert fiedler_eigenvalue(adj) > 0

    def test_complete_graph(self):
        adj = np.ones((3, 3)) - np.eye(3)
        assert fiedler_eigenvalue(adj) > 2.5


class TestPercolationScan:
    def test_returns_result(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.array([1.0, 3.0]))
        assert isinstance(result, PercolationScanResult)
        assert len(result.k_values) == 2

    def test_weak_coupling_no_percolation(self):
        """Very weak coupling → product ground state → no entanglement."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.array([0.01]))
        assert result.fiedler_values[0] < 1e-6

    def test_strong_coupling_percolation(self):
        """Strong coupling → entangled ground state → percolation."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.array([5.0]))
        assert result.fiedler_values[0] > 0
        assert result.max_concurrence[0] > 0

    def test_percolation_threshold_exists(self):
        """Scanning K should find a percolation threshold."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.linspace(0.1, 6.0, 8))
        # At some K, percolation should kick in
        assert result.k_percolation is not None or np.any(result.fiedler_values > 0)

    def test_R_values_are_valid(self):
        """Order parameter R should be in [0, 1]."""
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.array([0.5, 2.0, 5.0]))
        for r in result.R_values:
            assert 0.0 <= r <= 1.0 + 1e-10

    def test_4qubit_scan(self):
        n = 4
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = percolation_scan(omega, T, k_range=np.array([2.0, 5.0]))
        assert len(result.n_entangled_pairs) == 2
