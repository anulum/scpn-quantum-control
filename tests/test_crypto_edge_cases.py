"""Edge-case tests to cover remaining uncovered branches in crypto modules."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from scpn_quantum_control.crypto.noise_analysis import security_analysis
from scpn_quantum_control.crypto.percolation import (
    _concurrence_2qubit,
    concurrence_map,
    key_rate_per_channel,
    percolation_threshold,
    robustness_random_removal,
    robustness_targeted_removal,
)
from scpn_quantum_control.crypto.topology_auth import (
    normalized_laplacian_fingerprint,
    spectral_fingerprint,
    topology_distance,
)


class TestPercolationEdgeCases:
    def test_threshold_zero_matrix_returns_inf(self):
        K = np.zeros((4, 4))
        assert percolation_threshold(K) == float("inf")

    def test_threshold_fallback_single_edge(self):
        K = np.zeros((3, 3))
        K[0, 1] = K[1, 0] = 0.5
        result = percolation_threshold(K)
        assert np.isfinite(result)

    def test_key_rate_zero_concurrence(self):
        conc = np.zeros((3, 3))
        rates = key_rate_per_channel(conc)
        np.testing.assert_allclose(rates, 0.0)

    def test_key_rate_high_concurrence(self):
        conc = np.zeros((2, 2))
        conc[0, 1] = conc[1, 0] = 0.5
        rates = key_rate_per_channel(conc)
        assert rates[0, 1] > 0

    def test_robustness_random_zero_matrix(self):
        K = np.zeros((3, 3))
        result = robustness_random_removal(K, n_trials=5)
        assert result["n_edges"] == 0
        assert result["mean_resilience"] == 0.0

    def test_robustness_targeted_full_connected(self):
        K = np.ones((4, 4)) * 0.5
        np.fill_diagonal(K, 0)
        result = robustness_targeted_removal(K)
        assert result["edges_to_disconnect"] >= 1

    def test_concurrence_product_state(self):
        rho = np.zeros((4, 4))
        rho[0, 0] = 1.0  # |00><00|
        c = _concurrence_2qubit(rho)
        assert c < 1e-10


class TestTopologyAuthEdgeCases:
    def test_fingerprint_zero_matrix(self):
        K = np.zeros((3, 3))
        fp = spectral_fingerprint(K)
        assert fp["spectral_entropy"] == 0.0
        assert fp["n_components"] == 3

    def test_normalized_fingerprint_zero_matrix(self):
        # L_sym = I for K=0, so all eigenvalues = 1.0 → entropy = log2(3)
        K = np.zeros((3, 3))
        fp = normalized_laplacian_fingerprint(K)
        assert fp["spectral_entropy_norm"] > 0

    def test_distance_different_sizes(self):
        fp3 = spectral_fingerprint(np.eye(3) * 0)
        fp4 = spectral_fingerprint(np.eye(4) * 0)
        assert topology_distance(fp3, fp4) == float("inf")


class TestConcurrenceMap:
    def test_concurrence_map_3q(self):
        K = np.array([[0, 0.5, 0.1], [0.5, 0, 0.3], [0.1, 0.3, 0]])
        omega = np.array([1.0, 1.1, 0.9])
        cmap = concurrence_map(K, omega, maxiter=30)
        assert cmap.shape == (3, 3)
        np.testing.assert_allclose(np.diag(cmap), 0.0)
        # Symmetric
        np.testing.assert_allclose(cmap, cmap.T, atol=1e-10)

    def test_concurrence_via_density_matrix(self):
        # Bell state: concurrence = 1
        rho = DensityMatrix([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        c = _concurrence_2qubit(rho)
        assert c > 0.95

    def test_robustness_targeted_tree_graph(self):
        # Tree: removing any edge disconnects
        K = np.zeros((4, 4))
        K[0, 1] = K[1, 0] = 0.5
        K[1, 2] = K[2, 1] = 0.3
        K[2, 3] = K[3, 2] = 0.4
        result = robustness_targeted_removal(K)
        assert result["edges_to_disconnect"] == 1


class TestNoiseAnalysisEdgeCases:
    def test_security_analysis_default_noise_range(self):
        sv = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        result = security_analysis(sv, [0], [1])
        assert len(result["aggregate_rate"]) == 16
