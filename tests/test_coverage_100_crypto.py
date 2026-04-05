# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Coverage 100 Crypto
"""Multi-angle tests for crypto/ subpackage: noise_analysis, percolation.

Covers: concurrence bounds, key rate computation, robustness analysis,
entanglement path finding, parametrised system sizes, edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


# =====================================================================
# Noise Analysis — Concurrence
# =====================================================================
class TestNoiseAnalysis:
    def test_concurrence_bounded_01(self):
        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        rng = np.random.default_rng(42)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        c = _concurrence_2qubit(rho)
        assert 0.0 <= c <= 1.0

    def test_concurrence_bell_state(self):
        """Bell state should have concurrence = 1."""
        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        c = _concurrence_2qubit(rho)
        np.testing.assert_allclose(c, 1.0, atol=1e-6)

    def test_concurrence_product_state(self):
        """Product state |00⟩ should have concurrence = 0."""
        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = np.outer(psi, psi.conj())
        c = _concurrence_2qubit(rho)
        np.testing.assert_allclose(c, 0.0, atol=1e-6)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_concurrence_always_bounded(self, seed):
        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        rng = np.random.default_rng(seed)
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        c = _concurrence_2qubit(rho)
        assert 0.0 - 1e-10 <= c <= 1.0 + 1e-10


# =====================================================================
# Percolation — Key Rate, Robustness, Routing
# =====================================================================
class TestPercolation:
    def test_key_rate_shape(self):
        from scpn_quantum_control.crypto.percolation import (
            concurrence_map,
            key_rate_per_channel,
        )

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        conc = concurrence_map(K, omega, maxiter=10)
        rates = key_rate_per_channel(conc)
        assert rates.shape == (2, 2)

    def test_key_rate_nonnegative(self):
        from scpn_quantum_control.crypto.percolation import (
            concurrence_map,
            key_rate_per_channel,
        )

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        conc = concurrence_map(K, omega, maxiter=10)
        rates = key_rate_per_channel(conc)
        assert np.all(rates >= -1e-10)

    def test_targeted_removal_has_fraction(self):
        from scpn_quantum_control.crypto.percolation import robustness_targeted_removal

        K = build_knm_paper27(L=2)
        result = robustness_targeted_removal(K)
        assert "edges_to_disconnect" in result
        assert "fraction" in result
        assert result["fraction"] > 0

    def test_targeted_removal_fraction_bounded(self):
        from scpn_quantum_control.crypto.percolation import robustness_targeted_removal

        K = build_knm_paper27(L=3)
        result = robustness_targeted_removal(K)
        assert 0.0 < result["fraction"] <= 1.0

    def test_routing_path_valid(self):
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        K = build_knm_paper27(L=3)
        result = best_entanglement_path(K, source=0, target=2)
        assert "path" in result
        assert "bottleneck" in result
        assert result["path"][0] == 0
        assert result["path"][-1] == 2

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_routing_across_sizes(self, n):
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        K = build_knm_paper27(L=n)
        result = best_entanglement_path(K, source=0, target=n - 1)
        assert len(result["path"]) >= 2

    def test_random_removal_resilience(self):
        from scpn_quantum_control.crypto.percolation import robustness_random_removal

        K = build_knm_paper27(L=3)
        result = robustness_random_removal(K, n_trials=5)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_key_rate_entropy_clamped(self):
        """Cover line 129: h_e = 0.0 when concurrence just above eps → e ≈ 0."""
        from scpn_quantum_control.crypto.percolation import key_rate_per_channel

        # C = 1e-9 (above CONCURRENCE_EPS=1e-10) → e ≈ C^2/4 ≈ 2.5e-19 < ENTROPY_CLAMP_EPS
        conc = np.array([[0.0, 1e-9], [1e-9, 0.0]])
        rates = key_rate_per_channel(conc)
        assert np.all(rates >= 0)
        # Key rate should be ≈ 1.0 when h_e clamped to 0
        assert rates[0, 1] > 0.99

    def test_targeted_removal_no_edges(self):
        """Cover line 205: no edges → loop body never executes → fallback return."""
        from scpn_quantum_control.crypto.percolation import robustness_targeted_removal

        # Single node → no edges
        K = np.zeros((1, 1))
        result = robustness_targeted_removal(K)
        assert result["edges_to_disconnect"] == 0
        assert result["fraction"] == 1.0

    def test_routing_invalid_source(self):
        """Cover line 231: source out of range raises ValueError."""
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="out of range"):
            best_entanglement_path(K, source=10, target=0)

    def test_routing_invalid_target(self):
        """Cover line 231: target out of range raises ValueError."""
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        K = build_knm_paper27(L=3)
        with pytest.raises(ValueError, match="out of range"):
            best_entanglement_path(K, source=0, target=-1)

    def test_routing_dijkstra_revisit(self):
        """Cover line 245: Dijkstra skip already-visited node.

        Graph designed so node 2 is pushed twice: first from 0 (bw=0.5),
        then from 1 (bw=0.9, better). The stale entry (-0.5, 2) is popped
        after the fresh entry visits 2, triggering continue on line 245.
        Stale pops before target 3 because -0.5 < -0.1 in min-heap.
        """
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        K = np.zeros((4, 4))
        K[0, 1] = K[1, 0] = 1.0
        K[0, 2] = K[2, 0] = 0.5
        K[1, 2] = K[2, 1] = 0.9
        K[2, 3] = K[3, 2] = 0.1
        result = best_entanglement_path(K, source=0, target=3)
        assert result["path"][0] == 0
        assert result["path"][-1] == 3
        assert result["bottleneck"] == pytest.approx(0.1)


# =====================================================================
# Noise Analysis — Concurrence warning path
# =====================================================================
class TestPercolationEdgeCases:
    """Cover remaining edge cases in percolation.py."""

    def test_percolation_threshold_no_edges(self):
        """Cover line 83: graph with no edges → return inf."""
        from scpn_quantum_control.crypto.percolation import percolation_threshold

        K = np.zeros((3, 3))
        result = percolation_threshold(K)
        assert result == float("inf")

    def test_random_removal_no_edges(self):
        """Cover line 153: graph with no edges → zero resilience."""
        from scpn_quantum_control.crypto.percolation import robustness_random_removal

        K = np.zeros((3, 3))
        result = robustness_random_removal(K)
        assert result["mean_resilience"] == 0.0
        assert result["n_edges"] == 0

    def test_routing_unreachable_target(self):
        """Cover line 263: disconnected graph → empty path."""
        from scpn_quantum_control.crypto.percolation import best_entanglement_path

        # Two disconnected components: {0,1} and {2}
        K = np.zeros((3, 3))
        K[0, 1] = K[1, 0] = 0.5
        result = best_entanglement_path(K, source=0, target=2)
        assert result["path"] == []
        assert result["bottleneck"] == 0.0

    def test_percolation_threshold_always_disconnected(self):
        """Cover line 97: no threshold keeps graph connected.

        Two disconnected components {0,1} and {2,3}: no threshold
        gives a connected graph → for loop exhausts → returns sorted_vals[0].
        """
        from scpn_quantum_control.crypto.percolation import percolation_threshold

        K = np.zeros((4, 4))
        K[0, 1] = K[1, 0] = 0.1
        K[2, 3] = K[3, 2] = 0.2
        result = percolation_threshold(K)
        assert result == pytest.approx(0.1)


class TestNoiseAnalysisEdge:
    def test_concurrence_imaginary_eigenvalues(self, caplog):
        """Cover line 87: warning when eigenvalues have imaginary part."""
        import logging

        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        # Non-physical density matrix: Hermitian but not PSD → rho @ rho_tilde
        # can have eigenvalues with significant imaginary parts
        rho = np.array(
            [
                [0.5, 0.3, 0.2, 0.4],
                [0.3, -0.1, 0.1, 0.2],
                [0.2, 0.1, 0.3, -0.2],
                [0.4, 0.2, -0.2, 0.3],
            ],
            dtype=complex,
        )
        with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.noise_analysis"):
            c = _concurrence_2qubit(rho)
        assert isinstance(c, float)


# =====================================================================
# Topology Auth — zero positive eigenvalues
# =====================================================================
class TestTopologyAuthEdge:
    def test_spectral_entropy_single_positive(self):
        """topology_auth line 90 is dead code: normalized Laplacian L_sym = I - D^{-1/2}KD^{-1/2}
        always has at least one eigenvalue > 0 (1.0 for isolated nodes, positive for connected).
        Verify with edge case: single node → eigenvalue 1.0, entropy=0 from scipy_entropy([1])."""
        from scpn_quantum_control.crypto.topology_auth import normalized_laplacian_fingerprint

        K = np.zeros((1, 1))
        result = normalized_laplacian_fingerprint(K)
        # scipy_entropy([1.0]) = 0.0, but via line 88 not line 90
        assert result["spectral_entropy_norm"] == 0.0
