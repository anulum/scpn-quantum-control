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
