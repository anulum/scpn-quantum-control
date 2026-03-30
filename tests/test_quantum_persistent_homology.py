# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Quantum Persistent Homology
"""Tests for quantum persistent homology pipeline."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from ripser import ripser  # noqa: F401

    _RIPSER = True
except ImportError:
    _RIPSER = False

from scpn_quantum_control.analysis.quantum_persistent_homology import (
    QuantumPHResult,
    _correlator_from_counts,
    compare_quantum_classical_ph,
    correlation_matrix_from_counts,
    correlation_to_distance,
    ph_sync_scan,
    quantum_persistent_homology,
)

pytestmark = pytest.mark.skipif(not _RIPSER, reason="ripser not installed")


class TestCorrelatorFromCounts:
    def test_all_zeros(self):
        counts = {"0000": 1000}
        corr = _correlator_from_counts(counts, 4)
        np.testing.assert_array_almost_equal(corr, np.ones((4, 4)))

    def test_mixed(self):
        counts = {"00": 500, "11": 500}
        corr = _correlator_from_counts(counts, 2)
        # Both outcomes give Z_0Z_1 = +1, so ⟨Z_0Z_1⟩ = 1
        assert corr[0, 1] == pytest.approx(1.0)

    def test_anticorrelated(self):
        counts = {"01": 500, "10": 500}
        corr = _correlator_from_counts(counts, 2)
        # Both outcomes give Z_0Z_1 = -1
        assert corr[0, 1] == pytest.approx(-1.0)

    def test_uncorrelated(self):
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        corr = _correlator_from_counts(counts, 2)
        assert corr[0, 1] == pytest.approx(0.0)

    def test_empty_counts(self):
        corr = _correlator_from_counts({}, 3)
        np.testing.assert_array_equal(corr, np.zeros((3, 3)))


class TestCorrelationMatrix:
    def test_synchronized_state(self):
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        corr = correlation_matrix_from_counts(x_counts, y_counts, 4)
        # XX + YY = 1 + 1 = 2 for all pairs
        assert corr[0, 1] == pytest.approx(2.0)
        assert corr.shape == (4, 4)

    def test_shape(self):
        corr = correlation_matrix_from_counts({"00": 100}, {"00": 100}, 2)
        assert corr.shape == (2, 2)


class TestCorrelationToDistance:
    def test_fully_correlated(self):
        corr = np.ones((4, 4)) * 2.0
        np.fill_diagonal(corr, 0.0)
        dist = correlation_to_distance(corr)
        # All off-diagonal: |C|/max = 1 → dist = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == pytest.approx(0.0)

    def test_uncorrelated(self):
        corr = np.zeros((4, 4))
        dist = correlation_to_distance(corr)
        # All off-diagonal should be 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == pytest.approx(1.0)

    def test_diagonal_zero(self):
        corr = np.random.default_rng(42).standard_normal((4, 4))
        dist = correlation_to_distance(corr)
        np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(4))

    def test_symmetric(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        dist = correlation_to_distance(corr)
        np.testing.assert_array_almost_equal(dist, dist.T)


class TestQuantumPH:
    def test_synchronized_low_p_h1(self):
        # All qubits aligned → low p_h1
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        result = quantum_persistent_homology(x_counts, y_counts, 4)
        assert isinstance(result, QuantumPHResult)
        assert result.p_h1 < 0.5
        assert result.n_qubits == 4

    def test_incoherent_higher_p_h1(self):
        # Random measurements → nontrivial topology
        rng = np.random.default_rng(42)
        x_counts = {
            format(i, "04b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 16] * 16))
        }
        y_counts = {
            format(i, "04b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 16] * 16))
        }
        result = quantum_persistent_homology(x_counts, y_counts, 4)
        # Incoherent state has more topological structure
        assert result.n_qubits == 4

    def test_two_qubit(self):
        x_counts = {"00": 500, "11": 500}
        y_counts = {"00": 500, "11": 500}
        result = quantum_persistent_homology(x_counts, y_counts, 2)
        assert result.n_qubits == 2

    def test_result_fields(self):
        x_counts = {"000": 1000}
        y_counts = {"000": 1000}
        result = quantum_persistent_homology(x_counts, y_counts, 3)
        assert hasattr(result, "p_h1")
        assert hasattr(result, "n_h1_persistent")
        assert hasattr(result, "correlation_matrix")
        assert hasattr(result, "distance_matrix")
        assert result.correlation_matrix.shape == (3, 3)
        assert result.distance_matrix.shape == (3, 3)


class TestCompareQuantumClassical:
    def test_returns_both(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        x_counts = {"000": 800, "011": 200}
        y_counts = {"000": 900, "010": 100}
        result = compare_quantum_classical_ph(x_counts, y_counts, 3, K, omega, t=0.5)
        assert "quantum_p_h1" in result
        assert "classical_p_h1" in result
        assert "delta_p_h1" in result
        assert isinstance(result["delta_p_h1"], float)


class TestPHSyncScan:
    def test_returns_matching_lengths(self):
        K_values = np.array([0.0, 0.5, 1.0])
        x_list = [{"0000": 1000}, {"0000": 700, "0011": 300}, {"0000": 1000}]
        y_list = [{"0000": 1000}, {"0000": 700, "0101": 300}, {"0000": 1000}]
        result = ph_sync_scan(x_list, y_list, 4, K_values)
        assert len(result["K_base"]) == 3
        assert len(result["p_h1"]) == 3
        assert len(result["n_h1"]) == 3
