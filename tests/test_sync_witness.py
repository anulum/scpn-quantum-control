# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Sync Witness
"""Tests for quantum synchronization witness operators."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.sync_witness import (
    WitnessResult,
    _two_point_correlator,
    calibrate_thresholds,
    correlation_witness_from_counts,
    evaluate_all_witnesses,
    fiedler_witness_from_correlator,
    fiedler_witness_from_counts,
    topological_witness_from_correlator,
)


class TestTwoPointCorrelator:
    def test_all_zeros(self):
        counts = {"0000": 1000}
        corr = _two_point_correlator(counts, 4)
        # All qubits in |0⟩ → all ⟨Z_i⟩ = +1 → ⟨Z_iZ_j⟩ = +1
        expected = np.ones((4, 4))
        np.testing.assert_array_almost_equal(corr, expected)

    def test_all_ones(self):
        counts = {"1111": 1000}
        corr = _two_point_correlator(counts, 4)
        # All qubits in |1⟩ → all ⟨Z_i⟩ = -1 → ⟨Z_iZ_j⟩ = +1
        expected = np.ones((4, 4))
        np.testing.assert_array_almost_equal(corr, expected)

    def test_alternating(self):
        counts = {"0101": 1000}
        corr = _two_point_correlator(counts, 4)
        # Qubits 0,2 → +1; qubits 1,3 → -1
        vals = np.array([1, -1, 1, -1])
        expected = np.outer(vals, vals)
        np.testing.assert_array_almost_equal(corr, expected)

    def test_mixed_state(self):
        counts = {"00": 500, "11": 500}
        corr = _two_point_correlator(counts, 2)
        # ⟨Z_0Z_1⟩ = (500 * (+1)(+1) + 500 * (-1)(-1)) / 1000 = 1.0
        assert corr[0, 1] == pytest.approx(1.0)
        # ⟨Z_0⟩ = (500 * 1 + 500 * (-1)) / 1000 = 0.0
        # But ⟨Z_0Z_0⟩ = ⟨Z_0²⟩ = 1.0 always
        assert corr[0, 0] == pytest.approx(1.0)

    def test_uncorrelated(self):
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        corr = _two_point_correlator(counts, 2)
        # Uniformly random → ⟨Z_0Z_1⟩ ≈ 0
        assert corr[0, 1] == pytest.approx(0.0)


class TestCorrelationWitness:
    def test_synchronized_state_fires(self):
        # Perfectly correlated: all same bitstring
        # XX correlator = 1.0 for each pair, YY correlator = 1.0 → sum = 2.0
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        result = correlation_witness_from_counts(x_counts, y_counts, 4, threshold=1.0)
        assert result.is_synchronized
        assert result.expectation_value < 0
        assert result.raw_observable == pytest.approx(2.0)

    def test_incoherent_state_does_not_fire(self):
        # All 16 bitstrings equally likely → ⟨Z_iZ_j⟩ ≈ 0 → correlator ≈ 0
        rng = np.random.default_rng(42)
        x_counts = {
            format(i, "04b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 16] * 16))
        }
        y_counts = {
            format(i, "04b"): int(c) for i, c in enumerate(rng.multinomial(4000, [1 / 16] * 16))
        }
        result = correlation_witness_from_counts(x_counts, y_counts, 4, threshold=1.0)
        assert not result.is_synchronized
        assert result.expectation_value >= 0

    def test_returns_correct_type(self):
        result = correlation_witness_from_counts({"00": 100}, {"00": 100}, 2, 0.0)
        assert isinstance(result, WitnessResult)
        assert result.witness_name == "correlation"


class TestFiedlerWitness:
    def test_fully_connected_fires(self):
        # All-ones correlation matrix → Fiedler eigenvalue = N
        corr = np.ones((4, 4))
        np.fill_diagonal(corr, 0)
        result = fiedler_witness_from_correlator(corr, threshold=1.0)
        assert result.is_synchronized
        assert result.raw_observable > 1.0

    def test_disconnected_does_not_fire(self):
        # Zero correlation → Fiedler eigenvalue = 0
        corr = np.zeros((4, 4))
        result = fiedler_witness_from_correlator(corr, threshold=0.5)
        assert not result.is_synchronized
        assert result.raw_observable == pytest.approx(0.0)

    def test_from_counts(self):
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        result = fiedler_witness_from_counts(x_counts, y_counts, 4, threshold=1.0)
        assert isinstance(result, WitnessResult)
        assert result.witness_name == "fiedler"


class TestTopologicalWitness:
    def test_rank_one_is_synchronized(self):
        # Nearly rank-1 → no persistent holes → p_H1 ≈ 0 → synced
        corr = np.ones((4, 4)) * 0.9
        np.fill_diagonal(corr, 1.0)
        result = topological_witness_from_correlator(corr, threshold=0.5)
        if np.isnan(result.expectation_value):
            pytest.skip("ripser not installed")
        assert result.is_synchronized

    def test_returns_witness_result(self):
        corr = np.eye(4)
        result = topological_witness_from_correlator(corr, threshold=0.5)
        assert isinstance(result, WitnessResult)
        assert result.witness_name == "topological"


class TestEvaluateAll:
    def test_returns_three_witnesses(self):
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        results = evaluate_all_witnesses(x_counts, y_counts, 4)
        assert "correlation" in results
        assert "fiedler" in results
        assert "topological" in results

    def test_all_fire_for_synchronized(self):
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        results = evaluate_all_witnesses(
            x_counts,
            y_counts,
            4,
            corr_threshold=0.5,
            fiedler_threshold=1.0,
            topo_threshold=0.5,
        )
        assert results["correlation"].is_synchronized
        assert results["fiedler"].is_synchronized


class TestCalibrateThresholds:
    def test_returns_three_thresholds(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        thresholds = calibrate_thresholds(K, omega, n_samples=5)
        assert "correlation" in thresholds
        assert "fiedler" in thresholds
        assert "topological" in thresholds
        assert all(isinstance(v, float) for v in thresholds.values())


# ---------------------------------------------------------------------------
# Coverage: edge cases, topological witness, operator builder
# ---------------------------------------------------------------------------


class TestSyncWitnessCoverage:
    """Cover missing lines: n_pairs=0, topological witness, operator builder."""

    def test_correlation_witness_single_qubit(self):
        """Cover line 91: n_pairs=0 for single qubit system."""
        x_counts = {"0": 500, "1": 500}
        y_counts = {"0": 500, "1": 500}
        result = correlation_witness_from_counts(x_counts, y_counts, n_qubits=1)
        assert isinstance(result, WitnessResult)
        assert result.n_qubits == 1

    def test_topological_witness_with_ripser(self):
        """Cover lines 219-231: topological witness using ripser (installed)."""
        rng = np.random.default_rng(42)
        corr = rng.uniform(0.1, 0.9, (4, 4))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        result = topological_witness_from_correlator(corr)
        assert isinstance(result, WitnessResult)
        assert result.witness_name == "topological"

    def test_topological_witness_h1_features(self):
        """Cover lines 223-229: H1 branch with uniform phases on circle.

        Uniformly spaced phases produce a clear 1-cycle in persistent homology.
        """
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        corr = np.cos(np.subtract.outer(theta, theta))
        result = topological_witness_from_correlator(corr, threshold=0.1)
        assert isinstance(result, WitnessResult)
        assert isinstance(result.expectation_value, float)

    def test_build_correlation_witness_operator(self):
        """Cover lines 345-368: build SparsePauliOp witness operator."""
        from scpn_quantum_control.analysis.sync_witness import (
            build_correlation_witness_operator,
        )

        W = build_correlation_witness_operator(3, threshold=0.5)
        assert W.num_qubits == 3

    def test_build_correlation_witness_operator_single_qubit(self):
        """Cover line 349: n_pairs=0 for single qubit."""
        from scpn_quantum_control.analysis.sync_witness import (
            build_correlation_witness_operator,
        )

        W = build_correlation_witness_operator(1, threshold=0.5)
        assert W.num_qubits == 1

    def test_calibrate_thresholds_r_always_above_half(self):
        """Cover line 328: R stays > 0.5 for all K → trans_idx = midpoint."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
        )

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        # Very weak coupling range → R stays ~1.0 (near initial coherent state)
        thresholds = calibrate_thresholds(
            K, omega, K_base_range=np.array([0.001, 0.002, 0.003]), n_samples=3
        )
        assert "correlation" in thresholds
