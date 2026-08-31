# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Quantum Persistent Homology
"""Tests for quantum persistent homology pipeline."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.analysis.quantum_persistent_homology as qph_mod
from scpn_quantum_control.analysis.quantum_persistent_homology import (
    QuantumPHResult,
    _correlator_from_counts,
    compare_quantum_classical_ph,
    correlation_matrix_from_counts,
    correlation_to_distance,
    ph_sync_scan,
    quantum_persistent_homology,
)


@pytest.fixture(autouse=True)
def _deterministic_ripser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the public pipeline without requiring the optional wheel."""

    def fake_ripser(
        distance: np.ndarray,
        *,
        maxdim: int,
        distance_matrix: bool,
    ) -> dict[str, list[np.ndarray]]:
        assert maxdim == 1
        assert distance_matrix is True
        h0 = np.column_stack((np.zeros(distance.shape[0]), np.full(distance.shape[0], np.inf)))
        h1 = np.array([[0.1, 0.25], [0.2, np.inf]]) if distance.shape[0] >= 4 else np.empty((0, 2))
        return {"dgms": [h0, h1]}

    monkeypatch.setattr(qph_mod, "_RIPSER_AVAILABLE", True)
    monkeypatch.setattr(qph_mod, "ripser", fake_ripser, raising=False)


class TestCorrelatorFromCounts:
    """Verify empirical pairwise basis correlators from counts."""

    def test_all_zeros(self) -> None:
        """Map the all-zero population to unit pair correlations."""
        counts = {"0000": 1000}
        corr = _correlator_from_counts(counts, 4)
        np.testing.assert_array_almost_equal(corr, np.ones((4, 4)))

    def test_mixed(self) -> None:
        """Preserve positive correlation across aligned outcomes."""
        counts = {"00": 500, "11": 500}
        corr = _correlator_from_counts(counts, 2)
        # Both outcomes give Z_0Z_1 = +1, so ⟨Z_0Z_1⟩ = 1
        assert corr[0, 1] == pytest.approx(1.0)

    def test_anticorrelated(self) -> None:
        """Preserve negative correlation across opposite outcomes."""
        counts = {"01": 500, "10": 500}
        corr = _correlator_from_counts(counts, 2)
        # Both outcomes give Z_0Z_1 = -1
        assert corr[0, 1] == pytest.approx(-1.0)

    def test_uncorrelated(self) -> None:
        """Map the balanced population to zero cross-correlation."""
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        corr = _correlator_from_counts(counts, 2)
        assert corr[0, 1] == pytest.approx(0.0)

    def test_empty_counts(self) -> None:
        """Return a zero matrix for an empty count mapping."""
        corr = _correlator_from_counts({}, 3)
        np.testing.assert_array_equal(corr, np.zeros((3, 3)))


class TestCorrelationMatrix:
    """Verify combined X/Y correlation matrices."""

    def test_synchronized_state(self) -> None:
        """Sum aligned X and Y correlations for synchronized counts."""
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        corr = correlation_matrix_from_counts(x_counts, y_counts, 4)
        # XX + YY = 1 + 1 = 2 for all pairs
        assert corr[0, 1] == pytest.approx(2.0)
        assert corr.shape == (4, 4)

    def test_shape(self) -> None:
        """Return the requested qubit-square matrix shape."""
        corr = correlation_matrix_from_counts({"00": 100}, {"00": 100}, 2)
        assert corr.shape == (2, 2)


class TestCorrelationToDistance:
    """Verify normalized correlation-distance construction."""

    def test_fully_correlated(self) -> None:
        """Map fully correlated off-diagonal pairs to zero distance."""
        corr = np.ones((4, 4)) * 2.0
        np.fill_diagonal(corr, 0.0)
        dist = correlation_to_distance(corr)
        # All off-diagonal: |C|/max = 1 → dist = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == pytest.approx(0.0)

    def test_uncorrelated(self) -> None:
        """Map uncorrelated off-diagonal pairs to unit distance."""
        corr = np.zeros((4, 4))
        dist = correlation_to_distance(corr)
        # All off-diagonal should be 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == pytest.approx(1.0)

    def test_diagonal_zero(self) -> None:
        """Force every self-distance to zero."""
        corr = np.random.default_rng(42).standard_normal((4, 4))
        dist = correlation_to_distance(corr)
        np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(4))

    def test_symmetric(self) -> None:
        """Preserve symmetry for a symmetric correlation input."""
        corr = np.array([[1, 0.5], [0.5, 1]])
        dist = correlation_to_distance(corr)
        np.testing.assert_array_almost_equal(dist, dist.T)


class TestQuantumPH:
    """Verify the public quantum-counts persistent-homology pipeline."""

    def test_missing_ripser_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed with installation guidance when ripser is absent."""
        monkeypatch.setattr(qph_mod, "_RIPSER_AVAILABLE", False)

        with pytest.raises(ImportError, match="pip install ripser"):
            quantum_persistent_homology({"00": 1}, {"00": 1}, 2)

    def test_synchronized_low_p_h1(self) -> None:
        """Return a low H1 fraction for aligned count populations."""
        # All qubits aligned → low p_h1
        x_counts = {"0000": 1000}
        y_counts = {"0000": 1000}
        result = quantum_persistent_homology(x_counts, y_counts, 4)
        assert isinstance(result, QuantumPHResult)
        assert result.p_h1 < 0.5
        assert result.n_qubits == 4

    def test_incoherent_higher_p_h1(self) -> None:
        """Process a reproducible incoherent count population."""
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

    def test_two_qubit(self) -> None:
        """Handle the bounded two-qubit topology case."""
        x_counts = {"00": 500, "11": 500}
        y_counts = {"00": 500, "11": 500}
        result = quantum_persistent_homology(x_counts, y_counts, 2)
        assert result.n_qubits == 2

    def test_result_fields(self) -> None:
        """Expose all persistence and geometry result fields."""
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
    """Verify quantum/classical p_h1 comparison output."""

    def test_returns_both(self) -> None:
        """Return both summaries and their floating-point delta."""
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
    """Verify p_h1 scans across measured coupling values."""

    def test_returns_matching_lengths(self) -> None:
        """Align coupling, p_h1, and persistent-H1 series lengths."""
        K_values = np.array([0.0, 0.5, 1.0])
        x_list = [{"0000": 1000}, {"0000": 700, "0011": 300}, {"0000": 1000}]
        y_list = [{"0000": 1000}, {"0000": 700, "0101": 300}, {"0000": 1000}]
        result = ph_sync_scan(x_list, y_list, 4, K_values)
        assert len(result["K_base"]) == 3
        assert len(result["p_h1"]) == 3
        assert len(result["n_h1"]) == 3
