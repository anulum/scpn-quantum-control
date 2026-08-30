# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Quantum Evs
"""Tests for quantum-enhanced EVS."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.quantum_evs import (
    QuantumEVSResult,
    quantum_evs_enhance,
)


class TestQuantumEVS:
    """Exercise the public quantum EVS enhancement contract."""

    def test_returns_result(self) -> None:
        """Return the public result record for a finite feature vector."""
        features = np.array([0.5, -0.3, 0.8, 0.1])
        result = quantum_evs_enhance(features, n_osc=4)
        assert isinstance(result, QuantumEVSResult)

    def test_quantum_features_longer(self) -> None:
        """Expose more quantum observables than input features."""
        features = np.array([0.5, -0.3, 0.8])
        result = quantum_evs_enhance(features, n_osc=4)
        assert len(result.quantum_features) > len(features)

    def test_r_global_bounded(self) -> None:
        """Keep the global synchronisation magnitude in the unit interval."""
        features = np.array([0.5, 0.3])
        result = quantum_evs_enhance(features, n_osc=3)
        assert 0 <= result.r_global <= 1.0

    def test_p_h1_proxy_bounded(self) -> None:
        """Keep the phase-topology proxy in the unit interval."""
        features = np.array([0.5, 0.3, 0.7])
        result = quantum_evs_enhance(features, n_osc=4)
        assert 0 <= result.p_h1_proxy <= 1.0

    def test_enhancement_positive(self) -> None:
        """Produce a positive norm enhancement for non-zero features."""
        features = np.array([0.5, 0.3, 0.7])
        result = quantum_evs_enhance(features, n_osc=4)
        assert result.enhancement_factor > 0

    def test_different_features_different_output(self) -> None:
        """Distinguish quantum features produced by distinct inputs."""
        r1 = quantum_evs_enhance(np.array([0.1, 0.1]), n_osc=3)
        r2 = quantum_evs_enhance(np.array([0.9, 0.9]), n_osc=3)
        assert not np.allclose(r1.quantum_features, r2.quantum_features)

    def test_classical_preserved(self) -> None:
        """Preserve the validated classical features in the result."""
        features = np.array([0.5, -0.3, 0.8])
        result = quantum_evs_enhance(features, n_osc=4)
        np.testing.assert_allclose(result.classical_features, features)

    def test_rejects_empty_feature_vector(self) -> None:
        """Reject an empty classical feature vector."""
        with pytest.raises(ValueError, match="at least one feature"):
            quantum_evs_enhance(np.array([]), n_osc=4)

    def test_rejects_non_vector_features(self) -> None:
        """Reject a feature array with more than one dimension."""
        with pytest.raises(ValueError, match="features must be a 1-D"):
            quantum_evs_enhance(np.ones((2, 2)), n_osc=4)

    def test_rejects_non_finite_features(self) -> None:
        """Reject non-finite feature values."""
        with pytest.raises(ValueError, match="features must contain only finite"):
            quantum_evs_enhance(np.array([0.1, np.nan]), n_osc=4)

    def test_rejects_invalid_oscillator_count(self) -> None:
        """Reject an oscillator count below the supported range."""
        with pytest.raises(ValueError, match="n_osc must be between"):
            quantum_evs_enhance(np.array([0.1]), n_osc=0)

    def test_rejects_invalid_time_step(self) -> None:
        """Reject a non-positive evolution time."""
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            quantum_evs_enhance(np.array([0.1]), n_osc=2, dt=0.0)

    def test_rejects_invalid_trotter_repetitions(self) -> None:
        """Reject a non-positive Trotter repetition count."""
        with pytest.raises(ValueError, match="trotter_reps must be positive"):
            quantum_evs_enhance(np.array([0.1]), n_osc=2, trotter_reps=0)


def test_quantum_evs_2_features() -> None:
    """Enhance a two-element classical feature vector."""
    features = np.array([0.5, 0.3])
    result = quantum_evs_enhance(features, n_osc=3)
    assert len(result.quantum_features) > 0


def test_quantum_evs_single_feature() -> None:
    """Enhance a single classical feature."""
    features = np.array([0.7])
    result = quantum_evs_enhance(features, n_osc=2)
    assert np.isfinite(result.enhancement_factor)


def test_quantum_evs_zero_features() -> None:
    """Produce finite quantum features from an all-zero vector."""
    features = np.zeros(3)
    result = quantum_evs_enhance(features, n_osc=4)
    assert np.all(np.isfinite(result.quantum_features))


def test_quantum_evs_negative_features() -> None:
    """Preserve signed classical features in the result."""
    features = np.array([-0.5, 0.3, -0.8])
    result = quantum_evs_enhance(features, n_osc=4)
    np.testing.assert_allclose(result.classical_features, features)
