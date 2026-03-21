# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for classical shadow tomography."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.shadow_tomography import (
    ShadowResult,
    _pauli_from_label,
    classical_shadow_estimation,
    estimate_pauli_expectation,
)


def _zero_state(n: int) -> np.ndarray:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    return psi


def _plus_state(n: int) -> np.ndarray:
    """Equal superposition |+>^n."""
    psi = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    return psi


class TestPauliFromLabel:
    def test_identity(self):
        P = _pauli_from_label("II", 2)
        np.testing.assert_allclose(P, np.eye(4), atol=1e-12)

    def test_z_single(self):
        P = _pauli_from_label("Z", 1)
        assert P[0, 0] == pytest.approx(1.0)
        assert P[1, 1] == pytest.approx(-1.0)

    def test_shape(self):
        P = _pauli_from_label("XYZ", 3)
        assert P.shape == (8, 8)


class TestEstimatePauliExpectation:
    def test_z_on_zero_state_positive(self):
        """<0|Z|0> = 1 → shadow estimate should be positive."""
        psi = _zero_state(1)
        val = estimate_pauli_expectation(psi, 1, "Z", n_shots=1000, seed=42)
        assert val > 0  # correct sign

    def test_x_on_zero_state(self):
        """<0|X|0> = 0."""
        psi = _zero_state(1)
        val = estimate_pauli_expectation(psi, 1, "X", n_shots=1000, seed=42)
        assert abs(val) < 3.0  # bounded noise

    def test_two_qubit_observable(self):
        """Shadow estimation runs on 2-qubit system."""
        psi = _zero_state(2)
        val = estimate_pauli_expectation(psi, 2, "ZZ", n_shots=500, seed=42)
        assert isinstance(val, float)

    def test_returns_float(self):
        psi = _zero_state(2)
        val = estimate_pauli_expectation(psi, 2, "XI", n_shots=100, seed=42)
        assert isinstance(val, float)


class TestClassicalShadowEstimation:
    def test_returns_shadow_result(self):
        psi = _zero_state(2)
        obs = {"Z0": "ZI", "Z1": "IZ"}
        result = classical_shadow_estimation(psi, 2, obs, n_shots=100, seed=42)
        assert isinstance(result, ShadowResult)

    def test_all_observables_estimated(self):
        psi = _zero_state(2)
        obs = {"Z0": "ZI", "Z1": "IZ", "XX": "XX"}
        result = classical_shadow_estimation(psi, 2, obs, n_shots=100, seed=42)
        assert set(result.estimated_observables.keys()) == {"Z0", "Z1", "XX"}

    def test_shadow_norm_bound_positive(self):
        psi = _zero_state(2)
        obs = {"ZZ": "ZZ"}
        result = classical_shadow_estimation(psi, 2, obs, n_shots=500, seed=42)
        assert result.shadow_norm_bound > 0

    def test_n_qubits_and_shots(self):
        psi = _zero_state(3)
        obs = {"Z0": "ZII"}
        result = classical_shadow_estimation(psi, 3, obs, n_shots=200, seed=42)
        assert result.n_qubits == 3
        assert result.n_shots == 200

    def test_plus_state_z_near_zero(self):
        """<+|Z|+> = 0."""
        psi = _plus_state(1)
        obs = {"Z": "Z"}
        result = classical_shadow_estimation(psi, 1, obs, n_shots=1000, seed=42)
        assert abs(result.estimated_observables["Z"]) < 0.5
