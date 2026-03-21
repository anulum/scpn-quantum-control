# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for quantum kernel classification."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.quantum_kernel import (
    QuantumKernelResult,
    compute_kernel_matrix,
    quantum_kernel_entry,
)
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27


class TestQuantumKernelEntry:
    def test_self_overlap_one(self):
        """K(x, x) = 1."""
        K = build_knm_paper27(L=3)
        x = np.array([0.5, 0.3, 0.7])
        val = quantum_kernel_entry(x, x, K, 3)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_bounded(self):
        K = build_knm_paper27(L=3)
        x1 = np.array([0.1, 0.2, 0.3])
        x2 = np.array([0.9, 0.8, 0.7])
        val = quantum_kernel_entry(x1, x2, K, 3)
        assert 0 <= val <= 1.0 + 1e-6

    def test_symmetric(self):
        K = build_knm_paper27(L=3)
        x1 = np.array([0.1, 0.5])
        x2 = np.array([0.9, 0.3])
        k12 = quantum_kernel_entry(x1, x2, K, 3)
        k21 = quantum_kernel_entry(x2, x1, K, 3)
        assert k12 == pytest.approx(k21, abs=1e-10)


class TestComputeKernelMatrix:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(4, 3))
        result = compute_kernel_matrix(X, K, 3)
        assert isinstance(result, QuantumKernelResult)

    def test_shape(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(5, 3))
        result = compute_kernel_matrix(X, K, 3)
        assert result.kernel_matrix.shape == (5, 5)

    def test_diagonal_one(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(3, 2))
        result = compute_kernel_matrix(X, K, 3)
        np.testing.assert_allclose(np.diag(result.kernel_matrix), 1.0, atol=1e-6)

    def test_symmetric_matrix(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(4, 2))
        result = compute_kernel_matrix(X, K, 3)
        np.testing.assert_allclose(result.kernel_matrix, result.kernel_matrix.T, atol=1e-10)

    def test_positive_semidefinite(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(4, 3))
        result = compute_kernel_matrix(X, K, 3)
        eigenvalues = np.linalg.eigvalsh(result.kernel_matrix)
        assert np.all(eigenvalues >= -1e-6)

    def test_n_samples_and_features(self):
        K = build_knm_paper27(L=3)
        X = np.random.default_rng(42).uniform(size=(6, 4))
        result = compute_kernel_matrix(X, K, 3)
        assert result.n_samples == 6
        assert result.feature_dim == 4
        assert result.n_qubits == 3
