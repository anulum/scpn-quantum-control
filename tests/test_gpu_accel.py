# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Gpu Accel
"""Tests for GPU acceleration layer."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.hardware.gpu_accel import (
    eigh,
    eigvalsh,
    expm,
    gpu_device_name,
    gpu_memory_free_mb,
    is_gpu_available,
    matmul,
)


class TestGPUAvailability:
    def test_is_gpu_returns_bool(self):
        assert isinstance(is_gpu_available(), bool)

    def test_device_name_string(self):
        name = gpu_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_memory_non_negative(self):
        mem = gpu_memory_free_mb()
        assert mem >= 0


class TestEigvalsh:
    def test_small_matrix(self):
        """Works regardless of GPU availability (falls back to numpy)."""
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        eigs = eigvalsh(A)
        assert len(eigs) == 2
        np.testing.assert_allclose(eigs, np.linalg.eigvalsh(A), atol=1e-10)


class TestEigh:
    def test_small_matrix(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        eigs, vecs = eigh(A)
        assert len(eigs) == 2
        np.testing.assert_allclose(eigs, np.linalg.eigvalsh(A), atol=1e-10)


class TestExpm:
    def test_identity(self):
        n = 4
        result = expm(np.zeros((n, n), dtype=complex))
        np.testing.assert_allclose(result, np.eye(n), atol=1e-10)

    def test_hermitian(self):
        from scipy.linalg import expm as scipy_expm

        A = np.array([[0, 1j], [-1j, 0]], dtype=complex)
        result = expm(A)
        expected = scipy_expm(A)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestMatmul:
    def test_small_matrix(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B, atol=1e-10)
