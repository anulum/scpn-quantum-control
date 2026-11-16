# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Gpu Accel
"""Tests for GPU acceleration layer."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest

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
    """Detecting the GPU."""

    def test_is_gpu_returns_bool(self) -> None:
        """The availability check returns a boolean."""
        assert isinstance(is_gpu_available(), bool)

    def test_import_guard_accepts_mocked_cupy_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mocked CuPy device counts as available."""
        import scpn_quantum_control.hardware.gpu_accel as gpu_mod

        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1)),
        )

        monkeypatch.setenv("SCPN_GPU_ENABLE", "1")
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        reloaded = importlib.reload(gpu_mod)
        assert reloaded.is_gpu_available() is True

        monkeypatch.setenv("SCPN_GPU_ENABLE", "0")
        importlib.reload(gpu_mod)

    def test_import_guard_exposes_cupy_runtime_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CuPy runtime failure is raised, not hidden."""
        import scpn_quantum_control.hardware.gpu_accel as gpu_mod

        def fail_device_count() -> NoReturn:
            raise RuntimeError("cuda unavailable")

        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=fail_device_count)),
        )

        monkeypatch.setenv("SCPN_GPU_ENABLE", "1")
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        with pytest.raises(RuntimeError, match="cuda unavailable"):
            importlib.reload(gpu_mod)

        monkeypatch.setenv("SCPN_GPU_ENABLE", "0")
        importlib.reload(gpu_mod)

    def test_import_guard_does_not_probe_cupy_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When disabled, CuPy is not probed."""
        import scpn_quantum_control.hardware.gpu_accel as gpu_mod

        def fail_device_count() -> NoReturn:
            raise AssertionError("CuPy must not be probed unless SCPN_GPU_ENABLE=1")

        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=fail_device_count)),
        )

        monkeypatch.setenv("SCPN_GPU_ENABLE", "0")
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        reloaded = importlib.reload(gpu_mod)

        assert reloaded.is_gpu_available() is False

    def test_device_name_string(self) -> None:
        """The device name is a non-empty string."""
        name = gpu_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_memory_non_negative(self) -> None:
        """The reported memory is non-negative."""
        mem = gpu_memory_free_mb()
        assert mem >= 0


class TestEigvalsh:
    """Eigenvalues of Hermitian matrices."""

    def test_small_matrix(self) -> None:
        """Works regardless of GPU availability (falls back to numpy)."""
        A = np.array([[1.0, 0.5], [0.5, 2.0]])
        eigs = eigvalsh(A)
        assert len(eigs) == 2
        np.testing.assert_allclose(eigs, np.linalg.eigvalsh(A), atol=1e-10)


class TestEigh:
    """Eigenpairs of Hermitian matrices."""

    def test_small_matrix(self) -> None:
        """The eigenvalues of a small matrix match NumPy."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        eigs, vecs = eigh(A)
        assert len(eigs) == 2
        np.testing.assert_allclose(eigs, np.linalg.eigvalsh(A), atol=1e-10)


class TestExpm:
    """The matrix exponential."""

    def test_identity(self) -> None:
        """The exponential of zero is the identity."""
        n = 4
        result = expm(np.zeros((n, n), dtype=complex))
        np.testing.assert_allclose(result, np.eye(n), atol=1e-10)

    def test_hermitian(self) -> None:
        """The exponential of a Hermitian matrix matches the reference."""
        from scipy.linalg import expm as scipy_expm

        A = np.array([[0, 1j], [-1j, 0]], dtype=complex)
        result = expm(A)
        expected = scipy_expm(A)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestMatmul:
    """Matrix multiplication."""

    def test_small_matrix(self) -> None:
        """A small product matches NumPy."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B, atol=1e-10)

    def test_identity_matmul(self) -> None:
        """Multiplying by the identity changes nothing."""
        A = np.eye(4)
        B = np.random.default_rng(42).uniform(size=(4, 4))
        result = matmul(A, B)
        np.testing.assert_allclose(result, B, atol=1e-10)

    def test_square_matmul(self) -> None:
        """Squaring a matrix matches NumPy."""
        A = np.random.default_rng(42).uniform(size=(8, 8))
        result = matmul(A, A)
        np.testing.assert_allclose(result, A @ A, atol=1e-10)


class TestEigvalshParity:
    """Verify GPU/CPU parity across sizes."""

    def test_4x4(self) -> None:
        """A 4-by-4 spectrum matches NumPy."""
        A = np.array([[4, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]], dtype=float)
        eigs = eigvalsh(A)
        np.testing.assert_allclose(eigs, np.linalg.eigvalsh(A), atol=1e-10)

    def test_random_hermitian(self) -> None:
        """A random Hermitian matrix gives six finite eigenvalues."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        A = (A + A.T) / 2
        eigs = eigvalsh(A)
        assert len(eigs) == 6
        assert np.all(np.isfinite(eigs))


class TestExpmProperties:
    """Properties of the matrix exponential."""

    def test_expm_unitary_for_skew_hermitian(self) -> None:
        """exp(iH) should be unitary for Hermitian H."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        U = expm(1j * H)
        identity = U @ U.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-10)

    def test_expm_determinant_one(self) -> None:
        """det(exp(A)) = exp(tr(A)) for traceless A."""
        A = np.array([[0, 1j], [-1j, 0]], dtype=complex)
        U = expm(A)
        det = np.linalg.det(U)
        np.testing.assert_allclose(abs(det), np.exp(np.trace(A).real), atol=1e-10)


class TestPipelineGPU:
    """The Hamiltonian pipeline on the GPU path."""

    def test_hamiltonian_diag_via_gpu(self) -> None:
        """Pipeline: build H → diagonalise via gpu_accel → ground energy."""
        from scpn_quantum_control.bridge.knm_hamiltonian import (
            OMEGA_N_16,
            build_knm_paper27,
            knm_to_hamiltonian,
        )

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        eigs = eigvalsh(np.array(mat, dtype=complex).real)
        assert eigs[0] < 0  # ground energy negative
