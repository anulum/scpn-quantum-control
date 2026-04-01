# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Batch3 Modules
"""Tests for batch 3 modules: backend_dispatch, plugin_registry,
gpu_batch_vqe, translation_symmetry, contraction_optimiser.

Multi-angle: parametrised sizes, ED comparison, edge cases,
physical invariants, strict bounds, error conditions, type checks.
"""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry tests."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return K, omega


# =====================================================================
# Backend Dispatch
# =====================================================================
class TestBackendDispatch:
    """Runtime array backend selection tests."""

    def test_default_is_numpy(self):
        from scpn_quantum_control.backend_dispatch import get_backend

        assert get_backend() == "numpy"

    def test_set_and_get_roundtrip(self):
        from scpn_quantum_control.backend_dispatch import get_backend, set_backend

        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_available_includes_numpy(self):
        from scpn_quantum_control.backend_dispatch import available_backends

        avail = available_backends()
        assert "numpy" in avail
        assert isinstance(avail, list)

    def test_all_available_are_settable(self):
        from scpn_quantum_control.backend_dispatch import (
            available_backends,
            get_backend,
            set_backend,
        )

        for backend in available_backends():
            set_backend(backend)
            assert get_backend() == backend
        set_backend("numpy")

    def test_to_numpy_preserves_data(self):
        from scpn_quantum_control.backend_dispatch import to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)
        assert isinstance(result, np.ndarray)

    def test_from_numpy_preserves_data(self):
        from scpn_quantum_control.backend_dispatch import from_numpy, set_backend

        set_backend("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        result = from_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_from_roundtrip(self):
        from scpn_quantum_control.backend_dispatch import (
            from_numpy,
            set_backend,
            to_numpy,
        )

        set_backend("numpy")
        arr = np.random.randn(5, 3)
        roundtripped = to_numpy(from_numpy(arr))
        np.testing.assert_array_equal(arr, roundtripped)

    def test_get_array_module_is_numpy(self):
        from scpn_quantum_control.backend_dispatch import get_array_module, set_backend

        set_backend("numpy")
        assert get_array_module() is np

    def test_to_numpy_multidim(self):
        from scpn_quantum_control.backend_dispatch import to_numpy

        arr = np.random.randn(3, 4, 5)
        result = to_numpy(arr)
        assert result.shape == (3, 4, 5)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_complex(self):
        from scpn_quantum_control.backend_dispatch import to_numpy

        arr = np.array([1 + 2j, 3 + 4j])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)


# =====================================================================
# Plugin Registry
# =====================================================================
class TestPluginRegistry:
    """Extensible backend registry tests."""

    def test_list_backends_includes_builtins(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        backends = registry.list_backends()
        assert "qiskit" in backends
        assert "pennylane" in backends
        assert "cirq" in backends

    def test_qiskit_available(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        assert registry.is_available("qiskit")

    def test_register_custom_backend(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        @registry.register("batch3_test_backend")
        class B3TestRunner:
            def __init__(self, K, omega, **kw):
                self.n = K.shape[0]
                self.K = K

            def run_trotter(self, t=0.1, reps=5):
                return {"energy": -1.0, "n": self.n}

        runner = registry.get_runner("batch3_test_backend", *_system())
        assert runner.n == 4
        result = runner.run_trotter(t=0.1, reps=3)
        assert result["energy"] == -1.0

    def test_unknown_backend_raises(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        with pytest.raises((ValueError, KeyError)):
            registry.get_runner("nonexistent_xyz_987", *_system())

    def test_available_is_subset_of_list(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        avail = registry.available_backends()
        listed = registry.list_backends()
        assert all(b in listed for b in avail)

    def test_is_available_returns_bool(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        assert isinstance(registry.is_available("qiskit"), bool)
        assert isinstance(registry.is_available("nonexistent"), bool)

    def test_register_class_programmatic(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        class ProgRunner:
            def __init__(self, K, omega, **kw):
                self.ok = True

        registry.register_class("prog_test_backend", ProgRunner)
        runner = registry.get_runner("prog_test_backend", *_system())
        assert runner.ok


# =====================================================================
# GPU Batch VQE
# =====================================================================
class TestGPUBatchVQE:
    """Parallel VQE evaluation tests."""

    def test_batch_energy_numpy_shape_and_finiteness(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_energy_numpy

        K, omega = _system(2)
        H = knm_to_dense_matrix(K, omega)
        dim = 4

        def ansatz(params):
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = np.cos(params[0])
            psi[1] = np.sin(params[0])
            return psi

        params = np.array([[0.5], [1.0], [1.5]], dtype=np.float64)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (3,)
        assert all(np.isfinite(energies))

    def test_batch_energy_is_real(self):
        """Energy expectation ⟨ψ|H|ψ⟩ must be real for Hermitian H."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_energy_numpy

        K, omega = _system(2)
        H = knm_to_dense_matrix(K, omega)
        dim = 4

        def ansatz(params):
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = np.cos(params[0])
            psi[1] = np.sin(params[0])
            return psi

        params = np.array([[0.3], [0.7], [1.1], [1.5]], dtype=np.float64)
        energies = batch_energy_numpy(H, params, ansatz)
        for e in energies:
            assert np.isreal(e) or abs(e.imag) < 1e-10

    @pytest.mark.parametrize("n_samples", [5, 20, 50, 100])
    def test_batch_vqe_scan_shape(self, n_samples):
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=n_samples, seed=42)
        assert len(result["energies"]) == n_samples
        assert result["n_samples"] == n_samples

    def test_batch_vqe_best_is_minimum(self):
        """best_energy should equal min(energies)."""
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=50, seed=42)
        np.testing.assert_allclose(
            result["best_energy"],
            np.min(result["energies"]),
            atol=1e-10,
        )

    def test_batch_vqe_output_keys(self):
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        assert set(result.keys()) == {
            "energies",
            "params",
            "best_energy",
            "best_params",
            "n_samples",
        }

    def test_batch_vqe_reproducible(self):
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(2)
        r1 = batch_vqe_scan(K, omega, n_samples=10, seed=42)
        r2 = batch_vqe_scan(K, omega, n_samples=10, seed=42)
        np.testing.assert_array_equal(r1["energies"], r2["energies"])

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_batch_vqe_multiple_sizes(self, n):
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(n)
        result = batch_vqe_scan(K, omega, n_samples=10, seed=42)
        assert all(np.isfinite(result["energies"]))


# =====================================================================
# Translation Symmetry
# =====================================================================
class TestTranslationSymmetry:
    """Cyclic translation symmetry tests."""

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_is_ti_homogeneous(self, n):
        from scpn_quantum_control.analysis.translation_symmetry import (
            is_translation_invariant,
        )

        K, omega = _homogeneous_system(n)
        assert is_translation_invariant(K, omega)

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_is_not_ti_heterogeneous(self, n):
        from scpn_quantum_control.analysis.translation_symmetry import (
            is_translation_invariant,
        )

        K, omega = _system(n)
        assert not is_translation_invariant(K, omega)

    def test_momentum_sectors_dimensions(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            momentum_sector_dimensions,
        )

        dims = momentum_sector_dimensions(4)
        assert sum(dims.values()) >= 2**4
        assert all(d > 0 for d in dims.values())

    @pytest.mark.parametrize("momentum", [0, 1, 2, 3])
    def test_eigh_various_momentum_sectors(self, momentum):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(4)
        result = eigh_with_translation(K, omega, momentum=momentum)
        assert len(result["eigvals"]) > 0
        assert result["momentum"] == momentum
        assert result["is_ti"]

    def test_heterogeneous_raises_valueerror(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _system(4)
        with pytest.raises(ValueError):
            eigh_with_translation(K, omega, momentum=0)

    @pytest.mark.parametrize("n", [4, 6])
    def test_k0_ground_within_full_spectrum(self, n):
        """k=0 sector ground energy ≥ full ground energy."""
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _homogeneous_system(n)
        H = knm_to_dense_matrix(K, omega)
        e_full = np.linalg.eigvalsh(H)

        result = eigh_with_translation(K, omega, momentum=0)
        assert result["eigvals"][0] >= e_full[0] - 1e-8

    def test_eigenvalues_are_real(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(4)
        result = eigh_with_translation(K, omega, momentum=0)
        assert all(np.isreal(e) for e in result["eigvals"])

    def test_eigenvalues_sorted(self):
        from scpn_quantum_control.analysis.translation_symmetry import (
            eigh_with_translation,
        )

        K, omega = _homogeneous_system(6)
        result = eigh_with_translation(K, omega, momentum=0)
        eigvals = result["eigvals"]
        np.testing.assert_array_equal(eigvals, np.sort(eigvals))


# =====================================================================
# Contraction Optimiser
# =====================================================================
class TestContractionOptimiser:
    """Tensor contraction path optimiser tests."""

    def test_contract_matches_einsum_matmul(self):
        from scpn_quantum_control.phase.contraction_optimiser import contract

        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 20))
        B = rng.standard_normal((20, 15))
        result = contract("ij,jk->ik", A, B)
        expected = np.einsum("ij,jk->ik", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_contract_matches_einsum_chain(self):
        """Three-matrix chain contraction."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 12))
        B = rng.standard_normal((12, 10))
        C = rng.standard_normal((10, 6))
        result = contract("ij,jk,kl->il", A, B, C)
        expected = A @ B @ C
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_contract_trace(self):
        """Trace via einsum: Tr(A) = einsum('ii->', A)."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        A = np.random.randn(10, 10)
        result = contract("ii->", A)
        np.testing.assert_allclose(result, np.trace(A), atol=1e-10)

    def test_contract_outer_product(self):
        from scpn_quantum_control.phase.contraction_optimiser import contract

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = contract("i,j->ij", a, b)
        expected = np.outer(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_optimal_path_returns_valid_info(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            optimal_contraction_path,
        )

        A = np.random.randn(5, 10)
        B = np.random.randn(10, 8)
        path, info = optimal_contraction_path("ij,jk->ik", A, B)
        assert "method" in info
        assert isinstance(path, list)

    def test_benchmark_returns_valid_results(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            benchmark_contraction,
        )

        A = np.random.randn(30, 30)
        B = np.random.randn(30, 30)
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=3)
        assert result["naive_ms"] > 0
        assert result["optimised_ms"] > 0
        assert result["speedup"] > 0
        assert np.isfinite(result["speedup"])

    def test_cotengra_availability_is_bool(self):
        from scpn_quantum_control.phase.contraction_optimiser import (
            is_cotengra_available,
        )

        assert isinstance(is_cotengra_available(), bool)

    def test_contract_identity(self):
        """Contracting with identity should preserve the matrix."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        A = np.random.randn(5, 5)
        eye = np.eye(5)
        result = contract("ij,jk->ik", A, eye)
        np.testing.assert_allclose(result, A, atol=1e-12)
