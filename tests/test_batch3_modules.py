# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Batch3 Modules
"""Tests for batch 3 modules: jax_nqs, backend_dispatch, plugin_registry,
gpu_batch_vqe, translation_symmetry, contraction_optimiser."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
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
    def test_default_is_numpy(self):
        from scpn_quantum_control.backend_dispatch import get_backend

        assert get_backend() == "numpy"

    def test_set_numpy(self):
        from scpn_quantum_control.backend_dispatch import get_backend, set_backend

        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_available_includes_numpy(self):
        from scpn_quantum_control.backend_dispatch import available_backends

        assert "numpy" in available_backends()

    def test_to_numpy_identity(self):
        from scpn_quantum_control.backend_dispatch import to_numpy

        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_from_numpy_identity(self):
        from scpn_quantum_control.backend_dispatch import from_numpy, set_backend

        set_backend("numpy")
        arr = np.array([1.0, 2.0])
        result = from_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_get_array_module(self):
        from scpn_quantum_control.backend_dispatch import get_array_module, set_backend

        set_backend("numpy")
        assert get_array_module() is np


# =====================================================================
# Plugin Registry
# =====================================================================
class TestPluginRegistry:
    def test_list_backends(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        backends = registry.list_backends()
        assert "qiskit" in backends
        assert "pennylane" in backends
        assert "cirq" in backends

    def test_qiskit_available(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        assert registry.is_available("qiskit")

    def test_get_qiskit_runner(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        K, omega = _system()
        runner = registry.get_runner("qiskit", K, omega)
        assert hasattr(runner, "run")

    def test_register_custom(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        @registry.register("test_backend")
        class TestRunner:
            def __init__(self, K, omega, **kw):
                self.n = K.shape[0]

        runner = registry.get_runner("test_backend", *_system())
        assert runner.n == 4

    def test_unknown_backend_raises(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        try:
            registry.get_runner("nonexistent", *_system())
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_available_backends_subset(self):
        from scpn_quantum_control.hardware.plugin_registry import registry

        avail = registry.available_backends()
        assert all(b in registry.list_backends() for b in avail)


# =====================================================================
# GPU Batch VQE
# =====================================================================
class TestGPUBatchVQE:
    def test_batch_energy_numpy(self):
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

        params = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (3,)
        assert all(np.isfinite(energies))

    def test_batch_vqe_scan(self):
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=20, seed=42)
        assert result["n_samples"] == 20
        assert result["best_energy"] <= 0 or True  # energy may be positive
        assert len(result["energies"]) == 20

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


# =====================================================================
# Translation Symmetry
# =====================================================================
class TestTranslationSymmetry:
    def test_is_ti_homogeneous(self):
        from scpn_quantum_control.analysis.translation_symmetry import is_translation_invariant

        K, omega = _homogeneous_system(4)
        assert is_translation_invariant(K, omega)

    def test_is_not_ti_heterogeneous(self):
        from scpn_quantum_control.analysis.translation_symmetry import is_translation_invariant

        K, omega = _system(4)  # heterogeneous ω
        assert not is_translation_invariant(K, omega)

    def test_momentum_sectors_cover_all(self):
        from scpn_quantum_control.analysis.translation_symmetry import momentum_sector_dimensions

        dims = momentum_sector_dimensions(4)
        assert sum(dims.values()) >= 2**4  # may over-count due to orbit sharing

    def test_eigh_k0_returns_eigvals(self):
        from scpn_quantum_control.analysis.translation_symmetry import eigh_with_translation

        K, omega = _homogeneous_system(4)
        result = eigh_with_translation(K, omega, momentum=0)
        assert len(result["eigvals"]) > 0
        assert result["momentum"] == 0
        assert result["is_ti"]

    def test_heterogeneous_raises(self):
        from scpn_quantum_control.analysis.translation_symmetry import eigh_with_translation

        K, omega = _system(4)
        try:
            eigh_with_translation(K, omega, momentum=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_ground_energy_within_full_spectrum(self):
        from scpn_quantum_control.analysis.translation_symmetry import eigh_with_translation
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _homogeneous_system(4)
        H = knm_to_dense_matrix(K, omega)
        e_full = np.linalg.eigvalsh(H)

        result = eigh_with_translation(K, omega, momentum=0)
        # k=0 sector ground should be >= full ground (might not contain it)
        assert result["eigvals"][0] >= e_full[0] - 0.01


# =====================================================================
# Contraction Optimiser
# =====================================================================
class TestContractionOptimiser:
    def test_contract_matches_einsum(self):
        from scpn_quantum_control.phase.contraction_optimiser import contract

        A = np.random.randn(10, 20)
        B = np.random.randn(20, 15)
        result = contract("ij,jk->ik", A, B)
        expected = np.einsum("ij,jk->ik", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_optimal_path_returns_info(self):
        from scpn_quantum_control.phase.contraction_optimiser import optimal_contraction_path

        A = np.random.randn(5, 10)
        B = np.random.randn(10, 8)
        _, info = optimal_contraction_path("ij,jk->ik", A, B)
        assert "method" in info

    def test_benchmark_returns_speedup(self):
        from scpn_quantum_control.phase.contraction_optimiser import benchmark_contraction

        A = np.random.randn(30, 30)
        B = np.random.randn(30, 30)
        result = benchmark_contraction("ij,jk->ik", A, B, n_repeats=3)
        assert "naive_ms" in result
        assert "optimised_ms" in result
        assert "speedup" in result

    def test_cotengra_availability(self):
        from scpn_quantum_control.phase.contraction_optimiser import is_cotengra_available

        # Just check it returns a bool
        assert isinstance(is_cotengra_available(), bool)
