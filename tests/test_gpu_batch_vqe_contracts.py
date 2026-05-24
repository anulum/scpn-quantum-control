# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — GPU batch VQE contract tests
"""Contract tests for batch VQE energy scans, reproducibility, and output selection."""

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
            "backend",
            "ansatz_family",
            "optimizer",
            "hardware_claim",
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
