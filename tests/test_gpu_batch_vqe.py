# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for GPU Batch VQE
"""Tests for GPU-batched VQE evaluation.

Covers:
    - batch_energy_numpy correctness
    - batch_energy_torch import error and CPU path
    - batch_vqe_scan output structure
    - Energy bounds (between min and max eigenvalue)
    - Seed reproducibility
    - Custom n_params
    - Edge cases: single sample, n=2
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.gpu_batch_vqe import (
    batch_energy_numpy,
    batch_energy_torch,
    batch_vqe_scan,
)


def _system(n: int = 3):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def _simple_ansatz(dim: int):
    """Return an ansatz function that produces normalised states."""

    def ansatz(params: np.ndarray) -> np.ndarray:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = np.cos(params[0])
        if dim > 1:
            psi[1] = np.sin(params[0])
        return psi / np.linalg.norm(psi)

    return ansatz


class TestBatchEnergyNumpy:
    def test_single_param_set(self):
        H = np.diag([1.0, 2.0, 3.0, 4.0])
        params = np.array([[0.5]])
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (1,)
        assert np.isfinite(energies[0])

    def test_multiple_param_sets(self):
        H = np.diag([1.0, 2.0, 3.0, 4.0])
        params = np.array([[0.1], [0.5], [1.0], [1.5]])
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (4,)
        assert all(np.isfinite(energies))

    def test_energy_bounded(self):
        """Energies should be between min and max eigenvalues."""
        H = np.diag([1.0, 2.0, 3.0, 4.0])
        params = np.random.default_rng(42).normal(0, 1, (10, 1)).astype(np.float32)
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert np.all(energies >= 1.0 - 1e-10)
        assert np.all(energies <= 4.0 + 1e-10)

    def test_ground_state_gives_min_energy(self):
        """Passing ground state parameters should give minimum energy."""
        H = np.diag([-3.0, -1.0, 1.0, 3.0])
        params = np.array([[0.0]])  # cos(0)=1 → |0⟩ → E=-3

        def gs_ansatz(p: np.ndarray) -> np.ndarray:
            psi = np.zeros(4, dtype=np.complex128)
            psi[0] = 1.0
            return psi

        energies = batch_energy_numpy(H, params, gs_ansatz)
        np.testing.assert_allclose(energies[0], -3.0, atol=1e-10)


class TestBatchEnergyTorch:
    def test_import_error(self):
        from unittest.mock import patch

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            batch_energy_torch(np.eye(4), np.zeros((1, 1)), lambda p: p)

    def test_cpu_fallback(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex64)

        def torch_ansatz(params: torch.Tensor) -> torch.Tensor:
            psi = torch.zeros(4, dtype=torch.complex64, device=params.device)
            psi[0] = torch.cos(params[0]).to(torch.complex64)
            psi[1] = torch.sin(params[0]).to(torch.complex64)
            return psi

        params = np.array([[0.5]], dtype=np.float32)
        energies = batch_energy_torch(H, params, torch_ansatz, device="cpu")
        assert energies.shape == (1,)
        assert np.isfinite(energies[0])


class TestBatchVQEScan:
    def test_output_keys(self):
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        expected = {"energies", "params", "best_energy", "best_params", "n_samples"}
        assert set(result.keys()) == expected

    def test_energies_shape(self):
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=10, seed=42)
        assert result["energies"].shape == (10,)

    def test_best_is_minimum(self):
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=20, seed=42)
        assert result["best_energy"] == np.min(result["energies"])

    def test_seed_reproducibility(self):
        K, omega = _system(3)
        r1 = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        r2 = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        np.testing.assert_array_equal(r1["energies"], r2["energies"])

    def test_n2(self):
        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        assert result["n_samples"] == 5

    def test_custom_n_params(self):
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=5, n_params=4, seed=42)
        assert result["params"].shape[1] == 4

    def test_single_sample(self):
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=1, seed=42)
        assert len(result["energies"]) == 1
