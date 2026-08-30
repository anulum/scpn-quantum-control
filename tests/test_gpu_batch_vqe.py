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

from collections.abc import Callable
from typing import Any, Never

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.gpu_batch_vqe import (
    batch_energy_numpy,
    batch_energy_torch,
    batch_vqe_scan,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
NumpyAnsatz = Callable[[FloatArray], ComplexArray]


def _system(n: int = 3) -> tuple[FloatArray, FloatArray]:
    K = (0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))).astype(np.float64)
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n, dtype=np.float64)
    return K, omega


def _simple_ansatz(dim: int) -> NumpyAnsatz:
    """Return an ansatz function that produces normalised states."""

    def ansatz(params: FloatArray) -> ComplexArray:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = np.cos(params[0])
        if dim > 1:
            psi[1] = np.sin(params[0])
        return psi / np.linalg.norm(psi)

    return ansatz


class TestBatchEnergyNumpy:
    """Exercise the NumPy batch-energy evaluator."""

    def test_single_param_set(self) -> None:
        """Evaluate one parameter vector to one finite energy."""
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
        params = np.array([[0.5]], dtype=np.float64)
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (1,)
        assert np.isfinite(energies[0])

    def test_multiple_param_sets(self) -> None:
        """Evaluate every row in a parameter batch."""
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
        params = np.array([[0.1], [0.5], [1.0], [1.5]], dtype=np.float64)
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert energies.shape == (4,)
        assert all(np.isfinite(energies))

    def test_energy_bounded(self) -> None:
        """Energies should be between min and max eigenvalues."""
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
        params = np.random.default_rng(42).normal(0, 1, (10, 1)).astype(np.float64)
        ansatz = _simple_ansatz(4)
        energies = batch_energy_numpy(H, params, ansatz)
        assert np.all(energies >= 1.0 - 1e-10)
        assert np.all(energies <= 4.0 + 1e-10)

    def test_ground_state_gives_min_energy(self) -> None:
        """Passing ground state parameters should give minimum energy."""
        H = np.diag([-3.0, -1.0, 1.0, 3.0]).astype(np.complex128)
        params = np.array([[0.0]], dtype=np.float64)  # cos(0)=1 → |0⟩ → E=-3

        def gs_ansatz(p: FloatArray) -> ComplexArray:  # noqa: ARG001
            psi = np.zeros(4, dtype=np.complex128)
            psi[0] = 1.0
            return psi

        energies = batch_energy_numpy(H, params, gs_ansatz)
        np.testing.assert_allclose(energies[0], -3.0, atol=1e-10)


class TestBatchEnergyTorch:
    """Exercise Torch admission and CPU evaluation contracts."""

    def test_import_error(self) -> None:
        """Report the explicit installation requirement when Torch is absent."""
        from unittest.mock import patch

        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            batch_energy_torch(
                np.eye(4, dtype=np.complex128),
                np.zeros((1, 1), dtype=np.float64),
                lambda p: p,
            )

    def test_cpu_fallback(self) -> None:
        """Evaluate the Torch path explicitly on a CPU device."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)

        def torch_ansatz(params: torch.Tensor) -> torch.Tensor:
            psi = torch.zeros(4, dtype=torch.complex64, device=params.device)
            psi[0] = torch.cos(params[0]).to(torch.complex64)
            psi[1] = torch.sin(params[0]).to(torch.complex64)
            return psi

        params = np.array([[0.5]], dtype=np.float64)
        energies = batch_energy_torch(H, params, torch_ansatz, device="cpu")
        assert energies.shape == (1,)
        assert np.isfinite(energies[0])


class TestBatchVQEScan:
    """Exercise the public diagnostic landscape scanner."""

    def test_rejects_dense_budget_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject oversized state before requesting a dense Hamiltonian."""
        import scpn_quantum_control.bridge.knm_hamiltonian as bridge_module

        K, omega = _system(10)

        def fail_if_dense_hamiltonian_is_requested(*args: object, **kwargs: object) -> Never:  # noqa: ARG001
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            bridge_module, "knm_to_dense_matrix", fail_if_dense_hamiltonian_is_requested
        )

        with pytest.raises(DenseAllocationError, match="batch VQE dense"):
            batch_vqe_scan(K, omega, n_samples=1, max_dense_gib=1e-12)

    def test_passes_dense_budget_to_bridge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forward the caller's dense-memory budget to matrix construction."""
        import scpn_quantum_control.bridge.knm_hamiltonian as bridge_module

        K, omega = _system(2)
        seen_budgets: list[float | None] = []

        def fake_dense_matrix(
            K_arg: FloatArray,
            omega_arg: FloatArray,
            **kwargs: float | None,
        ) -> ComplexArray:  # noqa: ARG001
            seen_budgets.append(kwargs.get("max_dense_gib"))
            return np.zeros((4, 4), dtype=complex)

        monkeypatch.setattr(bridge_module, "knm_to_dense_matrix", fake_dense_matrix)

        batch_vqe_scan(K, omega, n_samples=2, seed=42, max_dense_gib=0.25)

        assert seen_budgets == [0.25]

    def test_output_keys(self) -> None:
        """Return the complete diagnostic scan contract."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        expected = {
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
        assert set(result.keys()) == expected

    def test_energies_shape(self) -> None:
        """Return one energy for every requested sample."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=10, seed=42)
        assert result["energies"].shape == (10,)

    def test_best_is_minimum(self) -> None:
        """Select the minimum sampled energy as the best result."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=20, seed=42)
        assert result["best_energy"] == np.min(result["energies"])

    def test_seed_reproducibility(self) -> None:
        """Reproduce sampled energies for the same seed."""
        K, omega = _system(3)
        r1 = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        r2 = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        np.testing.assert_array_equal(r1["energies"], r2["energies"])

    def test_n2(self) -> None:
        """Run the bounded two-oscillator scan."""
        K, omega = _system(2)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        assert result["n_samples"] == 5

    def test_custom_n_params(self) -> None:
        """Honor an explicit diagnostic parameter count."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=5, n_params=4, seed=42)
        assert result["params"].shape[1] == 4

    def test_single_sample(self) -> None:
        """Accept the smallest non-empty sample batch."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=1, seed=42)
        assert len(result["energies"]) == 1

    def test_default_scan_reports_diagnostic_contract(self) -> None:
        """Label the default path as a NumPy diagnostic scan."""
        K, omega = _system(3)
        result = batch_vqe_scan(K, omega, n_samples=5, seed=42)
        assert result["backend"] == "numpy"
        assert result["ansatz_family"] == "product_ry_layers"
        assert result["optimizer"] == "random_parameter_scan"
        assert result["hardware_claim"] == "none_statevector_expectation_scan"

    def test_gpu_request_is_not_silently_ignored_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail instead of downgrading an explicit GPU request."""
        import sys

        K, omega = _system(2)
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="PyTorch not installed"):
            batch_vqe_scan(K, omega, n_samples=2, seed=42, use_gpu=True)

    def test_gpu_request_requires_cuda_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reject a GPU request when CUDA admission is unavailable."""
        torch = pytest.importorskip("torch")

        K, omega = _system(2)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="available CUDA device"):
            batch_vqe_scan(K, omega, n_samples=2, seed=42, use_gpu=True)

    def test_gpu_request_requires_usable_cuda_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject an admitted CUDA device that fails its allocation probe."""
        torch = pytest.importorskip("torch")
        K, omega = _system(2)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _fail_cuda_probe(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("probe failed")

        monkeypatch.setattr(torch, "zeros", _fail_cuda_probe)

        with pytest.raises(RuntimeError, match="usable CUDA device"):
            batch_vqe_scan(K, omega, n_samples=2, seed=42, use_gpu=True)

    def test_cuda_scan_reports_torch_backend_and_minimum_energy(self) -> None:
        """Report CUDA results when a compatible physical device exists."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")

        K, omega = _system(2)
        try:
            torch.zeros(1, device="cuda").cpu()
            torch.cuda.synchronize()
        except Exception:
            with pytest.raises(RuntimeError, match="usable CUDA device"):
                batch_vqe_scan(K, omega, n_samples=3, seed=123, use_gpu=True)
            return

        result = batch_vqe_scan(K, omega, n_samples=3, seed=123, use_gpu=True)

        assert result["backend"] == "torch_cuda"
        assert result["hardware_claim"] == "none_statevector_expectation_scan"
        assert result["energies"].shape == (3,)
        assert np.all(np.isfinite(result["energies"]))
        assert result["best_energy"] == pytest.approx(float(np.min(result["energies"])))
        np.testing.assert_array_equal(
            result["best_params"],
            result["params"][int(np.argmin(result["energies"]))],
        )

    def test_torch_branch_wires_ansatz_and_reports_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the Torch ansatz path through CPU-local emulation."""
        import scpn_quantum_control.phase.gpu_batch_vqe as vqe_mod

        torch = pytest.importorskip("torch")
        K, omega = _system(2)
        captured_norms: list[float] = []

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

        original_zeros = torch.zeros

        def cpu_zeros(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("device") == "cuda":
                kwargs = {**kwargs, "device": "cpu"}
            return original_zeros(*args, **kwargs)

        def recording_energy(
            H: ComplexArray,
            param_sets: FloatArray,
            ansatz_fn: Callable[[Any], Any],
            device: str,
        ) -> FloatArray:
            assert device == "cuda"
            energies = []
            for params in param_sets:
                psi = ansatz_fn(torch.tensor(params, dtype=torch.float32))
                captured_norms.append(float(torch.linalg.norm(psi)))
                energies.append(
                    float(np.real(psi.detach().numpy().conj() @ H @ psi.detach().numpy()))
                )
            return np.asarray(energies, dtype=np.float64)

        monkeypatch.setattr(torch, "zeros", cpu_zeros)
        monkeypatch.setattr(vqe_mod, "batch_energy_torch", recording_energy)

        result = vqe_mod.batch_vqe_scan(K, omega, n_samples=4, seed=321, use_gpu=True)

        assert result["backend"] == "torch_cuda"
        assert result["energies"].shape == (4,)
        assert result["best_energy"] == pytest.approx(float(np.min(result["energies"])))
        np.testing.assert_allclose(captured_norms, 1.0, atol=1e-6)

    def test_invalid_scan_sizes_are_rejected(self) -> None:
        """Reject empty sample batches and parameter vectors."""
        K, omega = _system(2)
        with pytest.raises(ValueError, match="n_samples"):
            batch_vqe_scan(K, omega, n_samples=0)
        with pytest.raises(ValueError, match="n_params"):
            batch_vqe_scan(K, omega, n_samples=2, n_params=0)
