# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX accelerator contract tests
"""Contract tests for JAX availability, device metadata, Hamiltonian construction, and dense-budget guards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.hardware import jax_accel as jax_mod


class _FakeJnp:
    """Minimal jax.numpy mock backed by real numpy."""

    def zeros(self, shape: int | tuple[int, ...]) -> _FakeJnpArray:
        return _FakeJnpArray(np.zeros(shape))

    def array(self, x: Any) -> _FakeJnpArray:
        return _FakeJnpArray(np.asarray(x))

    class linalg:
        @staticmethod
        def eigvalsh(H: Any) -> NDArray[np.float64]:
            return np.linalg.eigvalsh(H)

        @staticmethod
        def eigh(H: Any) -> Any:
            return np.linalg.eigh(H)

        @staticmethod
        def svd(M: Any, compute_uv: bool = True) -> Any:
            if compute_uv:
                return np.linalg.svd(M)
            return np.linalg.svd(M, compute_uv=False)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return np.where(cond, x, y)

    def sum(self, x: Any, **kw: Any) -> Any:
        return np.sum(x, **kw)

    def log2(self, x: Any) -> Any:
        return np.log2(x)

    def sort(self, x: Any) -> NDArray[Any]:
        return np.sort(x)


class _FakeJnpArray(np.ndarray):
    """Array that supports .at[].set() and .at[].add() for JAX-style mutation."""

    def __new__(cls, arr: Any) -> _FakeJnpArray:
        return np.asarray(arr).view(cls)

    @property
    def at(self) -> _AtHelper:
        return _AtHelper(self)


class _AtHelper:
    def __init__(self, arr: _FakeJnpArray) -> None:
        self._arr = arr

    def __getitem__(self, idx: Any) -> _AtIdx:
        return _AtIdx(self._arr, idx)


class _AtIdx:
    def __init__(self, arr: _FakeJnpArray, idx: Any) -> None:
        self._arr = arr
        self._idx = idx

    def set(self, val: Any) -> _FakeJnpArray:
        out = self._arr.copy().view(_FakeJnpArray)
        out[self._idx] = val
        return out

    def add(self, val: Any) -> _FakeJnpArray:
        out = self._arr.copy().view(_FakeJnpArray)
        out[self._idx] += val
        return out


@pytest.fixture()
def mock_jax(monkeypatch: pytest.MonkeyPatch) -> _FakeJnp:
    """Patch jax_accel to think JAX is available with a numpy-backed mock."""
    fake_jnp = _FakeJnp()

    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)
    monkeypatch.setattr(jax_mod, "_JAX_GPU", True)
    monkeypatch.setattr(jax_mod, "_jnp", fake_jnp)
    return fake_jnp


def test_is_jax_available_true(mock_jax: _FakeJnp) -> None:
    assert jax_mod.is_jax_available() is True


def test_is_jax_gpu_available_true(mock_jax: _FakeJnp) -> None:
    assert jax_mod.is_jax_gpu_available() is True


def test_jax_device_name_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    assert jax_mod.jax_device_name() == "unavailable"


def test_jax_device_name_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)
    mock_jax = MagicMock()
    mock_jax.devices.return_value = [MagicMock(__str__=lambda s: "cuda:0")]
    with patch.dict("sys.modules", {"jax": mock_jax}):
        name = jax_mod.jax_device_name()
    assert isinstance(name, str)


def test_build_xy_hamiltonian_jax(mock_jax: _FakeJnp) -> None:
    K = _FakeJnpArray(np.array([[0, 0.5], [0.5, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 2)
    assert H.shape == (4, 4)
    # Hermitian check
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_eigensolve_batch_jax(mock_jax: _FakeJnp, monkeypatch: pytest.MonkeyPatch) -> None:
    K_topo = np.array([[0, 1.0], [1.0, 0]])
    omega = np.array([1.0, 2.0])
    k_range = np.array([0.1, 0.5, 1.0])

    # Mock jax.jit and jax.vmap to just call the function
    mock_jax_module = MagicMock()
    mock_jax_module.jit = lambda fn: fn
    mock_jax_module.vmap = lambda fn: lambda xs: np.array([fn(x) for x in xs])
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)

    with patch.dict("sys.modules", {"jax": mock_jax_module}):
        result = jax_mod.eigensolve_batch_jax(K_topo, omega, k_range)

    assert "k_values" in result
    assert "eigenvalues" in result
    assert "spectral_gaps" in result
    assert "ground_energies" in result
    assert len(result["k_values"]) == 3


def test_eigensolve_batch_jax_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="JAX not available"):
        jax_mod.eigensolve_batch_jax(np.eye(2), np.ones(2), np.array([1.0]))


def test_entanglement_scan_jax_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="JAX not available"):
        jax_mod.entanglement_scan_jax(np.eye(2), np.ones(2), np.array([1.0]))


def test_entanglement_scan_jax(mock_jax: _FakeJnp, monkeypatch: pytest.MonkeyPatch) -> None:
    K_topo = np.array([[0, 1.0], [1.0, 0]])
    omega = np.array([1.0, 2.0])
    k_range = np.array([0.5, 1.0])

    mock_jax_module = MagicMock()
    mock_jax_module.jit = lambda fn: fn

    def fake_vmap(fn: Callable[..., Any]) -> Callable[..., Any]:
        def inner(batch: Any) -> Any:
            results = [fn(h) for h in batch]
            return tuple(np.array(x) for x in zip(*results, strict=True))

        return inner

    mock_jax_module.vmap = fake_vmap
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)

    with patch.dict("sys.modules", {"jax": mock_jax_module}):
        result = jax_mod.entanglement_scan_jax(K_topo, omega, k_range)

    assert "entropy" in result
    assert "schmidt_gap" in result
    assert "spectral_gap" in result
    assert len(result["k_values"]) == 2


def test_entanglement_scan_jax_rejects_dense_batch_budget(
    mock_jax: _FakeJnp, monkeypatch: pytest.MonkeyPatch
) -> None:
    K_topo = np.eye(4)
    omega = np.ones(4)
    k_range = np.array([0.5, 1.0])

    def fail_dense(*args: object, **kwargs: object) -> Any:
        raise AssertionError("dense builder must not run after JAX batch budget rejection")

    monkeypatch.setattr(
        "scpn_quantum_control.bridge.knm_hamiltonian.knm_to_dense_matrix",
        fail_dense,
    )

    with pytest.raises(DenseAllocationError, match="JAX entanglement dense batch"):
        jax_mod.entanglement_scan_jax(
            K_topo,
            omega,
            k_range,
            max_dense_gib=1e-12,
        )


def test_jax_hamiltonian_hermitian(mock_jax: _FakeJnp) -> None:
    """JAX-built H must be Hermitian (real symmetric for XY model)."""
    K = _FakeJnpArray(np.array([[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 1.5, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 3)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_jax_hamiltonian_traceless(mock_jax: _FakeJnp) -> None:
    """XY Hamiltonian should be traceless (all Pauli terms)."""
    K = _FakeJnpArray(np.array([[0, 0.5], [0.5, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 2)
    assert abs(np.trace(H)) < 1e-8


def test_jax_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JAX unavailable, is_jax_available returns False."""
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    assert jax_mod.is_jax_available() is False
    assert jax_mod.is_jax_gpu_available() is False


class TestJaxAccelFallback:
    def test_jax_not_available(self) -> None:
        from scpn_quantum_control.hardware.jax_accel import is_jax_available, is_jax_gpu_available

        # These should not crash regardless of JAX availability
        assert isinstance(is_jax_available(), bool)
        assert isinstance(is_jax_gpu_available(), bool)

    def test_jax_device_name(self) -> None:
        from scpn_quantum_control.hardware.jax_accel import jax_device_name

        name = jax_device_name()
        assert isinstance(name, str)

    def test_entanglement_scan_no_jax_gpu(self) -> None:
        """entanglement_vs_coupling should work without JAX GPU."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        K = build_knm_paper27(L=3)
        K_norm = K / np.max(K)
        omega = OMEGA_N_16[:3]
        result = entanglement_vs_coupling(omega, K_norm, np.linspace(1.0, 3.0, 5))
        assert len(result.entropy) == 5
        assert all(np.isfinite(result.entropy))
