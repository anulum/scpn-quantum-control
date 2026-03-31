# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — JAX Accel Mock Tests
"""Mock-based tests for JAX acceleration covering all code paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_quantum_control.hardware import jax_accel as jax_mod


class _FakeJnp:
    """Minimal jax.numpy mock backed by real numpy."""

    def zeros(self, shape):
        return np.zeros(shape)

    def array(self, x):
        return np.asarray(x)

    class linalg:
        @staticmethod
        def eigvalsh(H):
            return np.linalg.eigvalsh(H)

        @staticmethod
        def eigh(H):
            return np.linalg.eigh(H)

        @staticmethod
        def svd(M, compute_uv=True):
            if compute_uv:
                return np.linalg.svd(M)
            return np.linalg.svd(M, compute_uv=False)

    def where(self, cond, x, y):
        return np.where(cond, x, y)

    def sum(self, x, **kw):
        return np.sum(x, **kw)

    def log2(self, x):
        return np.log2(x)

    def sort(self, x):
        return np.sort(x)


class _FakeJnpArray(np.ndarray):
    """Array that supports .at[].set() and .at[].add() for JAX-style mutation."""

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    @property
    def at(self):
        return _AtHelper(self)


class _AtHelper:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, idx):
        return _AtIdx(self._arr, idx)


class _AtIdx:
    def __init__(self, arr, idx):
        self._arr = arr
        self._idx = idx

    def set(self, val):
        out = self._arr.copy().view(_FakeJnpArray)
        out[self._idx] = val
        return out

    def add(self, val):
        out = self._arr.copy().view(_FakeJnpArray)
        out[self._idx] += val
        return out


@pytest.fixture()
def mock_jax(monkeypatch):
    """Patch jax_accel to think JAX is available with a numpy-backed mock."""
    fake_jnp = _FakeJnp()

    # Override zeros to return _FakeJnpArray
    orig_zeros = fake_jnp.zeros

    def patched_zeros(shape):
        return _FakeJnpArray(orig_zeros(shape))

    fake_jnp.zeros = patched_zeros

    # Override array to return _FakeJnpArray
    def patched_array(x):
        return _FakeJnpArray(np.asarray(x))

    fake_jnp.array = patched_array

    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)
    monkeypatch.setattr(jax_mod, "_JAX_GPU", True)
    monkeypatch.setattr(jax_mod, "_jnp", fake_jnp)
    return fake_jnp


def test_is_jax_available_true(mock_jax):
    assert jax_mod.is_jax_available() is True


def test_is_jax_gpu_available_true(mock_jax):
    assert jax_mod.is_jax_gpu_available() is True


def test_jax_device_name_unavailable(monkeypatch):
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    assert jax_mod.jax_device_name() == "unavailable"


def test_jax_device_name_available(monkeypatch):
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)
    mock_jax = MagicMock()
    mock_jax.devices.return_value = [MagicMock(__str__=lambda s: "cuda:0")]
    with patch.dict("sys.modules", {"jax": mock_jax}):
        name = jax_mod.jax_device_name()
    assert isinstance(name, str)


def test_build_xy_hamiltonian_jax(mock_jax):
    K = _FakeJnpArray(np.array([[0, 0.5], [0.5, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 2)
    assert H.shape == (4, 4)
    # Hermitian check
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_eigensolve_batch_jax(mock_jax, monkeypatch):
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


def test_eigensolve_batch_jax_unavailable(monkeypatch):
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="JAX not available"):
        jax_mod.eigensolve_batch_jax(np.eye(2), np.ones(2), np.array([1.0]))


def test_entanglement_scan_jax_unavailable(monkeypatch):
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="JAX not available"):
        jax_mod.entanglement_scan_jax(np.eye(2), np.ones(2), np.array([1.0]))


def test_entanglement_scan_jax(mock_jax, monkeypatch):
    K_topo = np.array([[0, 1.0], [1.0, 0]])
    omega = np.array([1.0, 2.0])
    k_range = np.array([0.5, 1.0])

    mock_jax_module = MagicMock()
    mock_jax_module.jit = lambda fn: fn

    def fake_vmap(fn):
        def inner(batch):
            results = [fn(h) for h in batch]
            return tuple(np.array(x) for x in zip(*results))

        return inner

    mock_jax_module.vmap = fake_vmap
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", True)

    with patch.dict("sys.modules", {"jax": mock_jax_module}):
        result = jax_mod.entanglement_scan_jax(K_topo, omega, k_range)

    assert "entropy" in result
    assert "schmidt_gap" in result
    assert "spectral_gap" in result
    assert len(result["k_values"]) == 2


# ---------------------------------------------------------------------------
# JAX accel physics: Hamiltonian structure
# ---------------------------------------------------------------------------


def test_jax_hamiltonian_hermitian(mock_jax):
    """JAX-built H must be Hermitian (real symmetric for XY model)."""
    K = _FakeJnpArray(np.array([[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 1.5, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 3)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_jax_hamiltonian_traceless(mock_jax):
    """XY Hamiltonian should be traceless (all Pauli terms)."""
    K = _FakeJnpArray(np.array([[0, 0.5], [0.5, 0]]))
    omega = _FakeJnpArray(np.array([1.0, 2.0]))
    H = jax_mod._build_xy_hamiltonian_jax(K, omega, 2)
    assert abs(np.trace(H)) < 1e-8


# ---------------------------------------------------------------------------
# Pipeline: JAX fallback behaviour
# ---------------------------------------------------------------------------


def test_jax_unavailable_fallback(monkeypatch):
    """When JAX unavailable, is_jax_available returns False."""
    monkeypatch.setattr(jax_mod, "_JAX_AVAILABLE", False)
    assert jax_mod.is_jax_available() is False
    assert jax_mod.is_jax_gpu_available() is False
