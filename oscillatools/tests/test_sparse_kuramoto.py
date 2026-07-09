# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — sparse CPU Kuramoto force and integrator tests
"""Tests for the sparse CPU Kuramoto force and fixed-step integrators."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import oscillatools as kuramoto
import oscillatools.accel.sparse_kuramoto as sk


class _FakeSparseCoo:
    """Minimal SciPy COO-like test double for adapter coverage."""

    def __init__(
        self,
        *,
        row: NDArray[np.intp],
        col: NDArray[np.intp],
        data: NDArray[np.float64],
        shape: tuple[int, int],
    ) -> None:
        self.row = row
        self.col = col
        self.data = data
        self.shape = shape

    def sum_duplicates(self) -> None:
        """Mimic SciPy's duplicate summation for the one duplicated test edge."""
        buckets: dict[tuple[int, int], float] = {}
        for row, col, value in zip(self.row, self.col, self.data, strict=True):
            key = (int(row), int(col))
            buckets[key] = buckets.get(key, 0.0) + float(value)
        ordered = sorted(buckets.items())
        self.row = np.asarray([key[0][0] for key in ordered], dtype=np.intp)
        self.col = np.asarray([key[0][1] for key in ordered], dtype=np.intp)
        self.data = np.asarray([value for _, value in ordered], dtype=np.float64)


def _install_fake_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch sparse adapter hooks to use the local COO test double."""
    sparse_module = cast(Any, sk).sparse
    monkeypatch.setattr(sparse_module, "issparse", lambda value: isinstance(value, _FakeSparseCoo))
    monkeypatch.setattr(sparse_module, "coo_array", lambda raw, dtype, copy: raw)


def _sparse_problem() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a deterministic sparse six-oscillator dense reference problem."""
    theta = np.array([0.0, 0.3, 0.9, 1.7, 2.4, 3.0], dtype=np.float64)
    omega = np.array([0.1, -0.2, 0.05, 0.3, -0.15, 0.0], dtype=np.float64)
    coupling = np.zeros((6, 6), dtype=np.float64)
    for row, col, value in (
        (0, 1, 0.4),
        (1, 2, 0.7),
        (2, 4, 0.2),
        (3, 5, 0.5),
        (4, 0, 0.3),
        (5, 3, 0.6),
    ):
        coupling[row, col] = value
    return theta, omega, coupling


def _sparse_from_dense(coupling: NDArray[np.float64]) -> sk.SparseKuramotoCoupling:
    """Return the canonical sparse record for a dense test matrix."""
    row, col = np.nonzero(coupling)
    return sk.SparseKuramotoCoupling(
        n_oscillators=int(coupling.shape[0]),
        row=np.asarray(row, dtype=np.intp),
        col=np.asarray(col, dtype=np.intp),
        weight=np.asarray(coupling[row, col], dtype=np.float64),
    )


def _dense_force_loop(
    theta: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a dense reference force using Python loops."""
    result = np.zeros(theta.shape[0], dtype=np.float64)
    for row in range(theta.shape[0]):
        total = 0.0
        for col in range(theta.shape[0]):
            total += float(coupling[row, col]) * math.sin(float(theta[col] - theta[row]))
        result[row] = total
    return result


def _dense_euler_loop(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Return an explicit-Euler dense reference trajectory using Python loops."""
    trajectory = np.zeros((n_steps + 1, theta0.shape[0]), dtype=np.float64)
    trajectory[0] = theta0
    current = theta0.copy()
    for step in range(n_steps):
        current = current + dt * (omega + _dense_force_loop(current, coupling))
        trajectory[step + 1] = current
    return trajectory


def _dense_rk4_loop(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Return an RK4 dense reference trajectory using Python loops."""
    trajectory = np.zeros((n_steps + 1, theta0.shape[0]), dtype=np.float64)
    trajectory[0] = theta0
    current = theta0.copy()
    half = 0.5 * dt
    for step in range(n_steps):
        k1 = omega + _dense_force_loop(current, coupling)
        k2 = omega + _dense_force_loop(current + half * k1, coupling)
        k3 = omega + _dense_force_loop(current + half * k2, coupling)
        k4 = omega + _dense_force_loop(current + dt * k3, coupling)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    return trajectory


def test_sparse_force_matches_dense_networked_floor() -> None:
    """Sparse force agrees with the dense Python networked force for the same matrix."""
    theta, _, coupling = _sparse_problem()
    sparse_coupling = _sparse_from_dense(coupling)

    sparse_result = sk.sparse_networked_kuramoto_force(theta, sparse_coupling)
    dense_result = _dense_force_loop(theta, coupling)

    np.testing.assert_allclose(sparse_result, dense_result, atol=1e-14)
    assert sparse_coupling.nnz == int(np.count_nonzero(coupling))
    assert sparse_coupling.density == pytest.approx(sparse_coupling.nnz / 36.0)


def test_sparse_euler_and_rk4_match_dense_python_floors() -> None:
    """Sparse Euler and RK4 trajectories are parity-preserving CPU alternatives."""
    theta, omega, coupling = _sparse_problem()
    sparse_coupling = _sparse_from_dense(coupling)
    dt = 0.03
    n_steps = 8

    np.testing.assert_allclose(
        sk.sparse_kuramoto_euler_trajectory(theta, omega, sparse_coupling, dt, n_steps),
        _dense_euler_loop(theta, omega, coupling, dt, n_steps),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        sk.sparse_kuramoto_rk4_trajectory(theta, omega, sparse_coupling, dt, n_steps),
        _dense_rk4_loop(theta, omega, coupling, dt, n_steps),
        atol=1e-13,
    )


def test_sparse_canonicalisation_drops_diagonal_zeroes_and_sums_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SciPy sparse canonicalisation removes inert entries without losing edge mass."""
    _install_fake_sparse(monkeypatch)
    raw = _FakeSparseCoo(
        data=np.array([9.0, 0.0, 0.2, 0.3, 0.4], dtype=np.float64),
        row=np.array([0, 0, 1, 1, 2], dtype=np.intp),
        col=np.array([0, 1, 2, 2, 1], dtype=np.intp),
        shape=(3, 3),
    )
    coupling = sk.sparse_coupling_from_scipy(raw)

    np.testing.assert_array_equal(coupling.row, np.array([1, 2], dtype=np.intp))
    np.testing.assert_array_equal(coupling.col, np.array([2, 1], dtype=np.intp))
    np.testing.assert_allclose(coupling.weight, np.array([0.5, 0.4], dtype=np.float64))

    def fake_csr_array(args: object, *, shape: tuple[int, int], dtype: object) -> dict[str, Any]:
        """Capture the CSR conversion inputs without constructing SciPy arrays."""
        return {"args": args, "shape": shape, "dtype": dtype}

    sparse_module = cast(Any, sk).sparse
    monkeypatch.setattr(sparse_module, "csr_array", fake_csr_array)
    csr_payload = coupling.to_scipy_csr()
    assert csr_payload["shape"] == (3, 3)
    assert csr_payload["dtype"] is np.float64


def test_sparse_ring_100k_smoke_exercises_large_n_without_dense_matrix() -> None:
    """A 100k-node ring keeps O(N) storage and returns finite sparse trajectories."""
    n_oscillators = 100_000
    coupling = sk.ring_sparse_coupling(n_oscillators, coupling_strength=0.05)
    theta = np.linspace(0.0, 2.0 * math.pi, n_oscillators, endpoint=False, dtype=np.float64)
    omega = np.zeros(n_oscillators, dtype=np.float64)

    force = sk.sparse_networked_kuramoto_force(theta, coupling)
    trajectory = sk.sparse_kuramoto_euler_trajectory(theta, omega, coupling, dt=0.01, n_steps=2)

    assert coupling.nnz == 2 * n_oscillators
    assert coupling.density == pytest.approx(2.0 / n_oscillators)
    assert force.shape == (n_oscillators,)
    assert trajectory.shape == (3, n_oscillators)
    assert np.all(np.isfinite(trajectory))


def test_sparse_inputs_validate_shapes_values_and_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sparse APIs fail closed on invalid shapes, non-finite data, and steps."""
    with pytest.raises(TypeError, match="SciPy sparse"):
        sk.sparse_coupling_from_scipy(np.zeros((2, 2), dtype=np.float64))
    _install_fake_sparse(monkeypatch)
    with pytest.raises(ValueError, match="square sparse matrix"):
        sk.sparse_coupling_from_scipy(
            _FakeSparseCoo(
                row=np.zeros(0, dtype=np.intp),
                col=np.zeros(0, dtype=np.intp),
                data=np.zeros(0, dtype=np.float64),
                shape=(2, 3),
            )
        )
    with pytest.raises(ValueError, match="weights must be finite"):
        sk.sparse_coupling_from_scipy(
            _FakeSparseCoo(
                row=np.array([0], dtype=np.intp),
                col=np.array([1], dtype=np.intp),
                data=np.array([np.nan], dtype=np.float64),
                shape=(2, 2),
            )
        )
    with pytest.raises(ValueError, match="indices must be non-negative"):
        sk.SparseKuramotoCoupling(
            n_oscillators=2,
            row=np.array([-1], dtype=np.intp),
            col=np.array([0], dtype=np.intp),
            weight=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="within n_oscillators"):
        sk.SparseKuramotoCoupling(
            n_oscillators=2,
            row=np.array([0], dtype=np.intp),
            col=np.array([2], dtype=np.intp),
            weight=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="identical one-dimensional shapes"):
        sk.SparseKuramotoCoupling(
            n_oscillators=2,
            row=np.array([0], dtype=np.intp),
            col=np.array([1], dtype=np.intp),
            weight=np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        sk.SparseKuramotoCoupling(
            n_oscillators=2,
            row=np.array([[0]], dtype=np.intp),
            col=np.array([[1]], dtype=np.intp),
            weight=np.array([[1.0]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        sk.SparseKuramotoCoupling(
            n_oscillators=-1,
            row=np.zeros(0, dtype=np.intp),
            col=np.zeros(0, dtype=np.intp),
            weight=np.zeros(0, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="theta must have shape"):
        sk.sparse_networked_kuramoto_force(np.zeros(3), sk.ring_sparse_coupling(2))
    with pytest.raises(ValueError, match="theta must contain only finite values"):
        sk.sparse_networked_kuramoto_force(np.array([0.0, np.inf]), sk.ring_sparse_coupling(2))
    with pytest.raises(ValueError, match="omega must have shape"):
        sk.sparse_kuramoto_euler_trajectory(
            np.zeros(2), np.zeros(3), sk.ring_sparse_coupling(2), 0.1, 1
        )
    with pytest.raises(ValueError, match="dt must be finite"):
        sk.sparse_kuramoto_euler_trajectory(
            np.zeros(2), np.zeros(2), sk.ring_sparse_coupling(2), math.inf, 1
        )
    with pytest.raises(ValueError, match="n_steps must be a non-negative integer"):
        sk.sparse_kuramoto_rk4_trajectory(
            np.zeros(2), np.zeros(2), sk.ring_sparse_coupling(2), 0.1, -1
        )
    with pytest.raises(ValueError, match="positive integer"):
        sk.ring_sparse_coupling(0)
    with pytest.raises(ValueError, match="coupling_strength must be finite"):
        sk.ring_sparse_coupling(2, math.nan)
    zero = sk.ring_sparse_coupling(1, coupling_strength=0.0)
    assert zero.nnz == 0
    assert zero.density == 0.0
    empty = sk.SparseKuramotoCoupling(
        n_oscillators=0,
        row=np.zeros(0, dtype=np.intp),
        col=np.zeros(0, dtype=np.intp),
        weight=np.zeros(0, dtype=np.float64),
    )
    assert empty.density == 0.0
    np.testing.assert_array_equal(sk.sparse_networked_kuramoto_force(np.zeros(1), zero), [0.0])
    with pytest.raises(TypeError, match="SciPy sparse"):
        sk.sparse_networked_kuramoto_force(np.zeros(2), object())


def test_sparse_kuramoto_symbols_are_public_facade_capabilities() -> None:
    """Sparse large-N APIs are exported through the top-level facade and capability map."""
    capabilities = kuramoto.capabilities()

    assert kuramoto.SparseKuramotoCoupling is sk.SparseKuramotoCoupling
    assert kuramoto.sparse_networked_kuramoto_force is sk.sparse_networked_kuramoto_force
    assert kuramoto.sparse_kuramoto_euler_trajectory is sk.sparse_kuramoto_euler_trajectory
    assert kuramoto.sparse_kuramoto_rk4_trajectory is sk.sparse_kuramoto_rk4_trajectory
    assert "sparse_networked_kuramoto_force" in capabilities["forces"]
    assert "sparse_kuramoto_euler_trajectory" in capabilities["integrators"]
    assert "sparse_kuramoto_rk4_trajectory" in capabilities["integrators"]
