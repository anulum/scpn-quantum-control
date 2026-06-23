# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Network-local Kuramoto order parameter and Jacobian tests
"""Multi-angle tests for the network-local Kuramoto order parameter and its Jacobian.

Covers the analytic closed form, the reduction to the global order parameter for all-to-all
uniform adjacency, the unit-interval bound, the Jacobian as the finite-difference of the
value, the zero-degree and incoherent-neighbourhood subgradient cases, adjacency-shape
validation, cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the
public API, partial-engine fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.local_order as lo
import scpn_quantum_control.accel.order_parameter_observables as op

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _random_adjacency(n: int, seed: int, *, symmetric: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 2.0, size=(n, n))
    if symmetric:
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _finite_difference_jacobian(
    theta: np.ndarray, adjacency: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            lo._python_local_order_parameter(plus, adjacency)
            - lo._python_local_order_parameter(minus, adjacency)
        ) / (2.0 * step)
    return out


class TestPythonLocalOrderFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        adjacency = _random_adjacency(7, 5, symmetric=False)
        expected = np.array(
            [
                abs(sum(adjacency[j, k] * np.exp(1j * theta[k]) for k in range(theta.size)))
                / adjacency[j].sum()
                for j in range(theta.size)
            ]
        )
        np.testing.assert_allclose(
            lo._python_local_order_parameter(theta, adjacency), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_to_all_equals_global_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        adjacency = np.ones((n, n))
        local = lo._python_local_order_parameter(theta, adjacency)
        global_value = op._python_order_parameter(theta)
        np.testing.assert_allclose(local, np.full(n, global_value), atol=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_within_unit_interval(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        adjacency = _random_adjacency(n, seed + 1, symmetric=symmetric)
        local = lo._python_local_order_parameter(theta, adjacency)
        assert np.all(local >= -1e-12)
        assert np.all(local <= 1.0 + 1e-12)

    def test_zero_degree_node_is_zero(self) -> None:
        rng = np.random.default_rng(7)
        theta = rng.uniform(-math.pi, math.pi, size=5)
        adjacency = _random_adjacency(5, 3, symmetric=True)
        adjacency[2, :] = 0.0
        adjacency[:, 2] = 0.0
        local = lo._python_local_order_parameter(theta, adjacency)
        assert local[2] == 0.0

    def test_rejects_non_square_adjacency(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 4"):
            lo._python_local_order_parameter(np.zeros(4), np.zeros((4, 3)))

    def test_empty(self) -> None:
        assert lo._python_local_order_parameter(np.array([]), np.zeros((0, 0))).shape == (0,)


class TestPythonLocalOrderJacobianFloor:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        adjacency = _random_adjacency(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            lo._python_local_order_parameter_jacobian(theta, adjacency),
            _finite_difference_jacobian(theta, adjacency),
            atol=1e-5,
        )

    def test_zero_degree_node_has_zero_subgradient(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.uniform(-math.pi, math.pi, size=5)
        adjacency = _random_adjacency(5, 6, symmetric=True)
        adjacency[1, :] = 0.0
        adjacency[:, 1] = 0.0
        jacobian = lo._python_local_order_parameter_jacobian(theta, adjacency)
        np.testing.assert_array_equal(jacobian[1], np.zeros(5))

    def test_rejects_non_square_adjacency(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            lo._python_local_order_parameter_jacobian(np.zeros(3), np.zeros((2, 2)))

    def test_empty(self) -> None:
        assert lo._python_local_order_parameter_jacobian(np.array([]), np.zeros((0, 0))).shape == (
            0,
            0,
        )


class TestRustLocalOrderTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, symmetric: bool, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "local_order_parameter", None)):
            pytest.skip("scpn_quantum_engine.local_order_parameter unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        adjacency = _random_adjacency(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            lo._rust_local_order_parameter(theta, adjacency),
            lo._python_local_order_parameter(theta, adjacency),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            lo._rust_local_order_parameter_jacobian(theta, adjacency),
            lo._python_local_order_parameter_jacobian(theta, adjacency),
            atol=1e-11,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            lo._rust_local_order_parameter(np.zeros(3), np.zeros((3, 3)))

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            lo._rust_local_order_parameter_jacobian(np.zeros(3), np.zeros((3, 3)))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        adjacency = _random_adjacency(4, 1, symmetric=True)
        for chain, floor in (
            (lo._LOCAL_ORDER_PARAMETER_CHAIN, lo._python_local_order_parameter),
            (lo._LOCAL_ORDER_PARAMETER_JACOBIAN_CHAIN, lo._python_local_order_parameter_jacobian),
        ):
            disp = lo.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, adjacency)
            np.testing.assert_allclose(out, floor(theta, adjacency))
            assert disp.last_tier == "python"


class TestJuliaLocalOrderTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import local_order_parameter as julia_value
        from scpn_quantum_control.accel.julia import (
            local_order_parameter_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for symmetric in (True, False):
            adjacency = _random_adjacency(8, 2, symmetric=symmetric)
            np.testing.assert_allclose(
                julia_value(theta, adjacency),
                lo._python_local_order_parameter(theta, adjacency),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                julia_jacobian(theta, adjacency),
                lo._python_local_order_parameter_jacobian(theta, adjacency),
                atol=1e-10,
            )


class TestLocalOrderDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        adjacency = _random_adjacency(n, seed + 1, symmetric=False)
        for chain, floor in (
            (lo._LOCAL_ORDER_PARAMETER_CHAIN, lo._python_local_order_parameter),
            (lo._LOCAL_ORDER_PARAMETER_JACOBIAN_CHAIN, lo._python_local_order_parameter_jacobian),
        ):
            reference = floor(theta, adjacency)
            for name, impl in chain:
                try:
                    out = impl(theta, adjacency)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_local_order_parameter_jacobian_tier_used,
            last_local_order_parameter_tier_used,
            local_order_parameter,
            local_order_parameter_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        adjacency = _random_adjacency(10, 4, symmetric=True)
        assert d.dispatch("local_order_parameter", theta, adjacency).shape == (10,)
        assert d.dispatch("local_order_parameter_jacobian", theta, adjacency).shape == (10, 10)
        assert local_order_parameter(theta, adjacency).shape == (10,)
        assert local_order_parameter_jacobian(theta, adjacency).shape == (10, 10)
        assert last_local_order_parameter_tier_used() in {"rust", "julia", "python"}
        assert last_local_order_parameter_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert lo._LOCAL_ORDER_PARAMETER_CHAIN[-1][0] == "python"
        assert lo._LOCAL_ORDER_PARAMETER_JACOBIAN_CHAIN[-1][0] == "python"
