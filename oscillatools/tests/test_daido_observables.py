# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — the Daido higher-order order parameter and its derivatives tests
"""Multi-angle tests for the Daido higher-order order parameter and its derivatives: analytic floor, invariants, cross-tier parity and dispatch."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import oscillatools.accel.daido_observables as da_obs
import oscillatools.accel.dispatcher as d
import oscillatools.accel.order_parameter_observables as op_obs

_GLOBAL_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _order_parameter_value(theta: np.ndarray) -> float:
    """Reference scalar order parameter ``R = |<exp(i theta)>|``."""
    return float(abs(np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))))


# ---------------------------------------------------------------------------
# Daido higher-order order parameters — physics, reduction, parity, dispatch
# ---------------------------------------------------------------------------


def _daido_value(theta: np.ndarray, m: int) -> float:
    return float(abs(np.mean(np.exp(1j * m * theta))))


def _finite_difference_daido_gradient(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros(theta.size, dtype=np.float64)
    for j in range(theta.size):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        grad[j] = (_daido_value(plus, m) - _daido_value(minus, m)) / (2.0 * step)
    return grad


class TestPythonDaidoFloor:
    def test_two_cluster_state_is_detected(self) -> None:
        # Two antipodal clusters: r_1 = 0 (first harmonic cancels), r_2 = 1.
        theta = np.array([0.0, 0.0, math.pi, math.pi])
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 2) == pytest.approx(1.0, abs=1e-10)

    def test_three_cluster_state_is_detected(self) -> None:
        # Three evenly spaced clusters: r_1 = r_2 = 0 but r_3 = 1.
        theta = np.repeat([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0], 3)
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 2) == pytest.approx(0.0, abs=1e-10)
        assert da_obs._python_daido_order_parameter(theta, 3) == pytest.approx(1.0, abs=1e-10)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert da_obs._python_daido_order_parameter(theta, 1) == pytest.approx(
            op_obs._python_order_parameter(theta), abs=1e-12
        )
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_gradient(theta, 1),
            op_obs._python_order_parameter_gradient(theta),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_matches_finite_difference(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _daido_value(theta, m) < 1e-2:
            return
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_gradient(theta, m),
            _finite_difference_daido_gradient(theta, m),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_sums_to_zero(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert abs(float(np.sum(da_obs._python_daido_order_parameter_gradient(theta, m)))) < 1e-12

    def test_value_in_unit_interval(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.uniform(-math.pi, math.pi, size=40)
        for m in (1, 2, 3, 4):
            assert 0.0 - 1e-12 <= da_obs._python_daido_order_parameter(theta, m) <= 1.0 + 1e-12

    def test_rejects_non_positive_harmonic(self) -> None:
        theta = np.zeros(4)
        for bad in (0, -1, -3):
            with pytest.raises(ValueError, match="positive integer"):
                da_obs._python_daido_order_parameter(theta, bad)
            with pytest.raises(ValueError, match="positive integer"):
                da_obs._python_daido_order_parameter_gradient(theta, bad)

    def test_empty_input(self) -> None:
        assert da_obs._python_daido_order_parameter(np.array([]), 2) == 0.0
        assert da_obs._python_daido_order_parameter_gradient(np.array([]), 2).shape == (0,)

    def test_exact_incoherent_gradient_is_zero(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(
            da_obs._python_daido_order_parameter_gradient(theta, 1), np.zeros(4)
        )


class TestRustDaidoTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        m=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_order_parameter_gradient", None)):
            pytest.skip("scpn_quantum_engine.daido_order_parameter_gradient unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        assert da_obs._rust_daido_order_parameter(theta, m) == pytest.approx(
            da_obs._python_daido_order_parameter(theta, m), abs=1e-12
        )
        np.testing.assert_allclose(
            da_obs._rust_daido_order_parameter_gradient(theta, m),
            da_obs._python_daido_order_parameter_gradient(theta, m),
            atol=1e-12,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter(np.zeros(3), 2)

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter_gradient(np.zeros(3), 2)

    def test_rust_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            da_obs._rust_daido_order_parameter(np.zeros(3), 0)


class TestJuliaDaidoTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import daido_order_parameter as julia_value
        from oscillatools.accel.julia import daido_order_parameter_gradient as julia_grad

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        for m in (1, 2, 3):
            assert julia_value(theta, m) == pytest.approx(
                da_obs._python_daido_order_parameter(theta, m), abs=1e-10
            )
            np.testing.assert_allclose(
                julia_grad(theta, m),
                da_obs._python_daido_order_parameter_gradient(theta, m),
                atol=1e-10,
            )


class TestDaidoDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        value_ref = da_obs._python_daido_order_parameter(theta, m)
        grad_ref = da_obs._python_daido_order_parameter_gradient(theta, m)
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_CHAIN:
            try:
                assert impl(theta, m) == pytest.approx(value_ref, abs=1e-10), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, grad_ref, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            daido_order_parameter,
            daido_order_parameter_gradient,
            last_daido_gradient_tier_used,
            last_daido_tier_used,
        )

        assert d.dispatch("daido_order_parameter", np.zeros(4), 2) == pytest.approx(1.0)
        assert d.dispatch("daido_order_parameter_gradient", np.full(4, 0.3), 2).shape == (4,)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=18)
        assert isinstance(daido_order_parameter(theta, 3), float)
        assert daido_order_parameter_gradient(theta, 3).shape == (18,)
        assert last_daido_tier_used() in {"rust", "julia", "python"}
        assert last_daido_gradient_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert da_obs._DAIDO_ORDER_PARAMETER_CHAIN[-1][0] == "python"
        assert da_obs._DAIDO_ORDER_PARAMETER_GRADIENT_CHAIN[-1][0] == "python"


class TestDaidoPartialEngine:
    def test_partial_engine_value_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter),
                ("python", da_obs._python_daido_order_parameter),
            ],
        )
        assert disp(np.zeros(4), 2) == pytest.approx(1.0)
        assert disp.last_tier == "python"

    def test_partial_engine_gradient_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter_gradient),
                ("python", da_obs._python_daido_order_parameter_gradient),
            ],
        )
        out = disp(np.full(4, 0.5), 2)
        np.testing.assert_allclose(
            out, da_obs._python_daido_order_parameter_gradient(np.full(4, 0.5), 2)
        )
        assert disp.last_tier == "python"


# ---------------------------------------------------------------------------
# Daido Hessian — analytic floor, reduction, invariants, parity, dispatch
# ---------------------------------------------------------------------------


def _finite_difference_daido_hessian(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            da_obs._python_daido_order_parameter_gradient(plus, m)
            - da_obs._python_daido_order_parameter_gradient(minus, m)
        ) / (2.0 * step)
    return out


class TestPythonDaidoHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(43)
        theta = rng.uniform(-math.pi, math.pi, size=11)
        m = 2
        scaled = m * theta
        cos_mean = float(np.mean(np.cos(scaled)))
        sin_mean = float(np.mean(np.sin(scaled)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        aligned = (cos_mean * np.cos(scaled) + sin_mean * np.sin(scaled)) / magnitude
        expected = (m * m) * (
            np.outer(aligned, aligned) / (theta.size**2 * magnitude)
            - np.diag(aligned / theta.size)
        )
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, m), expected, atol=1e-15
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_order_parameter_hessian(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, 1),
            op_obs._python_order_parameter_hessian(theta),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_and_rows_sum_to_zero(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = da_obs._python_daido_order_parameter_hessian(theta, m)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-11)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _daido_value(theta, m) < 1e-2:
            return
        np.testing.assert_allclose(
            da_obs._python_daido_order_parameter_hessian(theta, m),
            _finite_difference_daido_hessian(theta, m),
            atol=1e-4,
        )

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            da_obs._python_daido_order_parameter_hessian(np.zeros(4), 0)

    def test_empty_and_incoherent(self) -> None:
        assert da_obs._python_daido_order_parameter_hessian(np.array([]), 2).shape == (0, 0)
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(
            da_obs._python_daido_order_parameter_hessian(theta, 1), np.zeros((4, 4))
        )


class TestRustDaidoHessianTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_order_parameter_hessian", None)):
            pytest.skip("scpn_quantum_engine.daido_order_parameter_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            da_obs._rust_daido_order_parameter_hessian(theta, m),
            da_obs._python_daido_order_parameter_hessian(theta, m),
            atol=1e-11,
        )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            da_obs._rust_daido_order_parameter_hessian(np.zeros(3), 2)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = da_obs.MultiLangDispatcher(
            [
                ("rust", da_obs._rust_daido_order_parameter_hessian),
                ("python", da_obs._python_daido_order_parameter_hessian),
            ],
        )
        out = disp(np.full(4, 0.5), 2)
        np.testing.assert_allclose(
            out, da_obs._python_daido_order_parameter_hessian(np.full(4, 0.5), 2)
        )
        assert disp.last_tier == "python"


class TestJuliaDaidoHessianTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import (
            daido_order_parameter_hessian as julia_hessian,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        for m in (1, 2, 3):
            np.testing.assert_allclose(
                julia_hessian(theta, m),
                da_obs._python_daido_order_parameter_hessian(theta, m),
                atol=1e-10,
            )


class TestDaidoHessianDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = da_obs._python_daido_order_parameter_hessian(theta, m)
        for name, impl in da_obs._DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            daido_order_parameter_hessian,
            last_daido_hessian_tier_used,
        )

        assert d.dispatch("daido_order_parameter_hessian", np.full(4, 0.3), 2).shape == (4, 4)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=16)
        assert daido_order_parameter_hessian(theta, 3).shape == (16, 16)
        assert last_daido_hessian_tier_used() in {"rust", "julia", "python"}

    def test_chain_ends_with_python_floor(self) -> None:
        assert da_obs._DAIDO_ORDER_PARAMETER_HESSIAN_CHAIN[-1][0] == "python"
