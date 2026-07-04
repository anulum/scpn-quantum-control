# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — the Kuramoto mean phase and its derivatives tests
"""Multi-angle tests for the Kuramoto mean phase and its derivatives: analytic floor, invariants, cross-tier parity and dispatch."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import oscillatools.accel.dispatcher as d
import oscillatools.accel.mean_phase_observables as mp_obs

_GLOBAL_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# ---------------------------------------------------------------------------
# Mean phase and its gradient — analytic floor, invariants, parity, dispatch
# ---------------------------------------------------------------------------


def _order_parameter_value(theta: np.ndarray) -> float:
    """Reference scalar order parameter ``R = |<exp(i theta)>|``."""
    return float(abs(np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)))))


def _finite_difference_mean_phase_gradient(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the circular mean phase, unwrapped at ±π."""
    grad = np.zeros(theta.size, dtype=np.float64)
    for j in range(theta.size):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        delta = mp_obs._python_mean_phase(plus) - mp_obs._python_mean_phase(minus)
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        grad[j] = delta / (2.0 * step)
    return grad


class TestPythonMeanPhaseFloor:
    def test_value_matches_atan2(self) -> None:
        rng = np.random.default_rng(31)
        theta = rng.uniform(-math.pi, math.pi, size=13)
        expected = math.atan2(float(np.mean(np.sin(theta))), float(np.mean(np.cos(theta))))
        assert mp_obs._python_mean_phase(theta) == pytest.approx(expected, abs=1e-12)

    def test_single_oscillator_is_identity(self) -> None:
        assert mp_obs._python_mean_phase(np.array([2.7])) == pytest.approx(2.7, abs=1e-12)

    def test_empty_input_is_zero(self) -> None:
        assert mp_obs._python_mean_phase(np.array([])) == 0.0

    def test_gradient_matches_closed_form(self) -> None:
        rng = np.random.default_rng(17)
        theta = rng.uniform(-math.pi, math.pi, size=15)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        expected = (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)) / (
            theta.size * magnitude**2
        )
        np.testing.assert_allclose(mp_obs._python_mean_phase_gradient(theta), expected, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_sums_to_one(self, n: int, seed: int) -> None:
        # A global phase shift advances ψ identically, so the gradient sums to one.
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert float(np.sum(mp_obs._python_mean_phase_gradient(theta))) == pytest.approx(
            1.0, abs=1e-12
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_gradient_matches_finite_difference(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-2:
            return  # near-incoherent: ψ is ill-conditioned
        np.testing.assert_allclose(
            mp_obs._python_mean_phase_gradient(theta),
            _finite_difference_mean_phase_gradient(theta),
            atol=1e-6,
        )

    def test_single_oscillator_gradient_is_one(self) -> None:
        np.testing.assert_allclose(
            mp_obs._python_mean_phase_gradient(np.array([2.7])), [1.0], atol=1e-15
        )

    def test_aligned_gradient_is_uniform(self) -> None:
        # All oscillators aligned: ψ = θ and ∂ψ/∂θ_j = 1/N for every j.
        grad = mp_obs._python_mean_phase_gradient(np.full(8, 0.7))
        np.testing.assert_allclose(grad, np.full(8, 1.0 / 8), atol=1e-15)

    def test_empty_gradient_is_empty(self) -> None:
        assert mp_obs._python_mean_phase_gradient(np.array([])).shape == (0,)

    def test_exact_incoherent_gradient_is_zero(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(mp_obs._python_mean_phase_gradient(theta), np.zeros(4))


class TestRustMeanPhaseTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "mean_phase_gradient", None)):
            pytest.skip("scpn_quantum_engine.mean_phase_gradient unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        assert mp_obs._rust_mean_phase(theta) == pytest.approx(
            mp_obs._python_mean_phase(theta), abs=1e-12
        )
        np.testing.assert_allclose(
            mp_obs._rust_mean_phase_gradient(theta),
            mp_obs._python_mean_phase_gradient(theta),
            atol=1e-12,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase(np.zeros(3))

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase_gradient(np.zeros(3))


class TestJuliaMeanPhaseTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import mean_phase as julia_value
        from oscillatools.accel.julia import mean_phase_gradient as julia_grad

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        assert julia_value(theta) == pytest.approx(mp_obs._python_mean_phase(theta), abs=1e-10)
        np.testing.assert_allclose(
            julia_grad(theta), mp_obs._python_mean_phase_gradient(theta), atol=1e-10
        )


class TestMeanPhaseDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        value_reference = mp_obs._python_mean_phase(theta)
        grad_reference = mp_obs._python_mean_phase_gradient(theta)
        for name, impl in mp_obs._MEAN_PHASE_CHAIN:
            try:
                assert impl(theta) == pytest.approx(value_reference, abs=1e-10), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        for name, impl in mp_obs._MEAN_PHASE_GRADIENT_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, grad_reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            last_mean_phase_gradient_tier_used,
            last_mean_phase_tier_used,
            mean_phase,
            mean_phase_gradient,
        )

        assert d.dispatch("mean_phase", np.zeros(4)) == pytest.approx(0.0)
        assert d.dispatch("mean_phase_gradient", np.full(4, 0.3)).shape == (4,)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=18)
        assert isinstance(mean_phase(theta), float)
        assert mean_phase_gradient(theta).shape == (18,)
        assert last_mean_phase_tier_used() in {"rust", "julia", "python"}
        assert last_mean_phase_gradient_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert mp_obs._MEAN_PHASE_CHAIN[-1][0] == "python"
        assert mp_obs._MEAN_PHASE_GRADIENT_CHAIN[-1][0] == "python"


class TestMeanPhasePartialEngine:
    def test_partial_engine_value_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [("rust", mp_obs._rust_mean_phase), ("python", mp_obs._python_mean_phase)],
        )
        assert disp(np.zeros(4)) == pytest.approx(0.0)
        assert disp.last_tier == "python"

    def test_partial_engine_gradient_falls_through_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", mp_obs._rust_mean_phase_gradient),
                ("python", mp_obs._python_mean_phase_gradient),
            ],
        )
        np.testing.assert_allclose(
            disp(np.full(4, 0.5)), mp_obs._python_mean_phase_gradient(np.full(4, 0.5))
        )
        assert disp.last_tier == "python"


# ---------------------------------------------------------------------------
# Mean phase Hessian — analytic floor, invariants, parity, and dispatch
# ---------------------------------------------------------------------------


def _finite_difference_mean_phase_hessian(theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference Hessian from the analytic mean-phase gradient floor."""
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[i] += step
        minus[i] -= step
        out[i] = (
            mp_obs._python_mean_phase_gradient(plus) - mp_obs._python_mean_phase_gradient(minus)
        ) / (2.0 * step)
    return out


class TestPythonMeanPhaseHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(41)
        theta = rng.uniform(-math.pi, math.pi, size=11)
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        magnitude = float(np.hypot(cos_mean, sin_mean))
        aligned_cos = (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)) / magnitude
        aligned_sin = (sin_mean * np.cos(theta) - cos_mean * np.sin(theta)) / magnitude
        expected = -(np.outer(aligned_sin, aligned_cos) + np.outer(aligned_cos, aligned_sin)) / (
            theta.size**2 * magnitude**2
        ) + np.diag(aligned_sin / (theta.size * magnitude))
        np.testing.assert_allclose(mp_obs._python_mean_phase_hessian(theta), expected, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_is_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rows_sum_to_zero(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-12)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if _order_parameter_value(theta) < 1e-2:
            return
        hessian = mp_obs._python_mean_phase_hessian(theta)
        np.testing.assert_allclose(
            hessian, _finite_difference_mean_phase_hessian(theta), atol=1e-5
        )

    def test_single_oscillator_is_zero(self) -> None:
        hessian = mp_obs._python_mean_phase_hessian(np.array([2.7]))
        assert hessian.shape == (1, 1)
        assert abs(float(hessian[0, 0])) < 1e-15

    def test_empty_input_returns_empty_matrix(self) -> None:
        assert mp_obs._python_mean_phase_hessian(np.array([])).shape == (0, 0)

    def test_exact_incoherent_returns_zero_matrix(self) -> None:
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(mp_obs._python_mean_phase_hessian(theta), np.zeros((4, 4)))


class TestRustMeanPhaseHessianTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "mean_phase_hessian", None)):
            pytest.skip("scpn_quantum_engine.mean_phase_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            mp_obs._rust_mean_phase_hessian(theta),
            mp_obs._python_mean_phase_hessian(theta),
            atol=1e-11,
        )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mp_obs._rust_mean_phase_hessian(np.zeros(3))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = d.MultiLangDispatcher(
            [
                ("rust", mp_obs._rust_mean_phase_hessian),
                ("python", mp_obs._python_mean_phase_hessian),
            ],
        )
        np.testing.assert_allclose(
            disp(np.full(4, 0.5)), mp_obs._python_mean_phase_hessian(np.full(4, 0.5))
        )
        assert disp.last_tier == "python"


class TestJuliaMeanPhaseHessianTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import mean_phase_hessian as julia_hessian

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        np.testing.assert_allclose(
            julia_hessian(theta), mp_obs._python_mean_phase_hessian(theta), atol=1e-10
        )


class TestMeanPhaseHessianDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = mp_obs._python_mean_phase_hessian(theta)
        for name, impl in mp_obs._MEAN_PHASE_HESSIAN_CHAIN:
            try:
                out = impl(theta)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            last_mean_phase_hessian_tier_used,
            mean_phase_hessian,
        )

        assert d.dispatch("mean_phase_hessian", np.full(4, 0.3)).shape == (4, 4)
        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=16)
        assert mean_phase_hessian(theta).shape == (16, 16)
        assert last_mean_phase_hessian_tier_used() in {"rust", "julia", "python"}

    def test_chain_ends_with_python_floor(self) -> None:
        assert mp_obs._MEAN_PHASE_HESSIAN_CHAIN[-1][0] == "python"
