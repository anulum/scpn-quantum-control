# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Daido m-th Fourier-mode phase and gradient tests
"""Multi-angle tests for the Daido m-th Fourier-mode phase and its gradient.

Covers the analytic closed form, the reduction to the Kuramoto mean phase at m = 1, the
gradient closed form, the gradient as the finite-difference of the value (with atan2
unwrapping), the sum-to-m invariant, the incoherent-mode subgradient, harmonic-order
validation, cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the
public API, partial-engine fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.daido_phase as dmp
import oscillatools.accel.dispatcher as d
import oscillatools.accel.mean_phase_observables as mp

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _finite_difference_gradient(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    n = theta.size
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        delta = dmp._python_daido_mode_phase(plus, m) - dmp._python_daido_mode_phase(minus, m)
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        out[j] = delta / (2.0 * step)
    return out


class TestPythonModePhaseFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        m = 3
        expected = math.atan2(float(np.sum(np.sin(m * theta))), float(np.sum(np.cos(m * theta))))
        assert dmp._python_daido_mode_phase(theta, m) == pytest.approx(expected, abs=1e-13)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_mean_phase(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        assert dmp._python_daido_mode_phase(theta, 1) == pytest.approx(
            mp._python_mean_phase(theta), abs=1e-12
        )

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            dmp._python_daido_mode_phase(np.zeros(4), 0)

    def test_empty(self) -> None:
        assert dmp._python_daido_mode_phase(np.array([]), 2) == 0.0


class TestPythonModePhaseGradientFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        m = 2
        cos_mean = float(np.mean(np.cos(m * theta)))
        sin_mean = float(np.mean(np.sin(m * theta)))
        magnitude_squared = cos_mean**2 + sin_mean**2
        expected = (m / (theta.size * magnitude_squared)) * (
            cos_mean * np.cos(m * theta) + sin_mean * np.sin(m * theta)
        )
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_gradient(theta, m), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        # keep away from the incoherent mode where the phase is ill-conditioned
        magnitude = abs(np.mean(np.exp(1j * m * theta)))
        if magnitude < 1e-1:
            return
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_gradient(theta, m),
            _finite_difference_gradient(theta, m),
            atol=1e-4,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sums_to_m(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if abs(np.mean(np.exp(1j * m * theta))) < 1e-6:
            return
        gradient = dmp._python_daido_mode_phase_gradient(theta, m)
        assert float(gradient.sum()) == pytest.approx(float(m), abs=1e-9)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_mean_phase_gradient(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_gradient(theta, 1),
            mp._python_mean_phase_gradient(theta),
            atol=1e-12,
        )

    def test_incoherent_mode_has_zero_subgradient(self) -> None:
        # m = 1 on two antipodal phases -> C_1 = S_1 = 0 exactly
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(dmp._python_daido_mode_phase_gradient(theta, 1), np.zeros(4))

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            dmp._python_daido_mode_phase_gradient(np.zeros(4), 0)

    def test_empty(self) -> None:
        assert dmp._python_daido_mode_phase_gradient(np.array([]), 2).shape == (0,)


class TestRustModePhaseTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_mode_phase", None)):
            pytest.skip("scpn_quantum_engine.daido_mode_phase unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        assert dmp._rust_daido_mode_phase(theta, m) == pytest.approx(
            dmp._python_daido_mode_phase(theta, m), abs=1e-11
        )
        np.testing.assert_allclose(
            dmp._rust_daido_mode_phase_gradient(theta, m),
            dmp._python_daido_mode_phase_gradient(theta, m),
            atol=1e-11,
        )

    def test_rust_value_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dmp._rust_daido_mode_phase(np.zeros(3), 2)

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dmp._rust_daido_mode_phase_gradient(np.zeros(3), 2)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        for chain, floor in (
            (dmp._DAIDO_MODE_PHASE_CHAIN, dmp._python_daido_mode_phase),
            (dmp._DAIDO_MODE_PHASE_GRADIENT_CHAIN, dmp._python_daido_mode_phase_gradient),
        ):
            disp = dmp.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, 2)
            np.testing.assert_allclose(out, floor(theta, 2))
            assert disp.last_tier == "python"


class TestJuliaModePhaseTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import daido_mode_phase as julia_phase
        from oscillatools.accel.julia import (
            daido_mode_phase_gradient as julia_gradient,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for m in (1, 2, 3):
            assert julia_phase(theta, m) == pytest.approx(
                dmp._python_daido_mode_phase(theta, m), abs=1e-10
            )
            np.testing.assert_allclose(
                julia_gradient(theta, m),
                dmp._python_daido_mode_phase_gradient(theta, m),
                atol=1e-10,
            )


class TestModePhaseDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        value_reference = dmp._python_daido_mode_phase(theta, m)
        for name, impl in dmp._DAIDO_MODE_PHASE_CHAIN:
            try:
                assert impl(theta, m) == pytest.approx(value_reference, abs=1e-10), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        gradient_reference = dmp._python_daido_mode_phase_gradient(theta, m)
        for name, impl in dmp._DAIDO_MODE_PHASE_GRADIENT_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, gradient_reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            daido_mode_phase,
            daido_mode_phase_gradient,
            last_daido_mode_phase_gradient_tier_used,
            last_daido_mode_phase_tier_used,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert isinstance(d.dispatch("daido_mode_phase", theta, 2), float)
        assert d.dispatch("daido_mode_phase_gradient", theta, 2).shape == (12,)
        assert isinstance(daido_mode_phase(theta, 3), float)
        assert daido_mode_phase_gradient(theta, 3).shape == (12,)
        assert last_daido_mode_phase_tier_used() in {"rust", "julia", "python"}
        assert last_daido_mode_phase_gradient_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert dmp._DAIDO_MODE_PHASE_CHAIN[-1][0] == "python"
        assert dmp._DAIDO_MODE_PHASE_GRADIENT_CHAIN[-1][0] == "python"


def _finite_difference_hessian(theta: np.ndarray, m: int, step: float = 1e-6) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            dmp._python_daido_mode_phase_gradient(plus, m)
            - dmp._python_daido_mode_phase_gradient(minus, m)
        ) / (2.0 * step)
    return out


class TestPythonModePhaseHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(13)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        m = 2
        cos_mean = float(np.mean(np.cos(m * theta)))
        sin_mean = float(np.mean(np.sin(m * theta)))
        magnitude = math.hypot(cos_mean, sin_mean)
        s = (sin_mean * np.cos(m * theta) - cos_mean * np.sin(m * theta)) / magnitude
        c = (cos_mean * np.cos(m * theta) + sin_mean * np.sin(m * theta)) / magnitude
        expected = (m * m) * (
            np.diag(s / (theta.size * magnitude))
            - (np.outer(s, c) + np.outer(c, s)) / (theta.size**2 * magnitude**2)
        )
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_hessian(theta, m), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=32),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_mean_phase_hessian(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_hessian(theta, 1),
            mp._python_mean_phase_hessian(theta),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        if abs(np.mean(np.exp(1j * m * theta))) < 1e-1:
            return
        np.testing.assert_allclose(
            dmp._python_daido_mode_phase_hessian(theta, m),
            _finite_difference_hessian(theta, m),
            atol=1e-4,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_and_rows_sum_to_zero(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        hessian = dmp._python_daido_mode_phase_hessian(theta, m)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-14)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-10)

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            dmp._python_daido_mode_phase_hessian(np.zeros(4), 0)

    def test_empty_and_incoherent(self) -> None:
        assert dmp._python_daido_mode_phase_hessian(np.array([]), 2).shape == (0, 0)
        theta = np.array([0.0, math.pi, 0.0, -math.pi])
        np.testing.assert_array_equal(
            dmp._python_daido_mode_phase_hessian(theta, 1), np.zeros((4, 4))
        )


class TestModePhaseHessianTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_mode_phase_hessian", None)):
            pytest.skip("scpn_quantum_engine.daido_mode_phase_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            dmp._rust_daido_mode_phase_hessian(theta, m),
            dmp._python_daido_mode_phase_hessian(theta, m),
            atol=1e-10,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import (
            daido_mode_phase_hessian as julia_hessian,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        for m in (1, 2, 3):
            np.testing.assert_allclose(
                julia_hessian(theta, m),
                dmp._python_daido_mode_phase_hessian(theta, m),
                atol=1e-10,
            )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dmp._rust_daido_mode_phase_hessian(np.zeros(3), 2)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        disp = dmp.MultiLangDispatcher(
            [dmp._DAIDO_MODE_PHASE_HESSIAN_CHAIN[0], dmp._DAIDO_MODE_PHASE_HESSIAN_CHAIN[-1]]
        )
        out = disp(np.full(4, 0.5), 2)
        np.testing.assert_allclose(out, dmp._python_daido_mode_phase_hessian(np.full(4, 0.5), 2))
        assert disp.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        m=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, m: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        reference = dmp._python_daido_mode_phase_hessian(theta, m)
        for name, impl in dmp._DAIDO_MODE_PHASE_HESSIAN_CHAIN:
            try:
                out = impl(theta, m)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            daido_mode_phase_hessian,
            last_daido_mode_phase_hessian_tier_used,
        )

        rng = np.random.default_rng(55)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert d.dispatch("daido_mode_phase_hessian", theta, 2).shape == (12, 12)
        assert daido_mode_phase_hessian(theta, 3).shape == (12, 12)
        assert last_daido_mode_phase_hessian_tier_used() in {"rust", "julia", "python"}

    def test_chain_ends_with_python_floor(self) -> None:
        assert dmp._DAIDO_MODE_PHASE_HESSIAN_CHAIN[-1][0] == "python"
