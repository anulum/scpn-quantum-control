# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Daido m-th-harmonic mean-field force and Jacobian tests
"""Multi-angle tests for the Daido m-th-harmonic mean-field force and stability Jacobian.

Covers the analytic closed form, the reduction to the mean-field force and Jacobian at m = 1,
the Jacobian as the finite-difference of the force, the symmetry and zero-row-sum (Goldstone)
invariants, harmonic-order validation, cross-tier parity (Rust ↔ Julia ↔ Python floor),
name-keyed dispatch and the public API, partial-engine fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.daido_mean_field as dmf
import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.kuramoto_mean_field as mf

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: float, m: int, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            dmf._python_daido_mean_field_force(plus, coupling, m)
            - dmf._python_daido_mean_field_force(minus, coupling, m)
        ) / (2.0 * step)
    return out


class TestPythonDaidoMeanFieldFloor:
    def test_force_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        coupling, m = 1.7, 3
        cos_mean = float(np.mean(np.cos(m * theta)))
        sin_mean = float(np.mean(np.sin(m * theta)))
        expected = coupling * (sin_mean * np.cos(m * theta) - cos_mean * np.sin(m * theta))
        np.testing.assert_allclose(
            dmf._python_daido_mean_field_force(theta, coupling, m), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_m1_reduces_to_mean_field(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            dmf._python_daido_mean_field_force(theta, coupling, 1),
            mf._python_mean_field_force(theta, coupling),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dmf._python_daido_mean_field_jacobian(theta, coupling, 1),
            mf._python_mean_field_jacobian(theta, coupling),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_matches_finite_difference(
        self, n: int, coupling: float, m: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            dmf._python_daido_mean_field_jacobian(theta, coupling, m),
            _finite_difference_jacobian(theta, coupling, m),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_symmetric_and_rows_sum_to_zero(
        self, n: int, coupling: float, m: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        jacobian = dmf._python_daido_mean_field_jacobian(theta, coupling, m)
        np.testing.assert_allclose(jacobian, jacobian.T, atol=1e-14)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-10)

    def test_rejects_non_positive_harmonic(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            dmf._python_daido_mean_field_force(np.zeros(4), 1.0, 0)
        with pytest.raises(ValueError, match="positive integer"):
            dmf._python_daido_mean_field_jacobian(np.zeros(4), 1.0, 0)

    def test_empty(self) -> None:
        assert dmf._python_daido_mean_field_force(np.array([]), 1.0, 2).shape == (0,)
        assert dmf._python_daido_mean_field_jacobian(np.array([]), 1.0, 2).shape == (0, 0)


class TestDaidoMeanFieldTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        m=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, coupling: float, m: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "daido_mean_field_force", None)):
            pytest.skip("scpn_quantum_engine.daido_mean_field_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            dmf._rust_daido_mean_field_force(theta, coupling, m),
            dmf._python_daido_mean_field_force(theta, coupling, m),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            dmf._rust_daido_mean_field_jacobian(theta, coupling, m),
            dmf._python_daido_mean_field_jacobian(theta, coupling, m),
            atol=1e-11,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import daido_mean_field_force as julia_force
        from scpn_quantum_control.accel.julia import (
            daido_mean_field_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for m in (1, 2, 3):
            for coupling in (1.0, -0.5):
                np.testing.assert_allclose(
                    julia_force(theta, coupling, m),
                    dmf._python_daido_mean_field_force(theta, coupling, m),
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    julia_jacobian(theta, coupling, m),
                    dmf._python_daido_mean_field_jacobian(theta, coupling, m),
                    atol=1e-10,
                )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dmf._rust_daido_mean_field_force(np.zeros(3), 1.0, 2)

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dmf._rust_daido_mean_field_jacobian(np.zeros(3), 1.0, 2)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        for chain, floor in (
            (dmf._DAIDO_MEAN_FIELD_FORCE_CHAIN, dmf._python_daido_mean_field_force),
            (dmf._DAIDO_MEAN_FIELD_JACOBIAN_CHAIN, dmf._python_daido_mean_field_jacobian),
        ):
            disp = dmf.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, 1.0, 2)
            np.testing.assert_allclose(out, floor(theta, 1.0, 2))
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
        for chain, floor in (
            (dmf._DAIDO_MEAN_FIELD_FORCE_CHAIN, dmf._python_daido_mean_field_force),
            (dmf._DAIDO_MEAN_FIELD_JACOBIAN_CHAIN, dmf._python_daido_mean_field_jacobian),
        ):
            reference = floor(theta, 1.5, m)
            for name, impl in chain:
                try:
                    out = impl(theta, 1.5, m)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            daido_mean_field_force,
            daido_mean_field_jacobian,
            last_daido_mean_field_force_tier_used,
            last_daido_mean_field_jacobian_tier_used,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert d.dispatch("daido_mean_field_force", theta, 1.5, 2).shape == (12,)
        assert d.dispatch("daido_mean_field_jacobian", theta, 1.5, 2).shape == (12, 12)
        assert daido_mean_field_force(theta, 2.0, 3).shape == (12,)
        assert daido_mean_field_jacobian(theta, 2.0, 3).shape == (12, 12)
        assert last_daido_mean_field_force_tier_used() in {"rust", "julia", "python"}
        assert last_daido_mean_field_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert dmf._DAIDO_MEAN_FIELD_FORCE_CHAIN[-1][0] == "python"
        assert dmf._DAIDO_MEAN_FIELD_JACOBIAN_CHAIN[-1][0] == "python"
