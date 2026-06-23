# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto mean-field force and stability Jacobian tests
"""Multi-angle tests for the Kuramoto mean-field force and stability Jacobian.

Covers the analytic closed form, the order-parameter identity F = K r sin(ψ − θ), the
Jacobian as the finite-difference derivative of the force, the symmetry and zero-row-sum
(Goldstone) invariants, the full-synchronisation eigenvalue spectrum, cross-tier parity
(Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the public API, partial-engine
fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.kuramoto_mean_field as mf

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _force_via_order_parameter(theta: np.ndarray, coupling: float) -> np.ndarray:
    """Reference K r sin(ψ − θ_j) via the complex order parameter."""
    z = np.mean(np.exp(1j * theta))
    r = float(abs(z))
    psi = float(np.angle(z))
    return coupling * r * np.sin(psi - theta)


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: float, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[k] += step
        minus[k] -= step
        out[:, k] = (
            mf._python_mean_field_force(plus, coupling)
            - mf._python_mean_field_force(minus, coupling)
        ) / (2.0 * step)
    return out


class TestPythonMeanFieldForceFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        coupling = 1.7
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        expected = coupling * (sin_mean * np.cos(theta) - cos_mean * np.sin(theta))
        np.testing.assert_allclose(
            mf._python_mean_field_force(theta, coupling), expected, atol=1e-15
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_equals_order_parameter_identity(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            mf._python_mean_field_force(theta, coupling),
            _force_via_order_parameter(theta, coupling),
            atol=1e-12,
        )

    def test_zero_coupling_and_empty(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        np.testing.assert_array_equal(mf._python_mean_field_force(theta, 0.0), np.zeros(6))
        assert mf._python_mean_field_force(np.array([]), 1.0).shape == (0,)


class TestPythonMeanFieldJacobianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        coupling = 1.3
        cos_mean = float(np.mean(np.cos(theta)))
        sin_mean = float(np.mean(np.sin(theta)))
        expected = (coupling / theta.size) * np.cos(theta[:, None] - theta[None, :])
        expected -= np.diag(coupling * (cos_mean * np.cos(theta) + sin_mean * np.sin(theta)))
        np.testing.assert_allclose(
            mf._python_mean_field_jacobian(theta, coupling), expected, atol=1e-15
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_force(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            mf._python_mean_field_jacobian(theta, coupling),
            _finite_difference_jacobian(theta, coupling),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_and_rows_sum_to_zero(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        jacobian = mf._python_mean_field_jacobian(theta, coupling)
        np.testing.assert_allclose(jacobian, jacobian.T, atol=1e-15)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-11)

    def test_full_synchronisation_spectrum_is_marginally_stable(self) -> None:
        # All phases equal: stable directions have eigenvalue −K, with one zero Goldstone
        # mode (the global phase rotation). For K > 0 synchronisation is marginally stable.
        theta = np.full(8, 0.4)
        eigenvalues = np.linalg.eigvalsh(mf._python_mean_field_jacobian(theta, 1.0))
        assert eigenvalues.max() < 1e-12
        assert abs(eigenvalues[-1]) < 1e-12
        np.testing.assert_allclose(np.sort(eigenvalues)[:-1], np.full(7, -1.0), atol=1e-12)

    def test_zero_coupling_and_empty(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.uniform(-math.pi, math.pi, size=5)
        np.testing.assert_array_equal(mf._python_mean_field_jacobian(theta, 0.0), np.zeros((5, 5)))
        assert mf._python_mean_field_jacobian(np.array([]), 1.0).shape == (0, 0)


class TestRustMeanFieldTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, coupling: float, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "mean_field_force", None)):
            pytest.skip("scpn_quantum_engine.mean_field_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            mf._rust_mean_field_force(theta, coupling),
            mf._python_mean_field_force(theta, coupling),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            mf._rust_mean_field_jacobian(theta, coupling),
            mf._python_mean_field_jacobian(theta, coupling),
            atol=1e-11,
        )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mf._rust_mean_field_force(np.zeros(3), 1.0)

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            mf._rust_mean_field_jacobian(np.zeros(3), 1.0)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        for chain, floor in (
            (mf._MEAN_FIELD_FORCE_CHAIN, mf._python_mean_field_force),
            (mf._MEAN_FIELD_JACOBIAN_CHAIN, mf._python_mean_field_jacobian),
        ):
            disp = mf.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(np.full(4, 0.5), 1.0)
            np.testing.assert_allclose(out, floor(np.full(4, 0.5), 1.0))
            assert disp.last_tier == "python"


class TestJuliaMeanFieldTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import (
            mean_field_force as julia_force,
        )
        from scpn_quantum_control.accel.julia import (
            mean_field_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        for coupling in (1.0, -0.5, 2.5):
            np.testing.assert_allclose(
                julia_force(theta, coupling),
                mf._python_mean_field_force(theta, coupling),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                julia_jacobian(theta, coupling),
                mf._python_mean_field_jacobian(theta, coupling),
                atol=1e-10,
            )


class TestMeanFieldDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        for chain, floor in (
            (mf._MEAN_FIELD_FORCE_CHAIN, mf._python_mean_field_force),
            (mf._MEAN_FIELD_JACOBIAN_CHAIN, mf._python_mean_field_jacobian),
        ):
            reference = floor(theta, coupling)
            for name, impl in chain:
                try:
                    out = impl(theta, coupling)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_mean_field_force_tier_used,
            last_mean_field_jacobian_tier_used,
            mean_field_force,
            mean_field_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert d.dispatch("mean_field_force", theta, 1.5).shape == (12,)
        assert d.dispatch("mean_field_jacobian", theta, 1.5).shape == (12, 12)
        assert mean_field_force(theta, 2.0).shape == (12,)
        assert mean_field_jacobian(theta, 2.0).shape == (12, 12)
        assert last_mean_field_force_tier_used() in {"rust", "julia", "python"}
        assert last_mean_field_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert mf._MEAN_FIELD_FORCE_CHAIN[-1][0] == "python"
        assert mf._MEAN_FIELD_JACOBIAN_CHAIN[-1][0] == "python"
