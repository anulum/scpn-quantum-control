# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Kuramoto–Sakaguchi frustrated force and stability Jacobian tests
"""Multi-angle tests for the Kuramoto–Sakaguchi frustrated force and stability Jacobian.

Covers the analytic closed form, the Jacobian as the finite-difference of the force, the
zero-row-sum (Goldstone) invariant, the frustration-induced asymmetry for symmetric coupling,
the reduction to the networked force at zero frustration, coupling-shape validation,
cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the public API,
partial-engine fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.dispatcher as d
import oscillatools.accel.networked_kuramoto as nk
import oscillatools.accel.sakaguchi_kuramoto as sk

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _random_coupling(n: int, seed: int, *, symmetric: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 2.0, size=(n, n))
    if symmetric:
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: np.ndarray, frustration: float, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            sk._python_sakaguchi_force(plus, coupling, frustration)
            - sk._python_sakaguchi_force(minus, coupling, frustration)
        ) / (2.0 * step)
    return out


class TestPythonSakaguchiForceFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        coupling = _random_coupling(7, 5, symmetric=False)
        alpha = 0.4
        expected = np.array(
            [
                sum(
                    coupling[j, k] * math.sin(theta[k] - theta[j] - alpha)
                    for k in range(theta.size)
                    if k != j
                )
                for j in range(theta.size)
            ]
        )
        np.testing.assert_allclose(
            sk._python_sakaguchi_force(theta, coupling, alpha), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_reduces_to_networked_at_zero_frustration(
        self, n: int, symmetric: bool, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            sk._python_sakaguchi_force(theta, coupling, 0.0),
            nk._python_networked_kuramoto_force(theta, coupling),
            atol=1e-12,
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 4"):
            sk._python_sakaguchi_force(np.zeros(4), np.zeros((4, 3)), 0.3)

    def test_empty(self) -> None:
        assert sk._python_sakaguchi_force(np.array([]), np.zeros((0, 0)), 0.5).shape == (0,)


class TestPythonSakaguchiJacobianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 8, symmetric=False)
        alpha = 0.7
        difference = theta[None, :] - theta[:, None] - alpha
        expected = coupling * np.cos(difference)
        np.fill_diagonal(expected, 0.0)
        np.fill_diagonal(expected, -expected.sum(axis=1))
        np.testing.assert_allclose(
            sk._python_sakaguchi_jacobian(theta, coupling, alpha), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        symmetric=st.booleans(),
        alpha=st.floats(min_value=-2.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_force(
        self, n: int, symmetric: bool, alpha: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            sk._python_sakaguchi_jacobian(theta, coupling, alpha),
            _finite_difference_jacobian(theta, coupling, alpha),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=20),
        symmetric=st.booleans(),
        alpha=st.floats(min_value=-2.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rows_sum_to_zero(self, n: int, symmetric: bool, alpha: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 2, symmetric=symmetric)
        jacobian = sk._python_sakaguchi_jacobian(theta, coupling, alpha)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-11)

    def test_frustration_breaks_symmetry_for_symmetric_coupling(self) -> None:
        rng = np.random.default_rng(2)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        coupling = _random_coupling(8, 3, symmetric=True)
        symmetric_jacobian = sk._python_sakaguchi_jacobian(theta, coupling, 0.0)
        np.testing.assert_allclose(symmetric_jacobian, symmetric_jacobian.T, atol=1e-14)
        frustrated = sk._python_sakaguchi_jacobian(theta, coupling, 0.6)
        assert np.max(np.abs(frustrated - frustrated.T)) > 1e-2

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            sk._python_sakaguchi_jacobian(np.zeros(3), np.zeros((2, 2)), 0.3)

    def test_empty(self) -> None:
        assert sk._python_sakaguchi_jacobian(np.array([]), np.zeros((0, 0)), 0.5).shape == (0, 0)


class TestRustSakaguchiTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        symmetric=st.booleans(),
        alpha=st.floats(min_value=-2.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(
        self, n: int, symmetric: bool, alpha: float, seed: int
    ) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "sakaguchi_force", None)):
            pytest.skip("scpn_quantum_engine.sakaguchi_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            sk._rust_sakaguchi_force(theta, coupling, alpha),
            sk._python_sakaguchi_force(theta, coupling, alpha),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            sk._rust_sakaguchi_jacobian(theta, coupling, alpha),
            sk._python_sakaguchi_jacobian(theta, coupling, alpha),
            atol=1e-11,
        )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            sk._rust_sakaguchi_force(np.zeros(3), np.zeros((3, 3)), 0.3)

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            sk._rust_sakaguchi_jacobian(np.zeros(3), np.zeros((3, 3)), 0.3)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        coupling = _random_coupling(4, 1, symmetric=True)
        for chain, floor in (
            (sk._SAKAGUCHI_FORCE_CHAIN, sk._python_sakaguchi_force),
            (sk._SAKAGUCHI_JACOBIAN_CHAIN, sk._python_sakaguchi_jacobian),
        ):
            disp = sk.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, coupling, 0.4)
            np.testing.assert_allclose(out, floor(theta, coupling, 0.4))
            assert disp.last_tier == "python"


class TestJuliaSakaguchiTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import sakaguchi_force as julia_force
        from oscillatools.accel.julia import sakaguchi_jacobian as julia_jacobian

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for alpha in (0.0, 0.5, -1.1):
            for symmetric in (True, False):
                coupling = _random_coupling(8, 2, symmetric=symmetric)
                np.testing.assert_allclose(
                    julia_force(theta, coupling, alpha),
                    sk._python_sakaguchi_force(theta, coupling, alpha),
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    julia_jacobian(theta, coupling, alpha),
                    sk._python_sakaguchi_jacobian(theta, coupling, alpha),
                    atol=1e-10,
                )


class TestSakaguchiDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        alpha=st.floats(min_value=-2.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, alpha: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=False)
        for chain, floor in (
            (sk._SAKAGUCHI_FORCE_CHAIN, sk._python_sakaguchi_force),
            (sk._SAKAGUCHI_JACOBIAN_CHAIN, sk._python_sakaguchi_jacobian),
        ):
            reference = floor(theta, coupling, alpha)
            for name, impl in chain:
                try:
                    out = impl(theta, coupling, alpha)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            last_sakaguchi_force_tier_used,
            last_sakaguchi_jacobian_tier_used,
            sakaguchi_force,
            sakaguchi_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        coupling = _random_coupling(10, 4, symmetric=True)
        assert d.dispatch("sakaguchi_force", theta, coupling, 0.3).shape == (10,)
        assert d.dispatch("sakaguchi_jacobian", theta, coupling, 0.3).shape == (10, 10)
        assert sakaguchi_force(theta, coupling, 0.3).shape == (10,)
        assert sakaguchi_jacobian(theta, coupling, 0.3).shape == (10, 10)
        assert last_sakaguchi_force_tier_used() in {"rust", "julia", "python"}
        assert last_sakaguchi_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert sk._SAKAGUCHI_FORCE_CHAIN[-1][0] == "python"
        assert sk._SAKAGUCHI_JACOBIAN_CHAIN[-1][0] == "python"
