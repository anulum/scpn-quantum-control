# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Networked Kuramoto coupling force and stability Jacobian tests
"""Multi-angle tests for the networked Kuramoto coupling force and stability Jacobian.

Covers the analytic closed form, the Jacobian as the finite-difference of the force,
symmetry for symmetric coupling and the zero-row-sum (Goldstone) invariant, independence
from the coupling diagonal, the all-to-all mean-field special case, coupling-shape
validation, cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the
public API, partial-engine fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.dispatcher as d
import oscillatools.accel.kuramoto_mean_field as mf
import oscillatools.accel.networked_kuramoto as nk

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _random_coupling(n: int, seed: int, *, symmetric: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 2.0, size=(n, n))
    if symmetric:
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            nk._python_networked_kuramoto_force(plus, coupling)
            - nk._python_networked_kuramoto_force(minus, coupling)
        ) / (2.0 * step)
    return out


class TestPythonNetworkedForceFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        coupling = _random_coupling(7, 5, symmetric=False)
        expected = np.array(
            [
                sum(coupling[j, k] * math.sin(theta[k] - theta[j]) for k in range(theta.size))
                for j in range(theta.size)
            ]
        )
        np.testing.assert_allclose(
            nk._python_networked_kuramoto_force(theta, coupling), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_recovers_mean_field_for_all_to_all_coupling(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        strength = 1.7
        coupling = np.full((n, n), strength / n)
        np.fill_diagonal(coupling, 0.0)
        np.testing.assert_allclose(
            nk._python_networked_kuramoto_force(theta, coupling),
            mf._python_mean_field_force(theta, strength),
            atol=1e-12,
        )

    def test_independent_of_coupling_diagonal(self) -> None:
        rng = np.random.default_rng(7)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 3, symmetric=False)
        baseline = nk._python_networked_kuramoto_force(theta, coupling)
        shifted = coupling.copy()
        np.fill_diagonal(shifted, 9.0)
        np.testing.assert_array_equal(
            nk._python_networked_kuramoto_force(theta, shifted), baseline
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 4"):
            nk._python_networked_kuramoto_force(np.zeros(4), np.zeros((4, 3)))

    def test_empty(self) -> None:
        assert nk._python_networked_kuramoto_force(np.array([]), np.zeros((0, 0))).shape == (0,)


class TestPythonNetworkedJacobianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 8, symmetric=False)
        difference = theta[None, :] - theta[:, None]
        expected = coupling * np.cos(difference)
        np.fill_diagonal(expected, 0.0)
        np.fill_diagonal(expected, -expected.sum(axis=1))
        np.testing.assert_allclose(
            nk._python_networked_kuramoto_jacobian(theta, coupling), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_force(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            nk._python_networked_kuramoto_jacobian(theta, coupling),
            _finite_difference_jacobian(theta, coupling),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=20),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rows_sum_to_zero(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 2, symmetric=symmetric)
        jacobian = nk._python_networked_kuramoto_jacobian(theta, coupling)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-11)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_for_symmetric_coupling(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 3, symmetric=True)
        jacobian = nk._python_networked_kuramoto_jacobian(theta, coupling)
        np.testing.assert_allclose(jacobian, jacobian.T, atol=1e-14)

    def test_independent_of_coupling_diagonal(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.uniform(-math.pi, math.pi, size=5)
        coupling = _random_coupling(5, 6, symmetric=False)
        baseline = nk._python_networked_kuramoto_jacobian(theta, coupling)
        shifted = coupling.copy()
        np.fill_diagonal(shifted, -3.0)
        np.testing.assert_array_equal(
            nk._python_networked_kuramoto_jacobian(theta, shifted), baseline
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            nk._python_networked_kuramoto_jacobian(np.zeros(3), np.zeros((2, 2)))

    def test_empty(self) -> None:
        assert nk._python_networked_kuramoto_jacobian(np.array([]), np.zeros((0, 0))).shape == (
            0,
            0,
        )


class TestRustNetworkedTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, symmetric: bool, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "networked_kuramoto_force", None)):
            pytest.skip("scpn_quantum_engine.networked_kuramoto_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            nk._rust_networked_kuramoto_force(theta, coupling),
            nk._python_networked_kuramoto_force(theta, coupling),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            nk._rust_networked_kuramoto_jacobian(theta, coupling),
            nk._python_networked_kuramoto_jacobian(theta, coupling),
            atol=1e-11,
        )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            nk._rust_networked_kuramoto_force(np.zeros(3), np.zeros((3, 3)))

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            nk._rust_networked_kuramoto_jacobian(np.zeros(3), np.zeros((3, 3)))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        coupling = _random_coupling(4, 1, symmetric=True)
        for chain, floor in (
            (nk._NETWORKED_KURAMOTO_FORCE_CHAIN, nk._python_networked_kuramoto_force),
            (nk._NETWORKED_KURAMOTO_JACOBIAN_CHAIN, nk._python_networked_kuramoto_jacobian),
        ):
            disp = nk.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, coupling)
            np.testing.assert_allclose(out, floor(theta, coupling))
            assert disp.last_tier == "python"


class TestJuliaNetworkedTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import (
            networked_kuramoto_force as julia_force,
        )
        from oscillatools.accel.julia import (
            networked_kuramoto_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for symmetric in (True, False):
            coupling = _random_coupling(8, 2, symmetric=symmetric)
            np.testing.assert_allclose(
                julia_force(theta, coupling),
                nk._python_networked_kuramoto_force(theta, coupling),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                julia_jacobian(theta, coupling),
                nk._python_networked_kuramoto_jacobian(theta, coupling),
                atol=1e-10,
            )


class TestNetworkedDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=False)
        for chain, floor in (
            (nk._NETWORKED_KURAMOTO_FORCE_CHAIN, nk._python_networked_kuramoto_force),
            (nk._NETWORKED_KURAMOTO_JACOBIAN_CHAIN, nk._python_networked_kuramoto_jacobian),
        ):
            reference = floor(theta, coupling)
            for name, impl in chain:
                try:
                    out = impl(theta, coupling)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            last_networked_kuramoto_force_tier_used,
            last_networked_kuramoto_jacobian_tier_used,
            networked_kuramoto_force,
            networked_kuramoto_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        coupling = _random_coupling(10, 4, symmetric=True)
        assert d.dispatch("networked_kuramoto_force", theta, coupling).shape == (10,)
        assert d.dispatch("networked_kuramoto_jacobian", theta, coupling).shape == (10, 10)
        assert networked_kuramoto_force(theta, coupling).shape == (10,)
        assert networked_kuramoto_jacobian(theta, coupling).shape == (10, 10)
        assert last_networked_kuramoto_force_tier_used() in {"rust", "julia", "python"}
        assert last_networked_kuramoto_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert nk._NETWORKED_KURAMOTO_FORCE_CHAIN[-1][0] == "python"
        assert nk._NETWORKED_KURAMOTO_JACOBIAN_CHAIN[-1][0] == "python"
