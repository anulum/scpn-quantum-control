# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Kuramoto interaction energy and gradient tests
"""Multi-angle tests for the Kuramoto interaction energy and its gradient.

Covers the analytic closed form, the gradient as the finite-difference of the energy, the
zero-sum (Goldstone) invariant, the gradient-equals-negated-networked-force identity for
symmetric coupling, the full-synchronisation minimum, coupling-shape validation, cross-tier
parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the public API, partial-engine
fall-through, and the empty/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.dispatcher as d
import oscillatools.accel.kuramoto_energy as ke
import oscillatools.accel.networked_kuramoto as nk

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _random_coupling(n: int, seed: int, *, symmetric: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.0, 2.0, size=(n, n))
    if symmetric:
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


def _finite_difference_gradient(
    theta: np.ndarray, coupling: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[j] += step
        minus[j] -= step
        out[j] = (
            ke._python_kuramoto_interaction_energy(plus, coupling)
            - ke._python_kuramoto_interaction_energy(minus, coupling)
        ) / (2.0 * step)
    return out


class TestPythonEnergyFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(11)
        theta = rng.uniform(-math.pi, math.pi, size=7)
        coupling = _random_coupling(7, 5, symmetric=False)
        expected = -0.5 * sum(
            coupling[j, k] * math.cos(theta[j] - theta[k])
            for j in range(theta.size)
            for k in range(theta.size)
        )
        assert ke._python_kuramoto_interaction_energy(theta, coupling) == pytest.approx(
            expected, abs=1e-12
        )

    def test_full_synchronisation_is_minimum(self) -> None:
        coupling = _random_coupling(8, 2, symmetric=True)
        synced = np.full(8, 0.7)
        energy_synced = ke._python_kuramoto_interaction_energy(synced, coupling)
        rng = np.random.default_rng(0)
        for _ in range(20):
            perturbed = synced + rng.uniform(-0.3, 0.3, size=8)
            assert energy_synced <= ke._python_kuramoto_interaction_energy(perturbed, coupling)

    def test_depends_only_on_symmetric_part(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 4, symmetric=False)
        symmetrised = 0.5 * (coupling + coupling.T)
        assert ke._python_kuramoto_interaction_energy(theta, coupling) == pytest.approx(
            ke._python_kuramoto_interaction_energy(theta, symmetrised), abs=1e-12
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 4"):
            ke._python_kuramoto_interaction_energy(np.zeros(4), np.zeros((4, 3)))

    def test_empty(self) -> None:
        assert ke._python_kuramoto_interaction_energy(np.array([]), np.zeros((0, 0))) == 0.0


class TestPythonEnergyGradientFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 8, symmetric=False)
        difference = theta[:, None] - theta[None, :]
        expected = 0.5 * np.sum((coupling + coupling.T) * np.sin(difference), axis=1)
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_gradient(theta, coupling), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_gradient(theta, coupling),
            _finite_difference_gradient(theta, coupling),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_sums_to_zero(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 2, symmetric=symmetric)
        gradient = ke._python_kuramoto_interaction_energy_gradient(theta, coupling)
        assert abs(float(gradient.sum())) < 1e-10

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_equals_negated_force_for_symmetric_coupling(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 3, symmetric=True)
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_gradient(theta, coupling),
            -nk._python_networked_kuramoto_force(theta, coupling),
            atol=1e-12,
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            ke._python_kuramoto_interaction_energy_gradient(np.zeros(3), np.zeros((2, 2)))

    def test_empty(self) -> None:
        assert ke._python_kuramoto_interaction_energy_gradient(
            np.array([]), np.zeros((0, 0))
        ).shape == (0,)


class TestRustEnergyTier:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, symmetric: bool, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "kuramoto_interaction_energy", None)):
            pytest.skip("scpn_quantum_engine.kuramoto_interaction_energy unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        assert ke._rust_kuramoto_interaction_energy(theta, coupling) == pytest.approx(
            ke._python_kuramoto_interaction_energy(theta, coupling), abs=1e-9, rel=1e-11
        )
        np.testing.assert_allclose(
            ke._rust_kuramoto_interaction_energy_gradient(theta, coupling),
            ke._python_kuramoto_interaction_energy_gradient(theta, coupling),
            atol=1e-10,
        )

    def test_rust_energy_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            ke._rust_kuramoto_interaction_energy(np.zeros(3), np.zeros((3, 3)))

    def test_rust_gradient_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            ke._rust_kuramoto_interaction_energy_gradient(np.zeros(3), np.zeros((3, 3)))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        coupling = _random_coupling(4, 1, symmetric=True)
        energy_disp = ke.MultiLangDispatcher(
            [ke._KURAMOTO_INTERACTION_ENERGY_CHAIN[0], ke._KURAMOTO_INTERACTION_ENERGY_CHAIN[-1]]
        )
        assert energy_disp(theta, coupling) == pytest.approx(
            ke._python_kuramoto_interaction_energy(theta, coupling)
        )
        assert energy_disp.last_tier == "python"
        gradient_disp = ke.MultiLangDispatcher(
            [
                ke._KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN[0],
                ke._KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN[-1],
            ]
        )
        np.testing.assert_allclose(
            gradient_disp(theta, coupling),
            ke._python_kuramoto_interaction_energy_gradient(theta, coupling),
        )
        assert gradient_disp.last_tier == "python"


class TestJuliaEnergyTier:
    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import (
            kuramoto_interaction_energy as julia_energy,
        )
        from oscillatools.accel.julia import (
            kuramoto_interaction_energy_gradient as julia_gradient,
        )

        rng = np.random.default_rng(20260623)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for symmetric in (True, False):
            coupling = _random_coupling(8, 2, symmetric=symmetric)
            assert julia_energy(theta, coupling) == pytest.approx(
                ke._python_kuramoto_interaction_energy(theta, coupling), abs=1e-9
            )
            np.testing.assert_allclose(
                julia_gradient(theta, coupling),
                ke._python_kuramoto_interaction_energy_gradient(theta, coupling),
                atol=1e-10,
            )


class TestEnergyDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=False)
        energy_reference = ke._python_kuramoto_interaction_energy(theta, coupling)
        for name, impl in ke._KURAMOTO_INTERACTION_ENERGY_CHAIN:
            try:
                assert impl(theta, coupling) == pytest.approx(
                    energy_reference, abs=1e-9, rel=1e-11
                ), name
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
        gradient_reference = ke._python_kuramoto_interaction_energy_gradient(theta, coupling)
        for name, impl in ke._KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN:
            try:
                out = impl(theta, coupling)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, gradient_reference, atol=1e-9, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            kuramoto_interaction_energy,
            kuramoto_interaction_energy_gradient,
            last_kuramoto_interaction_energy_gradient_tier_used,
            last_kuramoto_interaction_energy_tier_used,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        coupling = _random_coupling(10, 4, symmetric=True)
        assert isinstance(d.dispatch("kuramoto_interaction_energy", theta, coupling), float)
        assert d.dispatch("kuramoto_interaction_energy_gradient", theta, coupling).shape == (10,)
        assert isinstance(kuramoto_interaction_energy(theta, coupling), float)
        assert kuramoto_interaction_energy_gradient(theta, coupling).shape == (10,)
        assert last_kuramoto_interaction_energy_tier_used() in {"rust", "julia", "python"}
        assert last_kuramoto_interaction_energy_gradient_tier_used() in {
            "rust",
            "julia",
            "python",
        }

    def test_chains_end_with_python_floor(self) -> None:
        assert ke._KURAMOTO_INTERACTION_ENERGY_CHAIN[-1][0] == "python"
        assert ke._KURAMOTO_INTERACTION_ENERGY_GRADIENT_CHAIN[-1][0] == "python"


def _finite_difference_hessian(
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
            ke._python_kuramoto_interaction_energy_gradient(plus, coupling)
            - ke._python_kuramoto_interaction_energy_gradient(minus, coupling)
        ) / (2.0 * step)
    return out


class TestPythonEnergyHessianFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(13)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        coupling = _random_coupling(6, 9, symmetric=False)
        symmetrised = coupling + coupling.T
        difference = theta[:, None] - theta[None, :]
        expected = -0.5 * symmetrised * np.cos(difference)
        np.fill_diagonal(expected, 0.0)
        np.fill_diagonal(expected, -expected.sum(axis=1))
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_hessian(theta, coupling), expected, atol=1e-14
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_matches_finite_difference_of_gradient(
        self, n: int, symmetric: bool, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_hessian(theta, coupling),
            _finite_difference_hessian(theta, coupling),
            atol=1e-4,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_symmetric_and_rows_sum_to_zero(self, n: int, symmetric: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 2, symmetric=symmetric)
        hessian = ke._python_kuramoto_interaction_energy_hessian(theta, coupling)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-14)
        np.testing.assert_allclose(hessian.sum(axis=1), np.zeros(n), atol=1e-10)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_equals_negated_jacobian_for_symmetric_coupling(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 3, symmetric=True)
        np.testing.assert_allclose(
            ke._python_kuramoto_interaction_energy_hessian(theta, coupling),
            -nk._python_networked_kuramoto_jacobian(theta, coupling),
            atol=1e-12,
        )

    def test_rejects_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            ke._python_kuramoto_interaction_energy_hessian(np.zeros(3), np.zeros((2, 2)))

    def test_empty(self) -> None:
        assert ke._python_kuramoto_interaction_energy_hessian(
            np.array([]), np.zeros((0, 0))
        ).shape == (0, 0)


class TestEnergyHessianTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        symmetric=st.booleans(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, symmetric: bool, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "kuramoto_interaction_energy_hessian", None)):
            pytest.skip("scpn_quantum_engine.kuramoto_interaction_energy_hessian unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=symmetric)
        np.testing.assert_allclose(
            ke._rust_kuramoto_interaction_energy_hessian(theta, coupling),
            ke._python_kuramoto_interaction_energy_hessian(theta, coupling),
            atol=1e-11,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import (
            kuramoto_interaction_energy_hessian as julia_hessian,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for symmetric in (True, False):
            coupling = _random_coupling(8, 2, symmetric=symmetric)
            np.testing.assert_allclose(
                julia_hessian(theta, coupling),
                ke._python_kuramoto_interaction_energy_hessian(theta, coupling),
                atol=1e-10,
            )

    def test_rust_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            ke._rust_kuramoto_interaction_energy_hessian(np.zeros(3), np.zeros((3, 3)))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        coupling = _random_coupling(4, 1, symmetric=True)
        disp = ke.MultiLangDispatcher(
            [
                ke._KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN[0],
                ke._KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN[-1],
            ]
        )
        np.testing.assert_allclose(
            disp(theta, coupling),
            ke._python_kuramoto_interaction_energy_hessian(theta, coupling),
        )
        assert disp.last_tier == "python"

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            kuramoto_interaction_energy_hessian,
            last_kuramoto_interaction_energy_hessian_tier_used,
        )

        rng = np.random.default_rng(77)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        coupling = _random_coupling(10, 4, symmetric=True)
        assert d.dispatch("kuramoto_interaction_energy_hessian", theta, coupling).shape == (
            10,
            10,
        )
        assert kuramoto_interaction_energy_hessian(theta, coupling).shape == (10, 10)
        assert last_kuramoto_interaction_energy_hessian_tier_used() in {"rust", "julia", "python"}

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        coupling = _random_coupling(n, seed + 1, symmetric=False)
        reference = ke._python_kuramoto_interaction_energy_hessian(theta, coupling)
        for name, impl in ke._KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN:
            try:
                out = impl(theta, coupling)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_chain_ends_with_python_floor(self) -> None:
        assert ke._KURAMOTO_INTERACTION_ENERGY_HESSIAN_CHAIN[-1][0] == "python"
