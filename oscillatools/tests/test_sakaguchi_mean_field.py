# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Sakaguchi–Kuramoto mean-field force and Jacobian tests
"""Multi-angle tests for the Sakaguchi–Kuramoto mean-field force and stability Jacobian.

Covers the analytic closed form (``F_j = K r sin(ψ − θ_j − α)``), the reduction to the
mean-field force and Jacobian at zero frustration, the Jacobian as the finite-difference of the
force, the zero-row-sum Goldstone invariant, the broken symmetry under frustration, cross-tier
parity (Rust ↔ Julia ↔ Python floor), name-keyed dispatch and the public API, partial-engine
fall-through, and the empty edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import oscillatools.accel.dispatcher as d
import oscillatools.accel.kuramoto_mean_field as mf
import oscillatools.accel.sakaguchi_mean_field as smf

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: float, frustration: float, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            smf._python_sakaguchi_mean_field_force(plus, coupling, frustration)
            - smf._python_sakaguchi_mean_field_force(minus, coupling, frustration)
        ) / (2.0 * step)
    return out


class TestPythonSakaguchiMeanFieldFloor:
    def test_force_matches_closed_form(self) -> None:
        rng = np.random.default_rng(13)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        coupling, alpha = 1.7, 0.8
        radius = float(abs(np.mean(np.exp(1j * theta))))
        psi = float(np.angle(np.mean(np.exp(1j * theta))))
        expected = coupling * radius * np.sin(psi - theta - alpha)
        np.testing.assert_allclose(
            smf._python_sakaguchi_mean_field_force(theta, coupling, alpha), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=40),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_zero_frustration_reduces_to_mean_field(
        self, n: int, coupling: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            smf._python_sakaguchi_mean_field_force(theta, coupling, 0.0),
            mf._python_mean_field_force(theta, coupling),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            smf._python_sakaguchi_mean_field_jacobian(theta, coupling, 0.0),
            mf._python_mean_field_jacobian(theta, coupling),
            atol=1e-12,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        alpha=st.floats(min_value=-math.pi, max_value=math.pi),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_matches_finite_difference(
        self, n: int, coupling: float, alpha: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            smf._python_sakaguchi_mean_field_jacobian(theta, coupling, alpha),
            _finite_difference_jacobian(theta, coupling, alpha),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        alpha=st.floats(min_value=-math.pi, max_value=math.pi),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_rows_sum_to_zero(
        self, n: int, coupling: float, alpha: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        jacobian = smf._python_sakaguchi_mean_field_jacobian(theta, coupling, alpha)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-10)

    def test_frustration_breaks_symmetry(self) -> None:
        rng = np.random.default_rng(5)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        symmetric = smf._python_sakaguchi_mean_field_jacobian(theta, 1.5, 0.0)
        frustrated = smf._python_sakaguchi_mean_field_jacobian(theta, 1.5, 0.7)
        np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-14)
        assert np.max(np.abs(frustrated - frustrated.T)) > 1e-3

    def test_empty(self) -> None:
        assert smf._python_sakaguchi_mean_field_force(np.array([]), 1.0, 0.5).shape == (0,)
        assert smf._python_sakaguchi_mean_field_jacobian(np.array([]), 1.0, 0.5).shape == (0, 0)


class TestSakaguchiMeanFieldTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        alpha=st.floats(min_value=-math.pi, max_value=math.pi),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(
        self, n: int, coupling: float, alpha: float, seed: int
    ) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "sakaguchi_mean_field_force", None)):
            pytest.skip("scpn_quantum_engine.sakaguchi_mean_field_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            smf._rust_sakaguchi_mean_field_force(theta, coupling, alpha),
            smf._python_sakaguchi_mean_field_force(theta, coupling, alpha),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            smf._rust_sakaguchi_mean_field_jacobian(theta, coupling, alpha),
            smf._python_sakaguchi_mean_field_jacobian(theta, coupling, alpha),
            atol=1e-11,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from oscillatools.accel.julia import sakaguchi_mean_field_force as julia_force
        from oscillatools.accel.julia import (
            sakaguchi_mean_field_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for alpha in (0.0, 0.5, -1.2):
            for coupling in (1.0, -0.5):
                np.testing.assert_allclose(
                    julia_force(theta, coupling, alpha),
                    smf._python_sakaguchi_mean_field_force(theta, coupling, alpha),
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    julia_jacobian(theta, coupling, alpha),
                    smf._python_sakaguchi_mean_field_jacobian(theta, coupling, alpha),
                    atol=1e-10,
                )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            smf._rust_sakaguchi_mean_field_force(np.zeros(3), 1.0, 0.5)

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            smf._rust_sakaguchi_mean_field_jacobian(np.zeros(3), 1.0, 0.5)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        for chain, floor in (
            (smf._SAKAGUCHI_MEAN_FIELD_FORCE_CHAIN, smf._python_sakaguchi_mean_field_force),
            (
                smf._SAKAGUCHI_MEAN_FIELD_JACOBIAN_CHAIN,
                smf._python_sakaguchi_mean_field_jacobian,
            ),
        ):
            disp = smf.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, 1.0, 0.5)
            np.testing.assert_allclose(out, floor(theta, 1.0, 0.5))
            assert disp.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        alpha=st.floats(min_value=-math.pi, max_value=math.pi),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, alpha: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        for chain, floor in (
            (smf._SAKAGUCHI_MEAN_FIELD_FORCE_CHAIN, smf._python_sakaguchi_mean_field_force),
            (
                smf._SAKAGUCHI_MEAN_FIELD_JACOBIAN_CHAIN,
                smf._python_sakaguchi_mean_field_jacobian,
            ),
        ):
            reference = floor(theta, 1.5, alpha)
            for name, impl in chain:
                try:
                    out = impl(theta, 1.5, alpha)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from oscillatools.accel import (
            last_sakaguchi_mean_field_force_tier_used,
            last_sakaguchi_mean_field_jacobian_tier_used,
            sakaguchi_mean_field_force,
            sakaguchi_mean_field_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert d.dispatch("sakaguchi_mean_field_force", theta, 1.5, 0.5).shape == (12,)
        assert d.dispatch("sakaguchi_mean_field_jacobian", theta, 1.5, 0.5).shape == (12, 12)
        assert sakaguchi_mean_field_force(theta, 2.0, 0.3).shape == (12,)
        assert sakaguchi_mean_field_jacobian(theta, 2.0, 0.3).shape == (12, 12)
        assert last_sakaguchi_mean_field_force_tier_used() in {"rust", "julia", "python"}
        assert last_sakaguchi_mean_field_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert smf._SAKAGUCHI_MEAN_FIELD_FORCE_CHAIN[-1][0] == "python"
        assert smf._SAKAGUCHI_MEAN_FIELD_JACOBIAN_CHAIN[-1][0] == "python"
