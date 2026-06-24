# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Triadic (2-simplex) Kuramoto mean-field force and Jacobian tests
"""Multi-angle tests for the triadic (2-simplex) Kuramoto mean-field force and Jacobian.

Covers the analytic closed form (``F_j = K r² sin(2ψ − 2θ_j)``), the distinction from the
second Daido harmonic (squared first moment versus second moment), the Jacobian as the
finite-difference of the force, the zero-row-sum Goldstone invariant, the broken symmetry of
the non-variational higher-order coupling, cross-tier parity (Rust ↔ Julia ↔ Python floor),
name-keyed dispatch and the public API, partial-engine fall-through, and the empty edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.triadic_mean_field as tmf

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _finite_difference_jacobian(
    theta: np.ndarray, coupling: float, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        out[:, col] = (
            tmf._python_triadic_mean_field_force(plus, coupling)
            - tmf._python_triadic_mean_field_force(minus, coupling)
        ) / (2.0 * step)
    return out


class TestPythonTriadicMeanFieldFloor:
    def test_force_matches_squared_first_moment(self) -> None:
        rng = np.random.default_rng(13)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        coupling = 1.7
        z = np.mean(np.exp(1j * theta))
        radius_squared = float(abs(z) ** 2)
        psi = float(np.angle(z))
        expected = coupling * radius_squared * np.sin(2.0 * psi - 2.0 * theta)
        np.testing.assert_allclose(
            tmf._python_triadic_mean_field_force(theta, coupling), expected, atol=1e-13
        )

    def test_distinct_from_second_moment_force(self) -> None:
        rng = np.random.default_rng(17)
        theta = rng.uniform(-math.pi, math.pi, size=10)
        triadic = tmf._python_triadic_mean_field_force(theta, 1.0)
        # The triadic force uses the squared first moment ⟨e^{iθ}⟩²; a second-moment force would
        # use ⟨e^{2iθ}⟩. On a generic configuration the moments differ, so the forces differ.
        z = np.mean(np.exp(1j * theta))
        second_moment = np.mean(np.exp(2j * theta))
        assert abs(z * z - second_moment) > 1e-3
        second_moment_force = np.abs(second_moment) * np.sin(np.angle(second_moment) - 2.0 * theta)
        assert not np.allclose(triadic, second_moment_force)

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=18),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_matches_finite_difference(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            tmf._python_triadic_mean_field_jacobian(theta, coupling),
            _finite_difference_jacobian(theta, coupling),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_rows_sum_to_zero(self, n: int, coupling: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        jacobian = tmf._python_triadic_mean_field_jacobian(theta, coupling)
        np.testing.assert_allclose(jacobian.sum(axis=1), np.zeros(n), atol=1e-10)

    def test_higher_order_coupling_breaks_symmetry(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        jacobian = tmf._python_triadic_mean_field_jacobian(theta, 1.3)
        assert np.max(np.abs(jacobian - jacobian.T)) > 1e-3

    def test_empty(self) -> None:
        assert tmf._python_triadic_mean_field_force(np.array([]), 1.0).shape == (0,)
        assert tmf._python_triadic_mean_field_jacobian(np.array([]), 1.0).shape == (0, 0)


class TestTriadicMeanFieldTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=30),
        coupling=st.floats(min_value=-3.0, max_value=3.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, coupling: float, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "triadic_mean_field_force", None)):
            pytest.skip("scpn_quantum_engine.triadic_mean_field_force unavailable")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            tmf._rust_triadic_mean_field_force(theta, coupling),
            tmf._python_triadic_mean_field_force(theta, coupling),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            tmf._rust_triadic_mean_field_jacobian(theta, coupling),
            tmf._python_triadic_mean_field_jacobian(theta, coupling),
            atol=1e-11,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import triadic_mean_field_force as julia_force
        from scpn_quantum_control.accel.julia import (
            triadic_mean_field_jacobian as julia_jacobian,
        )

        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        for coupling in (1.0, -0.5, 2.3):
            np.testing.assert_allclose(
                julia_force(theta, coupling),
                tmf._python_triadic_mean_field_force(theta, coupling),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                julia_jacobian(theta, coupling),
                tmf._python_triadic_mean_field_jacobian(theta, coupling),
                atol=1e-10,
            )

    def test_rust_force_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            tmf._rust_triadic_mean_field_force(np.zeros(3), 1.0)

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            tmf._rust_triadic_mean_field_jacobian(np.zeros(3), 1.0)

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta = np.full(4, 0.5)
        for chain, floor in (
            (tmf._TRIADIC_MEAN_FIELD_FORCE_CHAIN, tmf._python_triadic_mean_field_force),
            (tmf._TRIADIC_MEAN_FIELD_JACOBIAN_CHAIN, tmf._python_triadic_mean_field_jacobian),
        ):
            disp = tmf.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, 1.0)
            np.testing.assert_allclose(out, floor(theta, 1.0))
            assert disp.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        for chain, floor in (
            (tmf._TRIADIC_MEAN_FIELD_FORCE_CHAIN, tmf._python_triadic_mean_field_force),
            (tmf._TRIADIC_MEAN_FIELD_JACOBIAN_CHAIN, tmf._python_triadic_mean_field_jacobian),
        ):
            reference = floor(theta, 1.5)
            for name, impl in chain:
                try:
                    out = impl(theta, 1.5)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_triadic_mean_field_force_tier_used,
            last_triadic_mean_field_jacobian_tier_used,
            triadic_mean_field_force,
            triadic_mean_field_jacobian,
        )

        rng = np.random.default_rng(99)
        theta = rng.uniform(0.0, 2 * math.pi, size=12)
        assert d.dispatch("triadic_mean_field_force", theta, 1.5).shape == (12,)
        assert d.dispatch("triadic_mean_field_jacobian", theta, 1.5).shape == (12, 12)
        assert triadic_mean_field_force(theta, 2.0).shape == (12,)
        assert triadic_mean_field_jacobian(theta, 2.0).shape == (12, 12)
        assert last_triadic_mean_field_force_tier_used() in {"rust", "julia", "python"}
        assert last_triadic_mean_field_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert tmf._TRIADIC_MEAN_FIELD_FORCE_CHAIN[-1][0] == "python"
        assert tmf._TRIADIC_MEAN_FIELD_JACOBIAN_CHAIN[-1][0] == "python"
