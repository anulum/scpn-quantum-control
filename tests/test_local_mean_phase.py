# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Network-local Kuramoto mean phase and Jacobian tests
"""Multi-angle tests for the network-local Kuramoto mean phase and its Jacobian.

Covers the analytic closed form, the all-to-all reduction to the global mean phase of the
remaining oscillators, the Jacobian as the unwrapped finite-difference of the phase, the
sparsity in the adjacency pattern, the zero-degree and incoherent-neighbourhood subgradient,
the consistency of the local complex order ``Z_j = r_j e^{iψ_j}`` with the local order
parameter, square-shape validation, cross-tier parity (Rust ↔ Julia ↔ Python floor),
name-keyed dispatch and the public API, partial-engine fall-through, and the empty edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.dispatcher as d
import scpn_quantum_control.accel.local_order as lo
import scpn_quantum_control.accel.local_phase as lp

_GLOBAL_SETTINGS = settings(max_examples=40, deadline=None)


def _random_adjacency(n: int, seed: int, density: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = (rng.random((n, n)) < density).astype(np.float64)
    weights = rng.uniform(0.1, 2.0, size=(n, n))
    adjacency = mask * weights
    adjacency = adjacency + adjacency.T
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _unwrapped_finite_difference(
    theta: np.ndarray, adjacency: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    n = theta.size
    out = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        plus = theta.astype(np.float64).copy()
        minus = theta.astype(np.float64).copy()
        plus[col] += step
        minus[col] -= step
        delta = lp._python_local_mean_phase(plus, adjacency) - lp._python_local_mean_phase(
            minus, adjacency
        )
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        out[:, col] = delta / (2.0 * step)
    return out


class TestPythonLocalMeanPhaseFloor:
    def test_matches_closed_form(self) -> None:
        rng = np.random.default_rng(7)
        adjacency = _random_adjacency(6, 7)
        theta = rng.uniform(-math.pi, math.pi, size=6)
        cos_sum = adjacency @ np.cos(theta)
        sin_sum = adjacency @ np.sin(theta)
        expected = np.arctan2(sin_sum, cos_sum)
        np.testing.assert_allclose(
            lp._python_local_mean_phase(theta, adjacency), expected, atol=1e-13
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_to_all_excludes_only_self(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        adjacency = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)
        phases = lp._python_local_mean_phase(theta, adjacency)
        for j in range(n):
            others = np.delete(theta, j)
            expected = math.atan2(float(np.sin(others).sum()), float(np.cos(others).sum()))
            assert abs(phases[j] - expected) < 1e-12

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=14),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_matches_finite_difference(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        adjacency = _random_adjacency(n, seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        np.testing.assert_allclose(
            lp._python_local_mean_phase_jacobian(theta, adjacency),
            _unwrapped_finite_difference(theta, adjacency),
            atol=1e-5,
        )

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_jacobian_is_sparse_in_adjacency_pattern(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        adjacency = _random_adjacency(n, seed, density=0.35)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        jacobian = lp._python_local_mean_phase_jacobian(theta, adjacency)
        # No entry exists where the adjacency is zero.
        assert np.all(jacobian[adjacency == 0.0] == 0.0)

    def test_local_complex_order_consistent_with_local_order(self) -> None:
        rng = np.random.default_rng(31)
        adjacency = _random_adjacency(8, 31)
        theta = rng.uniform(-math.pi, math.pi, size=8)
        radius = lo._python_local_order_parameter(theta, adjacency)
        phase = lp._python_local_mean_phase(theta, adjacency)
        degree = adjacency.sum(axis=1)
        reconstructed = degree * radius * np.exp(1j * phase)
        reference = adjacency @ np.exp(1j * theta)
        np.testing.assert_allclose(reconstructed, reference, atol=1e-12)

    def test_zero_degree_node(self) -> None:
        theta = np.array([0.3, 1.2, 2.1])
        adjacency = np.ones((3, 3), dtype=np.float64)
        adjacency[1, :] = 0.0
        np.testing.assert_array_equal(lp._python_local_mean_phase(theta, adjacency)[1], 0.0)
        np.testing.assert_array_equal(
            lp._python_local_mean_phase_jacobian(theta, adjacency)[1], np.zeros(3)
        )

    def test_rejects_non_square_adjacency(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            lp._python_local_mean_phase(np.zeros(3), np.ones((3, 2)))
        with pytest.raises(ValueError, match="square matrix of order 3"):
            lp._python_local_mean_phase_jacobian(np.zeros(3), np.ones((2, 2)))

    def test_empty(self) -> None:
        empty = np.zeros((0, 0))
        assert lp._python_local_mean_phase(np.array([]), empty).shape == (0,)
        assert lp._python_local_mean_phase_jacobian(np.array([]), empty).shape == (0, 0)


class TestLocalMeanPhaseTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "local_mean_phase", None)):
            pytest.skip("scpn_quantum_engine.local_mean_phase unavailable")
        rng = np.random.default_rng(seed)
        adjacency = _random_adjacency(n, seed) if n > 1 else np.zeros((n, n))
        theta = rng.uniform(-10 * math.pi, 10 * math.pi, size=n)
        np.testing.assert_allclose(
            lp._rust_local_mean_phase(theta, adjacency),
            lp._python_local_mean_phase(theta, adjacency),
            atol=1e-11,
        )
        np.testing.assert_allclose(
            lp._rust_local_mean_phase_jacobian(theta, adjacency),
            lp._python_local_mean_phase_jacobian(theta, adjacency),
            atol=1e-11,
        )

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import local_mean_phase as julia_phase
        from scpn_quantum_control.accel.julia import (
            local_mean_phase_jacobian as julia_jacobian,
        )

        adjacency = _random_adjacency(9, 2026)
        rng = np.random.default_rng(20260624)
        theta = rng.uniform(-math.pi, math.pi, size=9)
        np.testing.assert_allclose(
            julia_phase(theta, adjacency),
            lp._python_local_mean_phase(theta, adjacency),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            julia_jacobian(theta, adjacency),
            lp._python_local_mean_phase_jacobian(theta, adjacency),
            atol=1e-10,
        )

    def test_rust_phase_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            lp._rust_local_mean_phase(np.zeros(3), np.ones((3, 3)))

    def test_rust_jacobian_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            lp._rust_local_mean_phase_jacobian(np.zeros(3), np.ones((3, 3)))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        adjacency = _random_adjacency(4, 1)
        theta = np.full(4, 0.5)
        for chain, floor in (
            (lp._LOCAL_MEAN_PHASE_CHAIN, lp._python_local_mean_phase),
            (lp._LOCAL_MEAN_PHASE_JACOBIAN_CHAIN, lp._python_local_mean_phase_jacobian),
        ):
            disp = lp.MultiLangDispatcher([chain[0], chain[-1]])
            out = disp(theta, adjacency)
            np.testing.assert_allclose(out, floor(theta, adjacency))
            assert disp.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        adjacency = _random_adjacency(n, seed)
        theta = rng.uniform(-math.pi, math.pi, size=n)
        for chain, floor in (
            (lp._LOCAL_MEAN_PHASE_CHAIN, lp._python_local_mean_phase),
            (lp._LOCAL_MEAN_PHASE_JACOBIAN_CHAIN, lp._python_local_mean_phase_jacobian),
        ):
            reference = floor(theta, adjacency)
            for name, impl in chain:
                try:
                    out = impl(theta, adjacency)
                except (ImportError, ModuleNotFoundError, RuntimeError):
                    continue
                np.testing.assert_allclose(out, reference, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            last_local_mean_phase_jacobian_tier_used,
            last_local_mean_phase_tier_used,
            local_mean_phase,
            local_mean_phase_jacobian,
        )

        adjacency = _random_adjacency(10, 5)
        rng = np.random.default_rng(5)
        theta = rng.uniform(0.0, 2 * math.pi, size=10)
        assert d.dispatch("local_mean_phase", theta, adjacency).shape == (10,)
        assert d.dispatch("local_mean_phase_jacobian", theta, adjacency).shape == (10, 10)
        assert local_mean_phase(theta, adjacency).shape == (10,)
        assert local_mean_phase_jacobian(theta, adjacency).shape == (10, 10)
        assert last_local_mean_phase_tier_used() in {"rust", "julia", "python"}
        assert last_local_mean_phase_jacobian_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert lp._LOCAL_MEAN_PHASE_CHAIN[-1][0] == "python"
        assert lp._LOCAL_MEAN_PHASE_JACOBIAN_CHAIN[-1][0] == "python"
