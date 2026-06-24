# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable networked-Kuramoto RK4 integrator and adjoint tests
"""Multi-angle tests for the differentiable networked-Kuramoto RK4 integrator and its adjoint.

Covers the forward integrator (manual-RK4-step match, shape, fourth-order convergence, lower
error than Euler over a shared budget), the reverse-mode adjoint (every gradient channel
``∂L/∂{θ₀, ω, K}`` against central finite differences, the zero-step identity, linearity in the
cotangent), input validation, cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed
dispatch and the public API, partial-engine fall-through, and a gradient-descent control smoke
test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.diff_kuramoto_euler as dk_euler
import scpn_quantum_control.accel.diff_kuramoto_rk4 as rk
import scpn_quantum_control.accel.dispatcher as d

_GLOBAL_SETTINGS = settings(max_examples=30, deadline=None)


def _random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=n)
    omega = rng.uniform(-1.0, 1.0, size=n)
    coupling = rng.uniform(0.0, 0.4, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


def _objective(theta_final: np.ndarray) -> float:
    return float(np.sin(theta_final).sum())


def _cotangent(theta_final: np.ndarray) -> np.ndarray:
    return np.cos(theta_final)


class TestForwardTrajectory:
    def test_matches_manual_rk4_step(self) -> None:
        theta0, omega, coupling = _random_problem(5, 1)
        dt, n_steps = 0.05, 6
        trajectory = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        assert trajectory.shape == (n_steps + 1, 5)
        np.testing.assert_allclose(trajectory[0], theta0, atol=1e-15)

        def force(phases: np.ndarray) -> np.ndarray:
            return (coupling * np.sin(phases[None, :] - phases[:, None])).sum(axis=1)

        current = theta0.copy()
        for step in range(n_steps):
            k1 = omega + force(current)
            k2 = omega + force(current + 0.5 * dt * k1)
            k3 = omega + force(current + 0.5 * dt * k2)
            k4 = omega + force(current + dt * k3)
            current = current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            np.testing.assert_allclose(trajectory[step + 1], current, atol=1e-13)

    def test_fourth_order_convergence(self) -> None:
        theta0, omega, coupling = _random_problem(4, 2)

        def final(steps: int) -> np.ndarray:
            return rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, 1.0 / steps, steps)[
                -1
            ]

        err1 = float(np.max(np.abs(final(40) - final(80))))
        err2 = float(np.max(np.abs(final(80) - final(160))))
        # Halving dt cuts a fourth-order error by ~16×.
        assert err1 / err2 > 12.0

    def test_more_accurate_than_euler(self) -> None:
        theta0, omega, coupling = _random_problem(4, 2)
        total, steps = 2.0, 20
        dt = total / steps
        reference = rk._python_kuramoto_rk4_trajectory(
            theta0, omega, coupling, total / 4000.0, 4000
        )[-1]
        rk4_final = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, steps)[-1]
        euler_final = dk_euler._python_kuramoto_euler_trajectory(
            theta0, omega, coupling, dt, steps
        )[-1]
        assert np.max(np.abs(rk4_final - reference)) < np.max(np.abs(euler_final - reference))

    def test_zero_steps_is_initial_state(self) -> None:
        theta0, omega, coupling = _random_problem(4, 3)
        trajectory = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, 0.1, 0)
        assert trajectory.shape == (1, 4)
        np.testing.assert_allclose(trajectory[0], theta0, atol=1e-15)

    def test_rejects_inconsistent_shapes(self) -> None:
        with pytest.raises(ValueError, match="omega must have shape"):
            rk._python_kuramoto_rk4_trajectory(np.zeros(3), np.zeros(2), np.zeros((3, 3)), 0.1, 1)
        with pytest.raises(ValueError, match="coupling must have shape"):
            rk._python_kuramoto_rk4_trajectory(np.zeros(3), np.zeros(3), np.zeros((3, 2)), 0.1, 1)
        with pytest.raises(ValueError, match="n_steps must be non-negative"):
            rk._python_kuramoto_rk4_trajectory(np.zeros(3), np.zeros(3), np.zeros((3, 3)), 0.1, -1)


class TestAdjointGradients:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=8),
        n_steps=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_channels_match_finite_difference(self, n: int, n_steps: int, seed: int) -> None:
        theta0, omega, coupling = _random_problem(n, seed)
        dt = 0.05
        trajectory = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        cotangent = _cotangent(trajectory[-1])
        grad_theta0, grad_omega, grad_coupling = rk._python_kuramoto_rk4_vjp(
            trajectory, omega, coupling, dt, cotangent
        )
        step = 1e-6

        def terminal(t0: np.ndarray, om: np.ndarray, k: np.ndarray) -> float:
            return _objective(rk._python_kuramoto_rk4_trajectory(t0, om, k, dt, n_steps)[-1])

        for i in range(n):
            plus = theta0.copy()
            minus = theta0.copy()
            plus[i] += step
            minus[i] -= step
            fd = (terminal(plus, omega, coupling) - terminal(minus, omega, coupling)) / (
                2.0 * step
            )
            assert abs(grad_theta0[i] - fd) < 1e-5
        for i in range(n):
            plus = omega.copy()
            minus = omega.copy()
            plus[i] += step
            minus[i] -= step
            fd = (terminal(theta0, plus, coupling) - terminal(theta0, minus, coupling)) / (
                2.0 * step
            )
            assert abs(grad_omega[i] - fd) < 1e-5
        for p in range(n):
            for q in range(n):
                plus = coupling.copy()
                minus = coupling.copy()
                plus[p, q] += step
                minus[p, q] -= step
                fd = (terminal(theta0, omega, plus) - terminal(theta0, omega, minus)) / (
                    2.0 * step
                )
                assert abs(grad_coupling[p, q] - fd) < 1e-5

    def test_zero_steps_passes_cotangent_through(self) -> None:
        theta0, omega, coupling = _random_problem(4, 7)
        trajectory = theta0[None, :]
        cotangent = np.array([1.0, -2.0, 0.5, 3.0])
        grad_theta0, grad_omega, grad_coupling = rk._python_kuramoto_rk4_vjp(
            trajectory, omega, coupling, 0.1, cotangent
        )
        np.testing.assert_allclose(grad_theta0, cotangent, atol=1e-15)
        np.testing.assert_allclose(grad_omega, np.zeros(4), atol=1e-15)
        np.testing.assert_array_equal(grad_coupling, np.zeros((4, 4)))

    def test_linear_in_cotangent(self) -> None:
        theta0, omega, coupling = _random_problem(5, 9)
        dt, n_steps = 0.05, 8
        trajectory = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        rng = np.random.default_rng(11)
        u = rng.standard_normal(5)
        v = rng.standard_normal(5)
        gu = rk._python_kuramoto_rk4_vjp(trajectory, omega, coupling, dt, u)
        gv = rk._python_kuramoto_rk4_vjp(trajectory, omega, coupling, dt, v)
        guv = rk._python_kuramoto_rk4_vjp(trajectory, omega, coupling, dt, 2.0 * u - 3.0 * v)
        for combined, a, b in zip(guv, gu, gv, strict=True):
            np.testing.assert_allclose(combined, 2.0 * a - 3.0 * b, atol=1e-12)

    def test_rejects_inconsistent_shapes(self) -> None:
        with pytest.raises(ValueError, match="trajectory must be two-dimensional"):
            rk._python_kuramoto_rk4_vjp(
                np.zeros(3), np.zeros(3), np.zeros((3, 3)), 0.1, np.zeros(3)
            )
        with pytest.raises(ValueError, match="omega must have shape"):
            rk._python_kuramoto_rk4_vjp(
                np.zeros((2, 3)), np.zeros(2), np.zeros((3, 3)), 0.1, np.zeros(3)
            )
        with pytest.raises(ValueError, match="coupling must have shape"):
            rk._python_kuramoto_rk4_vjp(
                np.zeros((2, 3)), np.zeros(3), np.zeros((3, 2)), 0.1, np.zeros(3)
            )
        with pytest.raises(ValueError, match="cotangent must have shape"):
            rk._python_kuramoto_rk4_vjp(
                np.zeros((2, 3)), np.zeros(3), np.zeros((3, 3)), 0.1, np.zeros(2)
            )

    def test_gradient_descent_step_reduces_desync_objective(self) -> None:
        theta0, omega, coupling = _random_problem(8, 13)
        dt, n_steps = 0.05, 20

        def loss_and_grad_k(k: np.ndarray) -> tuple[float, np.ndarray]:
            trajectory = rk._python_kuramoto_rk4_trajectory(theta0, omega, k, dt, n_steps)
            final = trajectory[-1]
            order = np.mean(np.exp(1j * final))
            loss = 1.0 - float(abs(order)) ** 2
            c, s = order.real, order.imag
            cotangent = (2.0 / final.size) * (c * np.sin(final) - s * np.cos(final))
            _, _, grad_k = rk._python_kuramoto_rk4_vjp(trajectory, omega, k, dt, cotangent)
            return loss, grad_k

        loss0, grad_k = loss_and_grad_k(coupling)
        assert np.linalg.norm(grad_k) > 1e-6
        loss1, _ = loss_and_grad_k(coupling - 1e-3 * grad_k)
        assert loss1 < loss0


class TestTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=18),
        n_steps=st.integers(min_value=0, max_value=20),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, n_steps: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "kuramoto_rk4_trajectory", None)):
            pytest.skip("scpn_quantum_engine.kuramoto_rk4_trajectory unavailable")
        theta0, omega, coupling = _random_problem(n, seed)
        dt = 0.05
        rust_traj = rk._rust_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        py_traj = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        np.testing.assert_allclose(rust_traj, py_traj, atol=1e-11)
        cotangent = _cotangent(py_traj[-1])
        rust_grad = rk._rust_kuramoto_rk4_vjp(rust_traj, omega, coupling, dt, cotangent)
        py_grad = rk._python_kuramoto_rk4_vjp(py_traj, omega, coupling, dt, cotangent)
        for rust_channel, py_channel in zip(rust_grad, py_grad, strict=True):
            np.testing.assert_allclose(rust_channel, py_channel, atol=1e-10)

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import kuramoto_rk4_trajectory as julia_traj
        from scpn_quantum_control.accel.julia import kuramoto_rk4_vjp as julia_vjp

        theta0, omega, coupling = _random_problem(7, 2026)
        dt, n_steps = 0.05, 12
        jt = julia_traj(theta0, omega, coupling, dt, n_steps)
        pt = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        np.testing.assert_allclose(jt, pt, atol=1e-10)
        cotangent = _cotangent(pt[-1])
        jg = julia_vjp(jt, omega, coupling, dt, cotangent)
        pg = rk._python_kuramoto_rk4_vjp(pt, omega, coupling, dt, cotangent)
        for julia_channel, py_channel in zip(jg, pg, strict=True):
            np.testing.assert_allclose(julia_channel, py_channel, atol=1e-10)

    def test_rust_forward_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            rk._rust_kuramoto_rk4_trajectory(np.zeros(3), np.zeros(3), np.zeros((3, 3)), 0.1, 1)

    def test_rust_vjp_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            rk._rust_kuramoto_rk4_vjp(
                np.zeros((2, 3)), np.zeros(3), np.zeros((3, 3)), 0.1, np.zeros(3)
            )

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta0, omega, coupling = _random_problem(4, 5)
        forward = rk.MultiLangDispatcher(
            [rk._KURAMOTO_RK4_TRAJECTORY_CHAIN[0], rk._KURAMOTO_RK4_TRAJECTORY_CHAIN[-1]]
        )
        trajectory = forward(theta0, omega, coupling, 0.05, 6)
        np.testing.assert_allclose(
            trajectory, rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, 0.05, 6)
        )
        assert forward.last_tier == "python"
        reverse = rk.MultiLangDispatcher(
            [rk._KURAMOTO_RK4_VJP_CHAIN[0], rk._KURAMOTO_RK4_VJP_CHAIN[-1]]
        )
        cotangent = _cotangent(trajectory[-1])
        grads = reverse(trajectory, omega, coupling, 0.05, cotangent)
        reference = rk._python_kuramoto_rk4_vjp(trajectory, omega, coupling, 0.05, cotangent)
        for channel, expected in zip(grads, reference, strict=True):
            np.testing.assert_allclose(channel, expected)
        assert reverse.last_tier == "python"

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        theta0, omega, coupling = _random_problem(n, seed)
        dt, n_steps = 0.05, 6
        reference_traj = rk._python_kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        for name, impl in rk._KURAMOTO_RK4_TRAJECTORY_CHAIN:
            try:
                out = impl(theta0, omega, coupling, dt, n_steps)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference_traj, atol=1e-10, err_msg=name)
        cotangent = _cotangent(reference_traj[-1])
        reference_grad = rk._python_kuramoto_rk4_vjp(
            reference_traj, omega, coupling, dt, cotangent
        )
        for name, impl in rk._KURAMOTO_RK4_VJP_CHAIN:
            try:
                grads = impl(reference_traj, omega, coupling, dt, cotangent)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            for channel, expected in zip(grads, reference_grad, strict=True):
                np.testing.assert_allclose(channel, expected, atol=1e-10, err_msg=name)

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            kuramoto_rk4_trajectory,
            kuramoto_rk4_vjp,
            last_kuramoto_rk4_trajectory_tier_used,
            last_kuramoto_rk4_vjp_tier_used,
        )

        theta0, omega, coupling = _random_problem(6, 17)
        dt, n_steps = 0.05, 8
        trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        assert trajectory.shape == (n_steps + 1, 6)
        assert d.dispatch(
            "kuramoto_rk4_trajectory", theta0, omega, coupling, dt, n_steps
        ).shape == (n_steps + 1, 6)
        cotangent = _cotangent(trajectory[-1])
        grads = kuramoto_rk4_vjp(trajectory, omega, coupling, dt, cotangent)
        assert grads[0].shape == (6,)
        assert grads[2].shape == (6, 6)
        assert d.dispatch("kuramoto_rk4_vjp", trajectory, omega, coupling, dt, cotangent)[
            2
        ].shape == (6, 6)
        assert last_kuramoto_rk4_trajectory_tier_used() in {"rust", "julia", "python"}
        assert last_kuramoto_rk4_vjp_tier_used() in {"rust", "julia", "python"}

    def test_chains_end_with_python_floor(self) -> None:
        assert rk._KURAMOTO_RK4_TRAJECTORY_CHAIN[-1][0] == "python"
        assert rk._KURAMOTO_RK4_VJP_CHAIN[-1][0] == "python"
