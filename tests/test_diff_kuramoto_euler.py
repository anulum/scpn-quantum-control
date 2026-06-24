# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable networked-Kuramoto Euler integrator and adjoint tests
"""Multi-angle tests for the differentiable networked-Kuramoto Euler integrator and its adjoint.

Covers the forward integrator (manual-Euler match, shape, initial row, reduction to the engine's
final-state ``kuramoto_euler``), the reverse-mode adjoint (every gradient channel
``∂L/∂{θ₀, ω, K}`` against central finite differences, the zero-step identity, linearity in the
cotangent), input validation, cross-tier parity (Rust ↔ Julia ↔ Python floor), name-keyed
dispatch and the public API, partial-engine fall-through, and a gradient-descent control smoke
test (one step of optimal-coupling design decreases a desynchronisation objective).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scpn_quantum_control.accel.diff_kuramoto_euler as dk
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
    def test_matches_manual_euler(self) -> None:
        theta0, omega, coupling = _random_problem(5, 1)
        dt, n_steps = 0.05, 12
        trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        assert trajectory.shape == (n_steps + 1, 5)
        np.testing.assert_allclose(trajectory[0], theta0, atol=1e-15)
        current = theta0.copy()
        for step in range(n_steps):
            force = (coupling * np.sin(current[None, :] - current[:, None])).sum(axis=1)
            current = current + dt * (omega + force)
            np.testing.assert_allclose(trajectory[step + 1], current, atol=1e-13)

    def test_final_state_matches_engine_kuramoto_euler(self) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "kuramoto_euler", None)):
            pytest.skip("scpn_quantum_engine.kuramoto_euler unavailable")
        theta0, omega, coupling = _random_problem(6, 2)
        dt, n_steps = 0.05, 20
        trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        engine_final = np.asarray(
            engine.kuramoto_euler(theta0, omega, coupling, dt, n_steps), dtype=np.float64
        )
        np.testing.assert_allclose(trajectory[-1], engine_final, atol=1e-10)

    def test_zero_steps_is_initial_state(self) -> None:
        theta0, omega, coupling = _random_problem(4, 3)
        trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, 0.1, 0)
        assert trajectory.shape == (1, 4)
        np.testing.assert_allclose(trajectory[0], theta0, atol=1e-15)

    def test_rejects_inconsistent_shapes(self) -> None:
        with pytest.raises(ValueError, match="omega must have shape"):
            dk._python_kuramoto_euler_trajectory(
                np.zeros(3), np.zeros(2), np.zeros((3, 3)), 0.1, 1
            )
        with pytest.raises(ValueError, match="coupling must have shape"):
            dk._python_kuramoto_euler_trajectory(
                np.zeros(3), np.zeros(3), np.zeros((3, 2)), 0.1, 1
            )
        with pytest.raises(ValueError, match="n_steps must be non-negative"):
            dk._python_kuramoto_euler_trajectory(
                np.zeros(3), np.zeros(3), np.zeros((3, 3)), 0.1, -1
            )


class TestAdjointGradients:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=8),
        n_steps=st.integers(min_value=1, max_value=12),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_channels_match_finite_difference(self, n: int, n_steps: int, seed: int) -> None:
        theta0, omega, coupling = _random_problem(n, seed)
        dt = 0.05
        trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        cotangent = _cotangent(trajectory[-1])
        grad_theta0, grad_omega, grad_coupling = dk._python_kuramoto_euler_vjp(
            trajectory, coupling, dt, cotangent
        )
        step = 1e-6

        def terminal(t0: np.ndarray, om: np.ndarray, k: np.ndarray) -> float:
            return _objective(dk._python_kuramoto_euler_trajectory(t0, om, k, dt, n_steps)[-1])

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
        theta0, _, coupling = _random_problem(4, 7)
        trajectory = theta0[None, :]
        cotangent = np.array([1.0, -2.0, 0.5, 3.0])
        grad_theta0, grad_omega, grad_coupling = dk._python_kuramoto_euler_vjp(
            trajectory, coupling, 0.1, cotangent
        )
        np.testing.assert_allclose(grad_theta0, cotangent, atol=1e-15)
        np.testing.assert_allclose(grad_omega, np.zeros(4), atol=1e-15)
        np.testing.assert_array_equal(grad_coupling, np.zeros((4, 4)))

    def test_linear_in_cotangent(self) -> None:
        theta0, omega, coupling = _random_problem(5, 9)
        dt, n_steps = 0.05, 10
        trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        rng = np.random.default_rng(11)
        u = rng.standard_normal(5)
        v = rng.standard_normal(5)
        gu = dk._python_kuramoto_euler_vjp(trajectory, coupling, dt, u)
        gv = dk._python_kuramoto_euler_vjp(trajectory, coupling, dt, v)
        guv = dk._python_kuramoto_euler_vjp(trajectory, coupling, dt, 2.0 * u - 3.0 * v)
        for combined, a, b in zip(guv, gu, gv, strict=True):
            np.testing.assert_allclose(combined, 2.0 * a - 3.0 * b, atol=1e-12)

    def test_rejects_inconsistent_shapes(self) -> None:
        with pytest.raises(ValueError, match="trajectory must be two-dimensional"):
            dk._python_kuramoto_euler_vjp(np.zeros(3), np.zeros((3, 3)), 0.1, np.zeros(3))
        with pytest.raises(ValueError, match="coupling must have shape"):
            dk._python_kuramoto_euler_vjp(np.zeros((2, 3)), np.zeros((3, 2)), 0.1, np.zeros(3))
        with pytest.raises(ValueError, match="cotangent must have shape"):
            dk._python_kuramoto_euler_vjp(np.zeros((2, 3)), np.zeros((3, 3)), 0.1, np.zeros(2))

    def test_gradient_descent_step_reduces_desync_objective(self) -> None:
        # Control smoke test: one gradient step on K reduces L = 1 − r(θ_T)² (desynchronisation),
        # demonstrating optimal-coupling design through the differentiable integrator.
        theta0, omega, coupling = _random_problem(8, 13)
        dt, n_steps = 0.05, 40

        def loss_and_grad_k(k: np.ndarray) -> tuple[float, np.ndarray]:
            trajectory = dk._python_kuramoto_euler_trajectory(theta0, omega, k, dt, n_steps)
            final = trajectory[-1]
            order = np.mean(np.exp(1j * final))
            radius = float(abs(order))
            loss = 1.0 - radius**2
            # L = 1 − r², so ∂L/∂θ_{T,j} = −∂(r²)/∂θ_j = (2/N)(C sin θ_j − S cos θ_j), z = C + iS.
            c, s = order.real, order.imag
            cotangent = (2.0 / final.size) * (c * np.sin(final) - s * np.cos(final))
            _, _, grad_k = dk._python_kuramoto_euler_vjp(trajectory, k, dt, cotangent)
            return loss, grad_k

        loss0, grad_k = loss_and_grad_k(coupling)
        assert np.linalg.norm(grad_k) > 1e-6  # non-trivial descent direction
        # A small step along the (verified) negative gradient strictly decreases the objective,
        # by the first-order Taylor expansion L(K − ε∇L) = L(K) − ε‖∇L‖² + O(ε²).
        loss1, _ = loss_and_grad_k(coupling - 1e-3 * grad_k)
        assert loss1 < loss0


class TestTiersAndDispatch:
    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=20),
        n_steps=st.integers(min_value=0, max_value=25),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_rust_matches_python_floor(self, n: int, n_steps: int, seed: int) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        if not callable(getattr(engine, "kuramoto_euler_trajectory", None)):
            pytest.skip("scpn_quantum_engine.kuramoto_euler_trajectory unavailable")
        theta0, omega, coupling = _random_problem(n, seed)
        dt = 0.05
        rust_traj = dk._rust_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        py_traj = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        np.testing.assert_allclose(rust_traj, py_traj, atol=1e-11)
        cotangent = _cotangent(py_traj[-1])
        rust_grad = dk._rust_kuramoto_euler_vjp(rust_traj, coupling, dt, cotangent)
        py_grad = dk._python_kuramoto_euler_vjp(py_traj, coupling, dt, cotangent)
        for rust_channel, py_channel in zip(rust_grad, py_grad, strict=True):
            np.testing.assert_allclose(rust_channel, py_channel, atol=1e-10)

    def test_julia_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        from scpn_quantum_control.accel.julia import (
            kuramoto_euler_trajectory as julia_traj,
        )
        from scpn_quantum_control.accel.julia import kuramoto_euler_vjp as julia_vjp

        theta0, omega, coupling = _random_problem(7, 2026)
        dt, n_steps = 0.05, 18
        jt = julia_traj(theta0, omega, coupling, dt, n_steps)
        pt = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        np.testing.assert_allclose(jt, pt, atol=1e-10)
        cotangent = _cotangent(pt[-1])
        jg = julia_vjp(jt, coupling, dt, cotangent)
        pg = dk._python_kuramoto_euler_vjp(pt, coupling, dt, cotangent)
        for julia_channel, py_channel in zip(jg, pg, strict=True):
            np.testing.assert_allclose(julia_channel, py_channel, atol=1e-10)

    def test_rust_forward_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dk._rust_kuramoto_euler_trajectory(np.zeros(3), np.zeros(3), np.zeros((3, 3)), 0.1, 1)

    def test_rust_vjp_absence_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(d, "optional_rust_engine", lambda: None)
        with pytest.raises(ModuleNotFoundError, match="scpn_quantum_engine"):
            dk._rust_kuramoto_euler_vjp(np.zeros((2, 3)), np.zeros((3, 3)), 0.1, np.zeros(3))

    def test_partial_engine_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PartialEngine:
            pass

        monkeypatch.setattr(d, "optional_rust_engine", lambda: PartialEngine())
        theta0, omega, coupling = _random_problem(4, 5)
        forward = dk.MultiLangDispatcher(
            [dk._KURAMOTO_EULER_TRAJECTORY_CHAIN[0], dk._KURAMOTO_EULER_TRAJECTORY_CHAIN[-1]]
        )
        trajectory = forward(theta0, omega, coupling, 0.05, 6)
        np.testing.assert_allclose(
            trajectory, dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, 0.05, 6)
        )
        assert forward.last_tier == "python"
        reverse = dk.MultiLangDispatcher(
            [dk._KURAMOTO_EULER_VJP_CHAIN[0], dk._KURAMOTO_EULER_VJP_CHAIN[-1]]
        )
        cotangent = _cotangent(trajectory[-1])
        grads = reverse(trajectory, coupling, 0.05, cotangent)
        reference = dk._python_kuramoto_euler_vjp(trajectory, coupling, 0.05, cotangent)
        for channel, expected in zip(grads, reference, strict=True):
            np.testing.assert_allclose(channel, expected)
        assert reverse.last_tier == "python"

    def test_registry_and_public_api(self) -> None:
        from scpn_quantum_control.accel import (
            kuramoto_euler_trajectory,
            kuramoto_euler_vjp,
            last_kuramoto_euler_trajectory_tier_used,
            last_kuramoto_euler_vjp_tier_used,
        )

        theta0, omega, coupling = _random_problem(6, 17)
        dt, n_steps = 0.05, 10
        trajectory = kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        assert trajectory.shape == (n_steps + 1, 6)
        assert d.dispatch(
            "kuramoto_euler_trajectory", theta0, omega, coupling, dt, n_steps
        ).shape == (
            n_steps + 1,
            6,
        )
        cotangent = _cotangent(trajectory[-1])
        grads = kuramoto_euler_vjp(trajectory, coupling, dt, cotangent)
        assert grads[0].shape == (6,)
        assert grads[1].shape == (6,)
        assert grads[2].shape == (6, 6)
        assert d.dispatch("kuramoto_euler_vjp", trajectory, coupling, dt, cotangent)[2].shape == (
            6,
            6,
        )
        assert last_kuramoto_euler_trajectory_tier_used() in {"rust", "julia", "python"}
        assert last_kuramoto_euler_vjp_tier_used() in {"rust", "julia", "python"}

    @_GLOBAL_SETTINGS
    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_all_available_tiers_agree(self, n: int, seed: int) -> None:
        theta0, omega, coupling = _random_problem(n, seed)
        dt, n_steps = 0.05, 8
        reference_traj = dk._python_kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        for name, impl in dk._KURAMOTO_EULER_TRAJECTORY_CHAIN:
            try:
                out = impl(theta0, omega, coupling, dt, n_steps)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            np.testing.assert_allclose(out, reference_traj, atol=1e-10, err_msg=name)
        cotangent = _cotangent(reference_traj[-1])
        reference_grad = dk._python_kuramoto_euler_vjp(reference_traj, coupling, dt, cotangent)
        for name, impl in dk._KURAMOTO_EULER_VJP_CHAIN:
            try:
                grads = impl(reference_traj, coupling, dt, cotangent)
            except (ImportError, ModuleNotFoundError, RuntimeError):
                continue
            for channel, expected in zip(grads, reference_grad, strict=True):
                np.testing.assert_allclose(channel, expected, atol=1e-10, err_msg=name)

    def test_chains_end_with_python_floor(self) -> None:
        assert dk._KURAMOTO_EULER_TRAJECTORY_CHAIN[-1][0] == "python"
        assert dk._KURAMOTO_EULER_VJP_CHAIN[-1][0] == "python"
