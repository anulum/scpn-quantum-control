# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the adaptive Dormand–Prince Kuramoto integrator
r"""Tests for :mod:`oscillatools.accel.diff_kuramoto_dopri`.

The error-controlled forward is checked against a high-accuracy ``solve_ivp`` reference and for
the roadmap acceptance that tightening the tolerance shrinks the terminal error, and the zero
coupling case is checked to reproduce the exact linear drift. The fixed-grid discrete adjoint is
checked for gradient parity — the gradients with respect to the initial phases, the natural
frequencies and the coupling matrix each match a central finite difference. The trajectory
structure and the validation branches are covered.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from oscillatools.accel.diff_kuramoto_dopri import (
    DopriTrajectory,
    kuramoto_dopri_trajectory,
    kuramoto_dopri_vjp,
)


def _problem(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 1.0, count)
    coupling = rng.normal(0.0, 0.3, (count, count))
    coupling = coupling + coupling.T
    np.fill_diagonal(coupling, 0.0)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    return theta0, omega, coupling


def _force(theta: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    difference = theta[None, :] - theta[:, None]
    return (coupling * np.sin(difference)).sum(axis=1)


# --------------------------------------------------------------------------- forward accuracy


def test_forward_matches_high_accuracy_reference() -> None:
    theta0, omega, coupling = _problem(5, 0)
    trajectory = kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=2.0, rtol=1e-9, atol=1e-12
    )
    reference = solve_ivp(
        lambda _t, y: omega + _force(y, coupling),
        [0.0, 2.0],
        theta0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    ).y[:, -1]
    assert np.allclose(trajectory.terminal_phases, reference, atol=1e-6)


def test_tolerance_is_honoured_and_tightens() -> None:
    theta0, omega, coupling = _problem(5, 1)
    reference = solve_ivp(
        lambda _t, y: omega + _force(y, coupling),
        [0.0, 2.0],
        theta0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    ).y[:, -1]
    previous = np.inf
    for rtol in (1e-4, 1e-6, 1e-8):
        trajectory = kuramoto_dopri_trajectory(
            theta0, omega, coupling, t_end=2.0, rtol=rtol, atol=rtol * 1e-3
        )
        error = float(np.max(np.abs(trajectory.terminal_phases - reference)))
        assert error <= previous + 1e-12  # error shrinks as the tolerance tightens
        previous = error


def test_zero_coupling_reproduces_linear_drift() -> None:
    theta0, omega, _ = _problem(4, 2)
    coupling = np.zeros((4, 4))
    trajectory = kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=1.5, rtol=1e-9, atol=1e-12
    )
    assert np.allclose(trajectory.terminal_phases, theta0 + omega * 1.5, atol=1e-7)


# --------------------------------------------------------------------------- trajectory structure


def test_trajectory_structure_is_consistent() -> None:
    theta0, omega, coupling = _problem(4, 3)
    trajectory = kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=1.0, rtol=1e-6)
    assert isinstance(trajectory, DopriTrajectory)
    steps = trajectory.steps.size
    assert trajectory.times.shape == (steps + 1,)
    assert trajectory.phases.shape == (steps + 1, 4)
    assert trajectory.times[0] == 0.0
    assert trajectory.times[-1] == pytest.approx(1.0)
    assert np.all(trajectory.steps > 0.0)
    assert trajectory.steps.sum() == pytest.approx(1.0)
    assert np.all(np.diff(trajectory.times) > 0.0)
    assert np.array_equal(trajectory.terminal_phases, trajectory.phases[-1])


def test_explicit_first_step_is_accepted() -> None:
    theta0, omega, coupling = _problem(4, 4)
    trajectory = kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=1.0, rtol=1e-7, first_step=0.05
    )
    assert trajectory.times[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- gradient parity


def _terminal(theta0: np.ndarray, omega: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    return kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=1.0, rtol=1e-9, atol=1e-12
    ).terminal_phases


def _finite_difference(function, base: np.ndarray, step: float = 1e-6) -> np.ndarray:
    numeric = np.zeros_like(base)
    flat_numeric = numeric.reshape(-1)
    flat_base = base.reshape(-1)
    for index in range(flat_base.size):
        forward = flat_base.copy()
        forward[index] += step
        backward = flat_base.copy()
        backward[index] -= step
        loss_forward = 0.5 * np.sum(function(forward.reshape(base.shape)) ** 2)
        loss_backward = 0.5 * np.sum(function(backward.reshape(base.shape)) ** 2)
        flat_numeric[index] = (loss_forward - loss_backward) / (2.0 * step)
    return numeric


def test_vjp_matches_finite_difference_across_all_inputs() -> None:
    theta0, omega, coupling = _problem(4, 5)
    trajectory = kuramoto_dopri_trajectory(
        theta0, omega, coupling, t_end=1.0, rtol=1e-9, atol=1e-12
    )
    seed = trajectory.terminal_phases.copy()  # dL/dtheta_T for L = 0.5 sum theta_T^2
    grad_theta0, grad_omega, grad_coupling = kuramoto_dopri_vjp(
        trajectory.phases, trajectory.steps, omega, coupling, seed
    )

    numeric_theta0 = _finite_difference(lambda value: _terminal(value, omega, coupling), theta0)
    numeric_omega = _finite_difference(lambda value: _terminal(theta0, value, coupling), omega)
    numeric_coupling = _finite_difference(lambda value: _terminal(theta0, omega, value), coupling)

    assert np.allclose(grad_theta0, numeric_theta0, atol=1e-5)
    assert np.allclose(grad_omega, numeric_omega, atol=1e-5)
    assert np.allclose(grad_coupling, numeric_coupling, atol=1e-5)


def test_vjp_shapes() -> None:
    theta0, omega, coupling = _problem(3, 6)
    trajectory = kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=0.5, rtol=1e-7)
    grad_theta0, grad_omega, grad_coupling = kuramoto_dopri_vjp(
        trajectory.phases, trajectory.steps, omega, coupling, np.ones(3)
    )
    assert grad_theta0.shape == (3,)
    assert grad_omega.shape == (3,)
    assert grad_coupling.shape == (3, 3)


# --------------------------------------------------------------------------- forward validation


def test_forward_rejects_non_positive_t_end() -> None:
    theta0, omega, coupling = _problem(3, 7)
    with pytest.raises(ValueError, match="t_end must be strictly positive"):
        kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=0.0)


def test_forward_rejects_non_positive_tolerances() -> None:
    theta0, omega, coupling = _problem(3, 8)
    with pytest.raises(ValueError, match="rtol and atol must be strictly positive"):
        kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=1.0, rtol=0.0)


def test_forward_rejects_non_vector_theta0() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        kuramoto_dopri_trajectory(np.zeros((2, 2)), np.zeros(4), np.zeros((4, 4)), t_end=1.0)


def test_forward_rejects_mismatched_omega() -> None:
    with pytest.raises(ValueError, match="omega must match"):
        kuramoto_dopri_trajectory(np.zeros(3), np.zeros(4), np.zeros((3, 3)), t_end=1.0)


def test_forward_rejects_non_square_coupling() -> None:
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        kuramoto_dopri_trajectory(np.zeros(3), np.zeros(3), np.zeros((3, 4)), t_end=1.0)


def test_forward_raises_when_max_steps_exceeded() -> None:
    theta0, omega, coupling = _problem(3, 9)
    with pytest.raises(ValueError, match="exceeded max_steps"):
        kuramoto_dopri_trajectory(theta0, omega, coupling, t_end=10.0, rtol=1e-10, max_steps=2)


# --------------------------------------------------------------------------- adjoint validation


def test_vjp_rejects_too_short_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        kuramoto_dopri_vjp(
            np.zeros((1, 3)), np.zeros(0), np.zeros(3), np.zeros((3, 3)), np.zeros(3)
        )


def test_vjp_rejects_mismatched_steps() -> None:
    with pytest.raises(ValueError, match="steps must be one-dimensional of length"):
        kuramoto_dopri_vjp(
            np.zeros((4, 3)), np.zeros(2), np.zeros(3), np.zeros((3, 3)), np.zeros(3)
        )


def test_vjp_rejects_mismatched_cotangent() -> None:
    with pytest.raises(ValueError, match="cotangent must have shape"):
        kuramoto_dopri_vjp(
            np.zeros((3, 3)), np.zeros(2), np.zeros(3), np.zeros((3, 3)), np.zeros(4)
        )


def test_vjp_rejects_mismatched_omega() -> None:
    with pytest.raises(ValueError, match="omega must have shape"):
        kuramoto_dopri_vjp(
            np.zeros((3, 3)), np.zeros(2), np.zeros(4), np.zeros((3, 3)), np.zeros(3)
        )


def test_vjp_rejects_non_square_coupling() -> None:
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        kuramoto_dopri_vjp(
            np.zeros((3, 3)), np.zeros(2), np.zeros(3), np.zeros((3, 4)), np.zeros(3)
        )
