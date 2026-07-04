# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the time-delayed Kuramoto simulation
r"""Tests for :mod:`oscillatools.accel.kuramoto_delayed`.

The delayed forces are checked against hand-evaluated sums and against the instantaneous
Kuramoto forces in the no-delay configuration; the delay-aware method-of-steps integrator is
checked against the exact drift solution (zero coupling, where RK4 is exact), against the
Yeung–Strogatz collective-frequency self-consistency for identical oscillators, and for the
delay-induced multistability that is the roadmap acceptance criterion — a run started from
different constant initial-history frequencies settles onto different stable branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_delayed import (
    DelayedTrajectory,
    delayed_mean_field_force,
    delayed_networked_force,
    integrate_delayed_kuramoto,
)
from oscillatools.accel.kuramoto_delayed_mean_field import (
    stable_synchronised_frequencies,
    synchronised_frequency_residual,
)


def _uniform_history(delay_steps: int, count: int, drift: float, dt: float) -> np.ndarray:
    """A history on ``[-τ, 0]`` winding at constant frequency ``drift`` with a tiny scatter."""
    base = np.linspace(0.0, 0.1, count)
    rows = [base + drift * (-(delay_steps - s) * dt) for s in range(delay_steps + 1)]
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------- forces


def test_delayed_mean_field_force_matches_hand_sum() -> None:
    current = np.array([0.0, 1.0, 2.0])
    delayed = np.array([0.5, 0.5, 0.5])
    coupling = 1.7
    expected = np.array([coupling * np.mean(np.sin(delayed - theta_j)) for theta_j in current])
    result = delayed_mean_field_force(current, delayed, coupling)
    assert np.allclose(result, expected)


def test_delayed_mean_field_force_reads_lagged_not_current() -> None:
    # With current == delayed the all-to-all force is the instantaneous mean-field force; swapping
    # the delayed vector changes the result, proving the force reads the lagged phases.
    current = np.array([0.0, 0.7, 1.3])
    same = delayed_mean_field_force(current, current, 2.0)
    lagged = delayed_mean_field_force(current, current + 0.4, 2.0)
    assert not np.allclose(same, lagged)


def test_delayed_networked_force_matches_loop() -> None:
    current = np.array([0.1, 0.6, 1.1, 1.9])
    delayed = np.array([0.2, 0.9, 1.4, 2.3])
    rng = np.random.default_rng(3)
    coupling = rng.normal(size=(4, 4))
    expected = np.array(
        [sum(coupling[j, k] * np.sin(delayed[k] - current[j]) for k in range(4)) for j in range(4)]
    )
    result = delayed_networked_force(current, delayed, coupling)
    assert np.allclose(result, expected)


def test_networked_force_reduces_to_mean_field_for_uniform_coupling() -> None:
    current = np.array([0.0, 0.5, 1.5, 2.0])
    delayed = np.array([0.3, 0.8, 1.1, 2.4])
    strength = 1.3
    uniform = np.full((4, 4), strength / 4)
    networked = delayed_networked_force(current, delayed, uniform)
    mean_field = delayed_mean_field_force(current, delayed, strength)
    assert np.allclose(networked, mean_field)


def test_mean_field_force_rejects_non_vector_current() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        delayed_mean_field_force(np.zeros((2, 2)), np.zeros(4), 1.0)


def test_mean_field_force_rejects_empty_current() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        delayed_mean_field_force(np.empty(0), np.empty(0), 1.0)


def test_mean_field_force_rejects_mismatched_delayed() -> None:
    with pytest.raises(ValueError, match="delayed_phases must have shape"):
        delayed_mean_field_force(np.zeros(3), np.zeros(4), 1.0)


def test_networked_force_rejects_wrong_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling_matrix must have shape"):
        delayed_networked_force(np.zeros(3), np.zeros(3), np.zeros((3, 4)))


# --------------------------------------------------------------------------- trajectory


def test_trajectory_terminal_phases_is_last_row() -> None:
    phases = np.arange(12.0).reshape(4, 3)
    traj = DelayedTrajectory(
        times=np.arange(4.0),
        phases=phases,
        order_parameter_series=np.ones(4),
        delay=1.0,
        delay_steps=10,
    )
    assert np.array_equal(traj.terminal_phases, phases[-1])


def test_collective_frequency_recovers_linear_slope() -> None:
    times = np.linspace(0.0, 1.0, 50)
    slope = 2.3
    phases = (slope * times)[:, None] + np.array([0.0, 0.5, 1.0])
    traj = DelayedTrajectory(times, phases, np.ones(50), delay=0.5, delay_steps=5)
    assert traj.collective_frequency() == pytest.approx(slope, abs=1e-9)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_collective_frequency_rejects_bad_fraction(fraction: float) -> None:
    traj = DelayedTrajectory(
        np.arange(4.0), np.zeros((4, 2)), np.ones(4), delay=1.0, delay_steps=3
    )
    with pytest.raises(ValueError, match="fraction must be in"):
        traj.collective_frequency(fraction=fraction)


def test_collective_frequency_rejects_degenerate_single_sample() -> None:
    traj = DelayedTrajectory(
        np.array([0.0]), np.zeros((1, 2)), np.ones(1), delay=1.0, delay_steps=1
    )
    with pytest.raises(ValueError, match="too short"):
        traj.collective_frequency()


def test_collective_frequency_full_window_matches_quarter() -> None:
    times = np.linspace(0.0, 2.0, 80)
    phases = (1.4 * times)[:, None] + np.zeros((80, 3))
    traj = DelayedTrajectory(times, phases, np.ones(80), delay=0.5, delay_steps=5)
    assert traj.collective_frequency(fraction=1.0) == pytest.approx(1.4, abs=1e-9)


# --------------------------------------------------------------------------- integrator validation


def test_integrate_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        integrate_delayed_kuramoto(
            np.zeros((6, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.5, dt=0.0, n_steps=1
        )


def test_integrate_rejects_non_positive_delay() -> None:
    with pytest.raises(ValueError, match="delay must be positive"):
        integrate_delayed_kuramoto(
            np.zeros((6, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.0, dt=0.1, n_steps=1
        )


def test_integrate_rejects_non_positive_n_steps() -> None:
    with pytest.raises(ValueError, match="n_steps must be positive"):
        integrate_delayed_kuramoto(
            np.zeros((6, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.5, dt=0.1, n_steps=0
        )


def test_integrate_rejects_delay_below_one_step() -> None:
    # delay/dt = 0.4 rounds to zero steps.
    with pytest.raises(ValueError, match="integer multiple of dt"):
        integrate_delayed_kuramoto(
            np.zeros((1, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.04, dt=0.1, n_steps=1
        )


def test_integrate_rejects_non_integer_delay_multiple() -> None:
    # delay/dt = 1.5 is not an integer number of steps.
    with pytest.raises(ValueError, match="integer multiple of dt"):
        integrate_delayed_kuramoto(
            np.zeros((3, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.15, dt=0.1, n_steps=1
        )


def test_integrate_rejects_non_vector_omega() -> None:
    with pytest.raises(ValueError, match="omega must be a non-empty"):
        integrate_delayed_kuramoto(
            np.zeros((6, 2)),
            np.zeros((2, 2)),
            lambda c, d: np.zeros(2),
            delay=0.5,
            dt=0.1,
            n_steps=1,
        )


def test_integrate_rejects_wrong_history_shape() -> None:
    # delay/dt = 5 needs a (6, N) history; a (4, N) history is rejected.
    with pytest.raises(ValueError, match="initial_history must have shape"):
        integrate_delayed_kuramoto(
            np.zeros((4, 2)), np.zeros(2), lambda c, d: np.zeros(2), delay=0.5, dt=0.1, n_steps=1
        )


# --------------------------------------------------------------------------- integrator dynamics


def test_zero_coupling_is_exact_drift() -> None:
    # With zero coupling the right-hand side is the constant ω, for which RK4 is exact.
    dt, delay = 0.05, 0.25
    delay_steps = round(delay / dt)
    omega = np.array([0.3, -0.7, 1.1])
    history = _uniform_history(delay_steps, 3, drift=0.0, dt=dt)
    n_steps = 40
    traj = integrate_delayed_kuramoto(
        history,
        omega,
        lambda current, delayed: delayed_mean_field_force(current, delayed, 0.0),
        delay=delay,
        dt=dt,
        n_steps=n_steps,
    )
    expected = history[-1] + omega * (n_steps * dt)
    assert np.allclose(traj.terminal_phases, expected, atol=1e-12)
    assert traj.collective_frequency() == pytest.approx(float(omega.mean()), abs=1e-9)
    assert traj.times.shape == (n_steps + 1,)
    assert traj.phases.shape == (n_steps + 1, 3)
    assert traj.order_parameter_series.shape == (n_steps + 1,)
    assert traj.delay == delay
    assert traj.delay_steps == delay_steps


def test_single_oscillator_settles_to_self_consistency_root() -> None:
    # One oscillator: θ̇ = ω₀ + K sin(θ(t−τ) − θ(t)) → Ω = ω₀ − K sin(Ω τ).
    omega0, coupling, delay, dt = 1.0, 2.0, 1.0, 0.01
    delay_steps = round(delay / dt)
    history = _uniform_history(delay_steps, 1, drift=omega0, dt=dt)
    traj = integrate_delayed_kuramoto(
        np.asarray(history),
        np.array([omega0]),
        lambda current, delayed: delayed_mean_field_force(current, delayed, coupling),
        delay=delay,
        dt=dt,
        n_steps=8000,
    )
    omega = traj.collective_frequency()
    assert abs(synchronised_frequency_residual(omega, omega0, coupling, delay)) < 1e-3


def test_identical_oscillators_synchronise_to_unit_order_parameter() -> None:
    omega0, coupling, delay, dt = 0.5, 1.0, 0.8, 0.01
    delay_steps = round(delay / dt)
    history = _uniform_history(delay_steps, 12, drift=omega0, dt=dt)
    traj = integrate_delayed_kuramoto(
        history,
        np.full(12, omega0),
        lambda current, delayed: delayed_mean_field_force(current, delayed, coupling),
        delay=delay,
        dt=dt,
        n_steps=6000,
    )
    assert traj.order_parameter_series[-1] == pytest.approx(1.0, abs=1e-6)


def test_delay_induced_multistability_selects_branch_from_history() -> None:
    # Roadmap acceptance: at large K·τ several stable collective frequencies coexist, and which
    # one the run settles onto is set by the initial history (delay-induced multistability).
    omega0, coupling, delay, dt = 1.0, 5.0, 1.5, 0.01
    delay_steps = round(delay / dt)
    stable = stable_synchronised_frequencies(omega0, coupling, delay)
    assert stable.size >= 2  # multistable regime

    def settle(initial_drift: float) -> float:
        history = _uniform_history(delay_steps, 8, drift=initial_drift, dt=dt)
        traj = integrate_delayed_kuramoto(
            history,
            np.full(8, omega0),
            lambda current, delayed: delayed_mean_field_force(current, delayed, coupling),
            delay=delay,
            dt=dt,
            n_steps=12000,
        )
        return traj.collective_frequency()

    low_branch = settle(-2.0)
    high_branch = settle(3.0)
    # Each settles onto a *stable* root, and the two histories pick *different* branches.
    assert min(abs(stable - low_branch)) < 1e-2
    assert min(abs(stable - high_branch)) < 1e-2
    assert abs(low_branch - high_branch) > 1.0


def test_branch_frequency_refines_towards_root_as_step_shrinks() -> None:
    # Halving dt drives the settled collective frequency closer to the exact self-consistency root.
    omega0, coupling, delay = 1.0, 3.0, 1.2

    def settled_residual(dt: float) -> float:
        delay_steps = round(delay / dt)
        history = _uniform_history(delay_steps, 6, drift=3.0, dt=dt)
        traj = integrate_delayed_kuramoto(
            history,
            np.full(6, omega0),
            lambda current, delayed: delayed_mean_field_force(current, delayed, coupling),
            delay=delay,
            dt=dt,
            n_steps=int(round(120.0 / dt)),
        )
        return abs(
            synchronised_frequency_residual(traj.collective_frequency(), omega0, coupling, delay)
        )

    coarse = settled_residual(0.02)
    fine = settled_residual(0.01)
    assert fine <= coarse + 1e-6
    assert fine < 1e-3


def test_run_is_deterministic() -> None:
    omega0, coupling, delay, dt = 0.7, 1.5, 0.5, 0.02
    delay_steps = round(delay / dt)
    history = _uniform_history(delay_steps, 5, drift=0.6, dt=dt)

    def force(current: np.ndarray, delayed: np.ndarray) -> np.ndarray:
        return delayed_mean_field_force(current, delayed, coupling)

    first = integrate_delayed_kuramoto(
        history, np.full(5, omega0), force, delay=delay, dt=dt, n_steps=300
    )
    second = integrate_delayed_kuramoto(
        history, np.full(5, omega0), force, delay=delay, dt=dt, n_steps=300
    )
    assert np.array_equal(first.phases, second.phases)


def test_settled_branch_is_linearly_stable() -> None:
    # The realised branch satisfies the Yeung–Strogatz stability condition 1 + K·τ·cos(Ω·τ) > 0.
    from oscillatools.accel.kuramoto_delayed_mean_field import (
        synchronised_branch_stability,
    )

    omega0, coupling, delay, dt = 1.0, 5.0, 1.5, 0.01
    delay_steps = round(delay / dt)
    history = _uniform_history(delay_steps, 8, drift=3.0, dt=dt)
    traj = integrate_delayed_kuramoto(
        history,
        np.full(8, omega0),
        lambda current, delayed: delayed_mean_field_force(current, delayed, coupling),
        delay=delay,
        dt=dt,
        n_steps=12000,
    )
    assert synchronised_branch_stability(traj.collective_frequency(), coupling, delay) > 0.0
