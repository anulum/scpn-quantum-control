# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the polyglot time-delayed networked-Kuramoto trajectory
r"""Tests for :mod:`oscillatools.accel.networked_delayed`.

The accelerated networked-force method-of-steps integrator is checked against two exact solutions
(the zero-coupling free rotor, which the delay-aware RK4 reproduces to machine precision, and a
Yeung–Strogatz phase-locked branch of the delayed mean-field, which a rigidly rotating history keeps
exactly locked), against the order-parameter series definition, and for bit-identity with the general
:func:`~oscillatools.accel.kuramoto_delayed.integrate_delayed_kuramoto` on the same networked delayed
force. The Rust → Julia → Python floor tier chain is checked for tolerance-parity and its fail-closed
fall-through, and every input-validation branch is covered.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import oscillatools.accel.dispatcher as dispatcher_module
import oscillatools.accel.networked_delayed as delayed_module
from oscillatools.accel.kuramoto_delayed import (
    DelayedTrajectory,
    delayed_networked_force,
    integrate_delayed_kuramoto,
)
from oscillatools.accel.networked_delayed import (
    last_networked_delayed_trajectory_tier_used,
    networked_delayed_trajectory,
)


def _problem(
    count: int, seed: int, *, delay_steps: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return ``(initial_history, omega, coupling, delay, dt)`` with a symmetric zero-diagonal ``K``.

    The history varies across its rows so the half-step delay interpolation is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    dt = 0.01
    delay = delay_steps * dt
    omega = rng.normal(0.0, 1.0, count)
    coupling = rng.normal(0.0, 0.4, (count, count))
    coupling = coupling + coupling.T
    np.fill_diagonal(coupling, 0.0)
    base = rng.uniform(0.0, 2.0 * np.pi, count)
    history = base[None, :] + rng.normal(0.0, 0.1, (delay_steps + 1, count))
    return history, omega, coupling, delay, dt


# --------------------------------------------------------------------------- ground-truth physics


def test_zero_coupling_is_the_free_rotor() -> None:
    """With no coupling each oscillator is a free rotor ``θ_j(t) = θ_j(0) + ω_j t`` — RK4 is exact."""
    count = 5
    rng = np.random.default_rng(0)
    dt = 0.01
    delay_steps = 3
    delay = delay_steps * dt
    omega = rng.normal(0.0, 1.0, count)
    coupling = np.zeros((count, count))
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    history = np.array(
        [theta0 + omega * (-(delay_steps - s) * dt) for s in range(delay_steps + 1)]
    )
    n_steps = 200

    trajectory = networked_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=dt, n_steps=n_steps
    )
    exact = theta0[None, :] + omega[None, :] * trajectory.times[:, None]
    np.testing.assert_allclose(trajectory.phases, exact, atol=1e-11)


def test_locked_synchronised_branch_is_reproduced() -> None:
    """A rigidly rotating history at a Yeung–Strogatz root ``Ω = ω₀ − K sin(Ωτ)`` stays locked.

    For identical oscillators under uniform all-to-all delayed coupling the phase-locked state
    ``θ_j(t) = Ω t`` (all phases equal) solves the self-consistency; started from its exact history it
    is reproduced by the delay-aware RK4, with unit coherence throughout and collective frequency
    ``Ω``.
    """
    count = 6
    dt = 0.01
    delay_steps = 5
    delay = delay_steps * dt
    collective, coupling_strength = 0.7, 1.3
    coupling = np.full((count, count), coupling_strength / count)
    natural = collective + coupling_strength * np.sin(collective * delay)
    omega = np.full(count, natural)
    history = np.array(
        [np.full(count, collective * (-(delay_steps - s) * dt)) for s in range(delay_steps + 1)]
    )
    n_steps = 300

    trajectory = networked_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=dt, n_steps=n_steps
    )
    exact = (collective * trajectory.times)[:, None] * np.ones(count)[None, :]
    np.testing.assert_allclose(trajectory.phases, exact, atol=1e-8)
    np.testing.assert_allclose(trajectory.order_parameter_series, 1.0, atol=1e-12)
    assert abs(trajectory.collective_frequency() - collective) < 1e-6


def test_order_parameter_series_matches_the_phases() -> None:
    """The order-parameter series equals ``|⟨e^{iθ}⟩|`` evaluated row-wise over the phases."""
    history, omega, coupling, delay, dt = _problem(8, 3)
    trajectory = networked_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=dt, n_steps=150
    )
    expected = np.abs(np.mean(np.exp(1j * trajectory.phases), axis=1))
    np.testing.assert_array_equal(trajectory.order_parameter_series, expected)


# --------------------------------------------------------------------------- floor equivalence


def test_floor_is_bit_identical_to_the_general_integrator() -> None:
    """The Python floor equals :func:`integrate_delayed_kuramoto` bound to the networked force."""
    history, omega, coupling, delay, dt = _problem(10, 5)
    n_steps = 200
    floor_times, floor_phases = delayed_module._python_networked_delayed_trajectory(
        history, omega, coupling, delay, dt, n_steps
    )
    reference = integrate_delayed_kuramoto(
        np.ascontiguousarray(history, dtype=np.float64),
        omega,
        lambda current, lagged: delayed_networked_force(current, lagged, coupling),
        delay=delay,
        dt=dt,
        n_steps=n_steps,
    )
    np.testing.assert_array_equal(floor_phases, reference.phases)
    np.testing.assert_array_equal(floor_times, reference.times)


def test_trajectory_structure_and_metadata() -> None:
    """The returned trajectory has the documented shapes, sample times and delay metadata."""
    history, omega, coupling, delay, dt = _problem(7, 4, delay_steps=4)
    n_steps = 120
    trajectory = networked_delayed_trajectory(
        history, omega, coupling, delay=delay, dt=dt, n_steps=n_steps
    )
    assert isinstance(trajectory, DelayedTrajectory)
    assert trajectory.phases.shape == (n_steps + 1, 7)
    assert trajectory.order_parameter_series.shape == (n_steps + 1,)
    np.testing.assert_allclose(trajectory.times, dt * np.arange(n_steps + 1))
    np.testing.assert_array_equal(trajectory.phases[0], history[-1])
    assert trajectory.delay == delay
    assert trajectory.delay_steps == 4


# --------------------------------------------------------------------------- validation branches


def test_rejects_non_positive_dt() -> None:
    history, omega, coupling, delay, _ = _problem(4, 5)
    with pytest.raises(ValueError, match="dt must be positive"):
        networked_delayed_trajectory(history, omega, coupling, delay=delay, dt=0.0, n_steps=10)


def test_rejects_non_positive_delay() -> None:
    history, omega, coupling, _, dt = _problem(4, 6)
    with pytest.raises(ValueError, match="delay must be positive"):
        networked_delayed_trajectory(history, omega, coupling, delay=0.0, dt=dt, n_steps=10)


def test_rejects_non_positive_n_steps() -> None:
    history, omega, coupling, delay, dt = _problem(4, 7)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        networked_delayed_trajectory(history, omega, coupling, delay=delay, dt=dt, n_steps=0)


def test_rejects_delay_not_an_integer_multiple_of_dt() -> None:
    history, omega, coupling, delay, dt = _problem(4, 8)
    with pytest.raises(ValueError, match="integer multiple of dt"):
        networked_delayed_trajectory(
            history, omega, coupling, delay=delay + 0.5 * dt, dt=dt, n_steps=10
        )


def test_rejects_non_vector_omega() -> None:
    history, _, coupling, delay, dt = _problem(4, 9)
    with pytest.raises(ValueError, match="omega must be a non-empty one-dimensional array"):
        networked_delayed_trajectory(
            history, np.zeros((4, 1)), coupling, delay=delay, dt=dt, n_steps=10
        )


def test_rejects_mismatched_history_shape() -> None:
    history, omega, coupling, delay, dt = _problem(5, 10)
    with pytest.raises(ValueError, match="initial_history must have shape"):
        networked_delayed_trajectory(history[:-1], omega, coupling, delay=delay, dt=dt, n_steps=10)


def test_rejects_non_square_coupling() -> None:
    history, omega, _, delay, dt = _problem(5, 11)
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        networked_delayed_trajectory(
            history, omega, np.zeros((5, 4)), delay=delay, dt=dt, n_steps=10
        )


# --------------------------------------------------------------------------- tier dispatch


class TestNetworkedDelayedTierDispatch:
    """The Rust → Julia → Python floor tier chain for the delayed forward trajectory."""

    def test_public_call_records_a_served_tier(self) -> None:
        history, omega, coupling, delay, dt = _problem(16, 20)
        networked_delayed_trajectory(history, omega, coupling, delay=delay, dt=dt, n_steps=100)
        assert last_networked_delayed_trajectory_tier_used() in {"rust", "julia", "python"}

    def test_python_floor_tier_direct(self) -> None:
        history, omega, coupling, delay, dt = _problem(14, 21)
        times, phases = delayed_module._python_networked_delayed_trajectory(
            history, omega, coupling, delay, dt, 120
        )
        assert phases.shape == (times.size, 14)
        assert times.size == 121

    def test_rust_tier_matches_python_floor(self) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        assert hasattr(engine, "kuramoto_delayed_trajectory")
        history, omega, coupling, delay, dt = _problem(32, 22, delay_steps=6)
        rust = delayed_module._rust_networked_delayed_trajectory(
            history, omega, coupling, delay, dt, 300
        )
        floor = delayed_module._python_networked_delayed_trajectory(
            history, omega, coupling, delay, dt, 300
        )
        np.testing.assert_allclose(rust[0], floor[0], atol=1e-12)
        np.testing.assert_allclose(rust[1], floor[1], atol=1e-11)

    def test_julia_tier_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        history, omega, coupling, delay, dt = _problem(24, 23, delay_steps=5)
        julia = delayed_module._julia_networked_delayed_trajectory(
            history, omega, coupling, delay, dt, 250
        )
        floor = delayed_module._python_networked_delayed_trajectory(
            history, omega, coupling, delay, dt, 250
        )
        np.testing.assert_allclose(julia[0], floor[0], atol=1e-12)
        np.testing.assert_allclose(julia[1], floor[1], atol=1e-10)

    def test_rust_tier_raises_when_engine_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        history, omega, coupling, delay, dt = _problem(8, 24)
        with pytest.raises(ModuleNotFoundError):
            delayed_module._rust_networked_delayed_trajectory(
                history, omega, coupling, delay, dt, 50
            )

    def test_rust_tier_raises_when_symbol_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: object())
        history, omega, coupling, delay, dt = _problem(8, 25)
        with pytest.raises(ImportError):
            delayed_module._rust_networked_delayed_trajectory(
                history, omega, coupling, delay, dt, 50
            )

    def test_fall_through_to_floor_when_engine_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        monkeypatch.setitem(sys.modules, "oscillatools.accel.julia", None)
        history, omega, coupling, delay, dt = _problem(10, 26)
        trajectory = networked_delayed_trajectory(
            history, omega, coupling, delay=delay, dt=dt, n_steps=80
        )
        assert last_networked_delayed_trajectory_tier_used() == "python"
        assert trajectory.phases.shape == (81, 10)
