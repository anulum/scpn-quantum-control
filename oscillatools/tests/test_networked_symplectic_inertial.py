# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the polyglot symplectic inertial networked-Kuramoto trajectory
r"""Tests for :mod:`oscillatools.accel.networked_symplectic_inertial`.

The accelerated networked-force velocity-Verlet integrator is checked against an exact solution (the
undamped free rotor, for which velocity-Verlet is exact), against the defining symplectic property
(bounded energy error without damping, monotone dissipation with it), and for bit-identity with the
general :func:`~oscillatools.accel.kuramoto_symplectic_inertial.integrate_symplectic_inertial` on the
same networked force. The Rust → Julia → Python floor tier chain is checked for tolerance-parity and
its fail-closed fall-through, and every input-validation branch is covered.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import oscillatools.accel.dispatcher as dispatcher_module
import oscillatools.accel.networked_symplectic_inertial as symplectic_module
from oscillatools.accel.kuramoto_inertial import InertialTrajectory
from oscillatools.accel.kuramoto_symplectic_inertial import integrate_symplectic_inertial
from oscillatools.accel.networked_symplectic_inertial import (
    _force,
    last_networked_symplectic_inertial_trajectory_tier_used,
    networked_symplectic_inertial_trajectory,
)


def _problem(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(theta0, v0, omega, coupling)`` with a symmetric zero-diagonal coupling."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    velocities = rng.normal(0.0, 0.3, count)
    omega = rng.normal(0.0, 1.0, count)
    coupling = rng.normal(0.0, 0.4, (count, count))
    coupling = coupling + coupling.T
    np.fill_diagonal(coupling, 0.0)
    return theta0, velocities, omega, coupling


def _networked_potential(coupling: np.ndarray):
    """Return the interaction potential ``U(θ) = −½ Σ K_jk cos(θ_k − θ_j)`` with ``F = −∇U``."""

    def potential(theta: np.ndarray) -> float:
        difference = theta[None, :] - theta[:, None]
        return float(-0.5 * (coupling * np.cos(difference)).sum())

    return potential


# --------------------------------------------------------------------------- ground-truth physics


def test_undamped_free_rotor_is_exact() -> None:
    """Velocity-Verlet is exact for the constant-acceleration undamped free rotor (no coupling)."""
    count = 6
    rng = np.random.default_rng(0)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    v0 = rng.normal(0.0, 0.5, count)
    omega = rng.normal(0.0, 1.0, count)
    coupling = np.zeros((count, count))
    mass, dt, n_steps = 1.3, 0.01, 400

    trajectory = networked_symplectic_inertial_trajectory(
        theta0, v0, omega, coupling, mass, damping=0.0, dt=dt, n_steps=n_steps
    )
    times = trajectory.times[:, None]
    velocity_exact = v0 + (omega / mass) * times
    phase_exact = theta0 + v0 * times + 0.5 * (omega / mass) * times**2
    np.testing.assert_allclose(trajectory.velocities, velocity_exact, atol=1e-10)
    np.testing.assert_allclose(trajectory.phases, phase_exact, atol=1e-10)


def test_undamped_flow_keeps_the_energy_error_bounded() -> None:
    """Without damping the scheme is symplectic: the mechanical-energy error stays bounded."""
    theta0, v0, omega, coupling = _problem(8, 2)
    trajectory = networked_symplectic_inertial_trajectory(
        theta0, v0, omega, coupling, mass=1.0, damping=0.0, dt=0.01, n_steps=800
    )
    energy = trajectory.energy(omega, _networked_potential(coupling))
    span = float(np.max(np.abs(energy - energy[0])))
    assert span < 5e-3


def test_damping_dissipates_the_mechanical_energy() -> None:
    """Under damping the mechanical energy is a Lyapunov function: it never increases."""
    theta0, v0, omega, coupling = _problem(10, 1)
    trajectory = networked_symplectic_inertial_trajectory(
        theta0, v0, omega, coupling, mass=1.0, damping=0.9, dt=0.01, n_steps=600
    )
    energy = trajectory.energy(omega, _networked_potential(coupling))
    assert np.all(np.diff(energy) <= 1e-9)
    assert energy[-1] < energy[0] - 1e-3


# --------------------------------------------------------------------------- floor equivalence


def test_floor_is_bit_identical_to_the_general_integrator() -> None:
    """The Python floor equals :func:`integrate_symplectic_inertial` bound to the networked force."""
    theta0, v0, omega, coupling = _problem(12, 3)
    mass, damping, dt, n_steps = 1.1, 0.5, 0.01, 300
    floor_times, floor_phases, floor_velocities = (
        symplectic_module._python_networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass, damping, dt, n_steps
        )
    )
    reference = integrate_symplectic_inertial(
        theta0,
        v0,
        omega,
        lambda theta: _force(theta, coupling),
        mass,
        damping=damping,
        dt=dt,
        n_steps=n_steps,
    )
    np.testing.assert_array_equal(floor_phases, reference.phases)
    np.testing.assert_array_equal(floor_velocities, reference.velocities)
    np.testing.assert_array_equal(floor_times, reference.times)


def test_trajectory_structure_is_consistent() -> None:
    """The returned trajectory has the documented shapes, sample times and metadata."""
    theta0, v0, omega, coupling = _problem(9, 4)
    dt, n_steps = 0.02, 150
    trajectory = networked_symplectic_inertial_trajectory(
        theta0, v0, omega, coupling, mass=1.4, damping=0.3, dt=dt, n_steps=n_steps
    )
    assert isinstance(trajectory, InertialTrajectory)
    assert trajectory.phases.shape == (n_steps + 1, 9)
    assert trajectory.velocities.shape == (n_steps + 1, 9)
    np.testing.assert_allclose(trajectory.times, dt * np.arange(n_steps + 1))
    np.testing.assert_array_equal(trajectory.phases[0], theta0)
    np.testing.assert_array_equal(trajectory.velocities[0], v0)
    assert trajectory.mass == 1.4
    assert trajectory.damping == 0.3


# --------------------------------------------------------------------------- validation branches


def test_rejects_non_vector_theta0() -> None:
    _, v0, omega, coupling = _problem(4, 5)
    with pytest.raises(ValueError, match="theta0 must be a non-empty one-dimensional array"):
        networked_symplectic_inertial_trajectory(
            np.zeros((2, 2)), v0, omega, coupling, mass=1.0, dt=0.01, n_steps=10
        )


def test_rejects_empty_theta0() -> None:
    with pytest.raises(ValueError, match="theta0 must be a non-empty one-dimensional array"):
        networked_symplectic_inertial_trajectory(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 0)), mass=1.0, dt=0.01, n_steps=10
        )


def test_rejects_mismatched_velocities() -> None:
    theta0, _, omega, coupling = _problem(5, 6)
    with pytest.raises(ValueError, match="velocities must match"):
        networked_symplectic_inertial_trajectory(
            theta0, np.zeros(4), omega, coupling, mass=1.0, dt=0.01, n_steps=10
        )


def test_rejects_mismatched_omega() -> None:
    theta0, v0, _, coupling = _problem(5, 7)
    with pytest.raises(ValueError, match="omega must match"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, np.zeros(4), coupling, mass=1.0, dt=0.01, n_steps=10
        )


def test_rejects_non_square_coupling() -> None:
    theta0, v0, omega, _ = _problem(5, 8)
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, np.zeros((5, 4)), mass=1.0, dt=0.01, n_steps=10
        )


def test_rejects_non_positive_mass() -> None:
    theta0, v0, omega, coupling = _problem(4, 9)
    with pytest.raises(ValueError, match="mass must be positive"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=0.0, dt=0.01, n_steps=10
        )


def test_rejects_negative_damping() -> None:
    theta0, v0, omega, coupling = _problem(4, 10)
    with pytest.raises(ValueError, match="damping must be non-negative"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=1.0, damping=-0.1, dt=0.01, n_steps=10
        )


def test_rejects_non_positive_dt() -> None:
    theta0, v0, omega, coupling = _problem(4, 11)
    with pytest.raises(ValueError, match="dt must be positive"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=1.0, dt=0.0, n_steps=10
        )


def test_rejects_non_positive_n_steps() -> None:
    theta0, v0, omega, coupling = _problem(4, 12)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=1.0, dt=0.01, n_steps=0
        )


# --------------------------------------------------------------------------- tier dispatch


class TestNetworkedSymplecticInertialTierDispatch:
    """The Rust → Julia → Python floor tier chain for the symplectic inertial forward trajectory."""

    def test_public_call_records_a_served_tier(self) -> None:
        theta0, v0, omega, coupling = _problem(16, 20)
        networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=1.0, damping=0.5, dt=0.01, n_steps=100
        )
        assert last_networked_symplectic_inertial_trajectory_tier_used() in {
            "rust",
            "julia",
            "python",
        }

    def test_python_floor_tier_direct(self) -> None:
        theta0, v0, omega, coupling = _problem(14, 21)
        times, phases, velocities = (
            symplectic_module._python_networked_symplectic_inertial_trajectory(
                theta0, v0, omega, coupling, 1.0, 0.5, 0.01, 120
            )
        )
        assert phases.shape == (times.size, 14)
        assert velocities.shape == (times.size, 14)
        assert times.size == 121

    def test_rust_tier_matches_python_floor(self) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        assert hasattr(engine, "kuramoto_symplectic_inertial_trajectory")
        theta0, v0, omega, coupling = _problem(32, 22)
        rust = symplectic_module._rust_networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, 1.2, 0.6, 0.01, 300
        )
        floor = symplectic_module._python_networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, 1.2, 0.6, 0.01, 300
        )
        np.testing.assert_allclose(rust[0], floor[0], atol=1e-12)
        np.testing.assert_allclose(rust[1], floor[1], atol=1e-11)
        np.testing.assert_allclose(rust[2], floor[2], atol=1e-11)

    def test_julia_tier_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        theta0, v0, omega, coupling = _problem(24, 23)
        julia = symplectic_module._julia_networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, 1.1, 0.4, 0.01, 250
        )
        floor = symplectic_module._python_networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, 1.1, 0.4, 0.01, 250
        )
        np.testing.assert_allclose(julia[0], floor[0], atol=1e-12)
        np.testing.assert_allclose(julia[1], floor[1], atol=1e-10)
        np.testing.assert_allclose(julia[2], floor[2], atol=1e-10)

    def test_rust_tier_raises_when_engine_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        theta0, v0, omega, coupling = _problem(8, 24)
        with pytest.raises(ModuleNotFoundError):
            symplectic_module._rust_networked_symplectic_inertial_trajectory(
                theta0, v0, omega, coupling, 1.0, 0.5, 0.01, 50
            )

    def test_rust_tier_raises_when_symbol_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: object())
        theta0, v0, omega, coupling = _problem(8, 25)
        with pytest.raises(ImportError):
            symplectic_module._rust_networked_symplectic_inertial_trajectory(
                theta0, v0, omega, coupling, 1.0, 0.5, 0.01, 50
            )

    def test_fall_through_to_floor_when_engine_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        monkeypatch.setitem(sys.modules, "oscillatools.accel.julia", None)
        theta0, v0, omega, coupling = _problem(10, 26)
        trajectory = networked_symplectic_inertial_trajectory(
            theta0, v0, omega, coupling, mass=1.0, damping=0.5, dt=0.01, n_steps=80
        )
        assert last_networked_symplectic_inertial_trajectory_tier_used() == "python"
        assert trajectory.phases.shape == (81, 10)
