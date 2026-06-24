# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Inertial (second-order) Kuramoto dynamics tests
"""Multi-angle tests for the inertial (second-order) Kuramoto model.

Covers the ``(θ, θ̇)`` phase-space vector field (its value, its block-structured Jacobian
cross-checked against a central finite difference), the overdamped reduction to the first-order
Kuramoto flow as the inertia vanishes, the RK4 integrator (sampling, terminal accessors,
reproducibility, energy conservation in the undamped limit), the mechanical-energy Lyapunov
function of the damped flow (``dE/dt = −γ‖v‖²``, monotone relaxation for ``ω = 0``) and the input
validation of every entry point.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scpn_quantum_control.accel import (
    InertialTrajectory,
    PhaseForce,
    PhaseJacobian,
    PhasePotential,
    inertial_energy,
    inertial_jacobian,
    inertial_vector_field,
    integrate_inertial,
    kuramoto_interaction_energy,
    mean_field_force,
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)


def _symmetric_coupling(n: int, seed: int) -> np.ndarray:
    """A symmetric zero-diagonal coupling matrix (a gradient force, so an energy exists)."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, n))
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _bound_force(coupling: np.ndarray) -> PhaseForce:
    return lambda theta: networked_kuramoto_force(theta, coupling)


def _bound_jacobian(coupling: np.ndarray) -> PhaseJacobian:
    return lambda theta: networked_kuramoto_jacobian(theta, coupling)


def _bound_potential(coupling: np.ndarray) -> PhasePotential:
    return lambda theta: kuramoto_interaction_energy(theta, coupling)


def _first_order_rk4(
    theta: np.ndarray, omega: np.ndarray, force: PhaseForce, dt: float, n_steps: int
) -> np.ndarray:
    """Terminal phases of the first-order Kuramoto flow ``θ̇ = ω + F(θ)`` by RK4."""
    state = theta.copy()
    for _ in range(n_steps):
        k1 = omega + force(state)
        k2 = omega + force(state + 0.5 * dt * k1)
        k3 = omega + force(state + 0.5 * dt * k2)
        k4 = omega + force(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _central_difference_jacobian(
    field: Callable[[np.ndarray], np.ndarray], point: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    dim = point.size
    out = np.empty((dim, dim), dtype=np.float64)
    for index in range(dim):
        shift = np.zeros(dim, dtype=np.float64)
        shift[index] = eps
        out[:, index] = (field(point + shift) - field(point - shift)) / (2.0 * eps)
    return out


_N = 6
_SEED = 7


class TestInertialVectorField:
    def test_matches_swing_equation_definition(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(_SEED)
        theta = rng.uniform(-np.pi, np.pi, _N)
        velocity = rng.standard_normal(_N)
        omega = rng.standard_normal(_N)
        mass, damping = 0.8, 1.3
        field = inertial_vector_field(
            theta, velocity, omega, _bound_force(coupling), mass, damping=damping
        )
        expected_acceleration = (
            omega + networked_kuramoto_force(theta, coupling) - damping * velocity
        ) / mass
        assert field.shape == (2 * _N,)
        np.testing.assert_allclose(field[:_N], velocity)
        np.testing.assert_allclose(field[_N:], expected_acceleration)

    def test_default_damping_is_unit(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(1)
        theta, velocity, omega = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        default = inertial_vector_field(theta, velocity, omega, _bound_force(coupling), 0.5)
        unit = inertial_vector_field(
            theta, velocity, omega, _bound_force(coupling), 0.5, damping=1.0
        )
        np.testing.assert_array_equal(default, unit)

    def test_overdamped_velocity_slaving(self) -> None:
        # As m → 0 the acceleration block forces v → (ω + F)/γ; here at the slaved velocity the
        # acceleration is identically zero regardless of the (small) inertia.
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(2)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        damping = 1.7
        slaved = (omega + networked_kuramoto_force(theta, coupling)) / damping
        field = inertial_vector_field(
            theta, slaved, omega, _bound_force(coupling), 1e-3, damping=damping
        )
        np.testing.assert_allclose(field[_N:], np.zeros(_N), atol=1e-12)

    @pytest.mark.parametrize(
        ("phases", "velocities", "omega", "match"),
        [
            (np.zeros(3), np.zeros(3), np.zeros((3, 1)), "non-empty one-dimensional"),
            (np.zeros(3), np.zeros(3), np.zeros(0), "non-empty one-dimensional"),
            (np.zeros(4), np.zeros(3), np.zeros(3), "phases must have shape"),
            (np.zeros(3), np.zeros(4), np.zeros(3), "velocities must have shape"),
        ],
    )
    def test_state_validation(
        self, phases: np.ndarray, velocities: np.ndarray, omega: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            inertial_vector_field(phases, velocities, omega, _bound_force(np.zeros((3, 3))), 1.0)

    def test_rejects_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            inertial_vector_field(
                np.zeros(3), np.zeros(3), np.zeros(3), _bound_force(np.zeros((3, 3))), 0.0
            )

    def test_rejects_negative_damping(self) -> None:
        with pytest.raises(ValueError, match="damping must be non-negative"):
            inertial_vector_field(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                _bound_force(np.zeros((3, 3))),
                1.0,
                damping=-0.1,
            )


class TestInertialJacobian:
    def test_block_structure(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(4)
        theta = rng.uniform(-np.pi, np.pi, _N)
        mass, damping = 0.6, 1.1
        jacobian = inertial_jacobian(theta, _bound_jacobian(coupling), mass, damping=damping)
        assert jacobian.shape == (2 * _N, 2 * _N)
        np.testing.assert_allclose(jacobian[:_N, :_N], np.zeros((_N, _N)))
        np.testing.assert_allclose(jacobian[:_N, _N:], np.eye(_N))
        np.testing.assert_allclose(
            jacobian[_N:, :_N], networked_kuramoto_jacobian(theta, coupling) / mass
        )
        np.testing.assert_allclose(jacobian[_N:, _N:], -(damping / mass) * np.eye(_N))

    def test_matches_central_difference(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, _N)
        velocity = rng.standard_normal(_N)
        omega = rng.standard_normal(_N)
        mass, damping = 0.9, 1.4
        force = _bound_force(coupling)

        def field(state: np.ndarray) -> np.ndarray:
            return inertial_vector_field(
                state[:_N], state[_N:], omega, force, mass, damping=damping
            )

        numerical = _central_difference_jacobian(field, np.concatenate([theta, velocity]))
        analytic = inertial_jacobian(theta, _bound_jacobian(coupling), mass, damping=damping)
        np.testing.assert_allclose(numerical, analytic, atol=1e-7)

    def test_rejects_empty_phases(self) -> None:
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            inertial_jacobian(np.zeros(0), _bound_jacobian(np.zeros((0, 0))), 1.0)

    def test_rejects_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            inertial_jacobian(np.zeros(3), _bound_jacobian(np.zeros((3, 3))), -1.0)

    def test_rejects_negative_damping(self) -> None:
        with pytest.raises(ValueError, match="damping must be non-negative"):
            inertial_jacobian(np.zeros(3), _bound_jacobian(np.zeros((3, 3))), 1.0, damping=-2.0)

    def test_rejects_mismatched_force_jacobian(self) -> None:
        def wrong(theta: np.ndarray) -> np.ndarray:
            return np.zeros((theta.size + 1, theta.size))

        with pytest.raises(ValueError, match="force_jacobian must return"):
            inertial_jacobian(np.zeros(3), wrong, 1.0)


class TestIntegrateInertial:
    def test_trajectory_shape_and_seed(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(8)
        theta, velocity, omega = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        trajectory = integrate_inertial(
            theta, velocity, omega, _bound_force(coupling), 0.7, dt=0.01, n_steps=50
        )
        assert isinstance(trajectory, InertialTrajectory)
        assert trajectory.phases.shape == (51, _N)
        assert trajectory.velocities.shape == (51, _N)
        assert trajectory.times.shape == (51,)
        np.testing.assert_allclose(trajectory.times, 0.01 * np.arange(51))
        np.testing.assert_array_equal(trajectory.phases[0], theta)
        np.testing.assert_array_equal(trajectory.velocities[0], velocity)
        np.testing.assert_array_equal(trajectory.terminal_phases, trajectory.phases[-1])
        np.testing.assert_array_equal(trajectory.terminal_velocities, trajectory.velocities[-1])
        assert trajectory.mass == 0.7
        assert trajectory.damping == 1.0

    def test_reduces_to_first_order_as_inertia_vanishes(self) -> None:
        # The overdamped limit is the statement that as m → 0 the inertial flow (γ = 1) tracks
        # the first-order Kuramoto flow θ̇ = ω + F(θ). The fast velocity mode relaxes on the
        # timescale m/γ, so the explicit RK4 step is scaled with the inertia (dt = 0.2·m) to keep
        # that mode resolved while a fixed physical horizon is integrated; the terminal-phase
        # gap against the first-order flow then shrinks ~linearly in m.
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(9)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        force = _bound_force(coupling)
        horizon = 0.3
        slaved = omega + networked_kuramoto_force(
            theta, coupling
        )  # γ = 1 slaved velocity at t = 0
        errors = []
        for mass in (0.04, 0.01, 0.0025):
            dt = 0.2 * mass
            n_steps = int(round(horizon / dt))
            reference = _first_order_rk4(theta, omega, force, dt, n_steps)
            trajectory = integrate_inertial(
                theta, slaved, omega, force, mass, dt=dt, n_steps=n_steps
            )
            errors.append(float(np.abs(trajectory.terminal_phases - reference).max()))
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 5e-3

    def test_undamped_flow_conserves_energy(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(10)
        theta, velocity = rng.uniform(-np.pi, np.pi, _N), 0.3 * rng.standard_normal(_N)
        omega = np.zeros(_N)
        trajectory = integrate_inertial(
            theta, velocity, omega, _bound_force(coupling), 0.9, damping=0.0, dt=0.002, n_steps=800
        )
        energy = trajectory.energy(omega, _bound_potential(coupling))
        assert float(energy.max() - energy.min()) < 1e-4

    def test_reproducible(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(11)
        theta, velocity, omega = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        first = integrate_inertial(
            theta, velocity, omega, _bound_force(coupling), 0.7, dt=0.01, n_steps=40
        )
        second = integrate_inertial(
            theta, velocity, omega, _bound_force(coupling), 0.7, dt=0.01, n_steps=40
        )
        np.testing.assert_array_equal(first.phases, second.phases)
        np.testing.assert_array_equal(first.velocities, second.velocities)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dt": 0.0, "n_steps": 10}, "dt must be positive"),
            ({"dt": 0.01, "n_steps": 0}, "n_steps must be positive"),
        ],
    )
    def test_step_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            integrate_inertial(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                _bound_force(np.zeros((3, 3))),
                1.0,
                **kwargs,
            )

    def test_rejects_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            integrate_inertial(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                _bound_force(np.zeros((3, 3))),
                0.0,
                dt=0.01,
                n_steps=5,
            )

    def test_rejects_negative_damping(self) -> None:
        with pytest.raises(ValueError, match="damping must be non-negative"):
            integrate_inertial(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                _bound_force(np.zeros((3, 3))),
                1.0,
                damping=-1.0,
                dt=0.01,
                n_steps=5,
            )

    def test_rejects_mismatched_state(self) -> None:
        with pytest.raises(ValueError, match="velocities must have shape"):
            integrate_inertial(
                np.zeros(3),
                np.zeros(4),
                np.zeros(3),
                _bound_force(np.zeros((3, 3))),
                1.0,
                dt=0.01,
                n_steps=5,
            )


class TestInertialEnergy:
    def test_matches_mechanical_energy_definition(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(12)
        theta, velocity, omega = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        mass = 1.2
        value = inertial_energy(theta, velocity, omega, _bound_potential(coupling), mass)
        expected = (
            0.5 * mass * float(np.dot(velocity, velocity))
            + kuramoto_interaction_energy(theta, coupling)
            - float(np.dot(omega, theta))
        )
        assert value == pytest.approx(expected)

    def test_lyapunov_dissipation_rate(self) -> None:
        # Along the damped flow dE/dt = −γ‖v‖²; compare a central difference of the energy
        # series to the closed-form rate.
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(13)
        theta, velocity = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        omega = rng.standard_normal(_N)
        mass, damping, dt = 0.8, 1.0, 0.001
        trajectory = integrate_inertial(
            theta,
            velocity,
            omega,
            _bound_force(coupling),
            mass,
            damping=damping,
            dt=dt,
            n_steps=400,
        )
        energy = trajectory.energy(omega, _bound_potential(coupling))
        numerical_rate = (energy[2:] - energy[:-2]) / (2.0 * dt)
        closed_form_rate = -damping * np.sum(trajectory.velocities[1:-1] ** 2, axis=1)
        np.testing.assert_allclose(numerical_rate, closed_form_rate, atol=5e-3)

    def test_monotone_relaxation_for_zero_frequencies(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(14)
        theta, velocity = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        omega = np.zeros(_N)
        trajectory = integrate_inertial(
            theta,
            velocity,
            omega,
            _bound_force(coupling),
            0.8,
            damping=1.0,
            dt=0.005,
            n_steps=3000,
        )
        energy = trajectory.energy(omega, _bound_potential(coupling))
        assert float(np.diff(energy).max()) <= 1e-6
        assert float(np.abs(trajectory.terminal_velocities).max()) < 1e-2

    def test_kinetic_energy_series(self) -> None:
        coupling = _symmetric_coupling(_N, _SEED)
        rng = np.random.default_rng(15)
        theta, velocity, omega = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        trajectory = integrate_inertial(
            theta, velocity, omega, _bound_force(coupling), 1.1, dt=0.01, n_steps=20
        )
        kinetic = trajectory.kinetic_energy()
        assert kinetic.shape == (21,)
        expected = 0.5 * 1.1 * np.sum(trajectory.velocities**2, axis=1)
        np.testing.assert_allclose(kinetic, expected)

    @pytest.mark.parametrize(
        ("phases", "velocities", "omega", "match"),
        [
            (np.zeros(3), np.zeros(4), np.zeros(3), "velocities must have shape"),
            (np.zeros(4), np.zeros(3), np.zeros(3), "phases must have shape"),
        ],
    )
    def test_state_validation(
        self, phases: np.ndarray, velocities: np.ndarray, omega: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            inertial_energy(phases, velocities, omega, _bound_potential(np.zeros((3, 3))), 1.0)

    def test_rejects_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            inertial_energy(
                np.zeros(3), np.zeros(3), np.zeros(3), _bound_potential(np.zeros((3, 3))), 0.0
            )


def test_mean_field_force_drives_inertial_field() -> None:
    # The vector field is agnostic to the Kuramoto model: the all-to-all mean-field force is a
    # valid PhaseForce too.
    rng = np.random.default_rng(16)
    theta, velocity, omega = (
        rng.uniform(-np.pi, np.pi, _N),
        rng.standard_normal(_N),
        rng.standard_normal(_N),
    )

    def force(phases: np.ndarray) -> np.ndarray:
        return mean_field_force(phases, 2.0)

    field = inertial_vector_field(theta, velocity, omega, force, 0.9)
    np.testing.assert_allclose(field[:_N], velocity)
    np.testing.assert_allclose(field[_N:], (omega + mean_field_force(theta, 2.0) - velocity) / 0.9)


def test_public_symbols_exported() -> None:
    import scpn_quantum_control.accel as accel

    for symbol in (
        "InertialTrajectory",
        "PhaseForce",
        "PhaseJacobian",
        "PhasePotential",
        "inertial_energy",
        "inertial_jacobian",
        "inertial_vector_field",
        "integrate_inertial",
    ):
        assert symbol in accel.__all__
        assert hasattr(accel, symbol)
