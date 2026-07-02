# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Unified first-order Kuramoto system/problem object tests
"""Contract, validation, and numerical tests for :mod:`kuramoto_system`.

The system object is a facade, so the load-bearing evidence is twofold: the
DynamicalSystems-style ``state / parameter / rule`` contract behaves correctly
(defensive copies, one-call reinit, parameter tuning, fail-closed validation),
and its generic RK4 stepper reproduces the dedicated networked RK4 integrator
``kuramoto_rk4_trajectory`` to machine precision — the independent anchor that
the composed rule and stepper agree with the shipped compute path.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory
from scpn_quantum_control.accel.kuramoto_mean_field import mean_field_force
from scpn_quantum_control.accel.kuramoto_system import (
    KuramotoParameters,
    KuramotoSystem,
    mean_field_phase_rule,
    networked_phase_rule,
)
from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force
from scpn_quantum_control.accel.order_parameter_observables import order_parameter
from scpn_quantum_control.accel.sakaguchi_kuramoto import sakaguchi_force
from scpn_quantum_control.accel.sakaguchi_mean_field import sakaguchi_mean_field_force


def _ring_coupling(size: int, strength: float) -> np.ndarray:
    """Return a symmetric nearest-neighbour ring coupling matrix."""

    matrix = np.zeros((size, size), dtype=np.float64)
    for node in range(size):
        matrix[node, (node + 1) % size] = strength
        matrix[node, (node - 1) % size] = strength
    return matrix


# --------------------------------------------------------------------------- #
# KuramotoParameters — construction, coercion, validation
# --------------------------------------------------------------------------- #
class TestKuramotoParameters:
    def test_mean_field_construction_coerces_scalar_and_array(self) -> None:
        parameters = KuramotoParameters([0.1, -0.2, 0.3], 1)
        assert parameters.size == 3
        assert parameters.is_networked is False
        assert isinstance(parameters.coupling, float)
        assert parameters.coupling == 1.0
        assert parameters.natural_frequencies.dtype == np.float64

    def test_networked_construction_keeps_matrix(self) -> None:
        coupling = _ring_coupling(4, 0.5)
        parameters = KuramotoParameters(np.zeros(4), coupling)
        assert parameters.is_networked is True
        assert isinstance(parameters.coupling, np.ndarray)
        np.testing.assert_array_equal(parameters.coupling, coupling)

    def test_frustration_defaults_to_zero_and_coerces(self) -> None:
        assert KuramotoParameters(np.zeros(2), 1.0).frustration == 0.0
        assert KuramotoParameters(np.zeros(2), 1.0, frustration=1).frustration == 1.0

    @pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.array([]), np.float64(1.0)])
    def test_rejects_non_vector_natural_frequencies(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-empty one-dimensional"):
            KuramotoParameters(bad, 1.0)

    def test_rejects_mismatched_coupling_matrix(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, N\) matrix"):
            KuramotoParameters(np.zeros(3), np.zeros((2, 2)))

    def test_size_and_is_networked_properties(self) -> None:
        assert KuramotoParameters(np.zeros(5), 0.3).size == 5
        assert KuramotoParameters(np.zeros(3), _ring_coupling(3, 0.2)).is_networked is True

    def test_with_parameter_replaces_each_field(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), 0.5, frustration=0.1)
        retuned = parameters.with_parameter("coupling", 0.9)
        assert retuned.coupling == 0.9
        assert parameters.coupling == 0.5  # original unchanged (frozen)
        assert parameters.with_parameter("frustration", 0.4).frustration == 0.4
        replaced = parameters.with_parameter("natural_frequencies", [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(replaced.natural_frequencies, [1.0, 2.0, 3.0])

    def test_with_parameter_revalidates(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), 0.5)
        with pytest.raises(ValueError, match=r"\(N, N\) matrix"):
            parameters.with_parameter("coupling", np.zeros((2, 2)))

    def test_with_parameter_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown Kuramoto parameter"):
            KuramotoParameters(np.zeros(2), 1.0).with_parameter("mass", 1.0)


# --------------------------------------------------------------------------- #
# Rules — topology dispatch, frustration, correctness
# --------------------------------------------------------------------------- #
class TestPhaseRules:
    def test_mean_field_rule_equals_omega_plus_force(self) -> None:
        theta = np.array([0.2, -0.5, 1.1, 0.7])
        omega = np.array([0.1, 0.2, -0.3, 0.4])
        parameters = KuramotoParameters(omega, 0.8)
        np.testing.assert_allclose(
            mean_field_phase_rule(theta, parameters, 0.0),
            omega + mean_field_force(theta, 0.8),
        )

    def test_mean_field_rule_uses_sakaguchi_when_frustrated(self) -> None:
        theta = np.array([0.2, -0.5, 1.1, 0.7])
        omega = np.zeros(4)
        parameters = KuramotoParameters(omega, 0.8, frustration=0.3)
        np.testing.assert_allclose(
            mean_field_phase_rule(theta, parameters, 0.0),
            omega + sakaguchi_mean_field_force(theta, 0.8, 0.3),
        )

    def test_mean_field_rule_rejects_matrix_coupling(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), _ring_coupling(3, 0.5))
        with pytest.raises(ValueError, match="scalar coupling"):
            mean_field_phase_rule(np.zeros(3), parameters, 0.0)

    def test_networked_rule_equals_omega_plus_force(self) -> None:
        theta = np.array([0.2, -0.5, 1.1])
        omega = np.array([0.1, 0.2, -0.3])
        coupling = _ring_coupling(3, 0.6)
        parameters = KuramotoParameters(omega, coupling)
        np.testing.assert_allclose(
            networked_phase_rule(theta, parameters, 0.0),
            omega + networked_kuramoto_force(theta, coupling),
        )

    def test_networked_rule_uses_sakaguchi_when_frustrated(self) -> None:
        theta = np.array([0.2, -0.5, 1.1])
        coupling = _ring_coupling(3, 0.6)
        parameters = KuramotoParameters(np.zeros(3), coupling, frustration=0.25)
        np.testing.assert_allclose(
            networked_phase_rule(theta, parameters, 0.0),
            sakaguchi_force(theta, coupling, 0.25),
        )

    def test_networked_rule_rejects_scalar_coupling(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), 0.5)
        with pytest.raises(ValueError, match="coupling matrix"):
            networked_phase_rule(np.zeros(3), parameters, 0.0)


# --------------------------------------------------------------------------- #
# KuramotoSystem — construction and validation
# --------------------------------------------------------------------------- #
class TestSystemConstruction:
    def test_mean_field_factory(self) -> None:
        system = KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.01)
        assert system.dimension == 3
        assert system.scheme == "rk4"
        assert system.dt == 0.01
        assert system.current_parameters.is_networked is False

    def test_networked_factory(self) -> None:
        system = KuramotoSystem.networked(
            np.zeros(4), np.zeros(4), _ring_coupling(4, 0.3), dt=0.02, scheme="euler"
        )
        assert system.current_parameters.is_networked is True
        assert system.scheme == "euler"

    def test_rejects_non_vector_state(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional phase vector"):
            KuramotoSystem.mean_field(np.zeros((2, 2)), np.zeros(4), 0.5, dt=0.01)

    def test_rejects_state_parameter_size_mismatch(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), 0.5)
        with pytest.raises(ValueError, match="same length"):
            KuramotoSystem(mean_field_phase_rule, np.zeros(4), parameters, dt=0.01)

    def test_rejects_unknown_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme must be one of"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.01, scheme="midpoint")

    @pytest.mark.parametrize("bad_dt", [0.0, -0.1])
    def test_rejects_non_positive_dt(self, bad_dt: float) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=bad_dt)


# --------------------------------------------------------------------------- #
# KuramotoSystem — state/parameter contract
# --------------------------------------------------------------------------- #
class TestSystemContract:
    def _system(self) -> KuramotoSystem:
        return KuramotoSystem.mean_field(
            np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 0.1, -0.1, 0.2]), 0.7, dt=0.05
        )

    def test_current_state_is_a_defensive_copy(self) -> None:
        system = self._system()
        snapshot = system.current_state
        snapshot[0] = 99.0
        assert system.current_state[0] == 0.1

    def test_set_state_updates_and_copies(self) -> None:
        system = self._system()
        new_state = np.array([1.0, 1.1, 1.2, 1.3])
        system.set_state(new_state)
        np.testing.assert_array_equal(system.current_state, new_state)
        new_state[0] = 5.0
        assert system.current_state[0] == 1.0  # stored copy is independent

    def test_set_state_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            self._system().set_state(np.zeros(3))

    def test_set_parameter_tunes_in_place(self) -> None:
        system = self._system()
        system.set_parameter("coupling", 2.0)
        assert system.current_parameters.coupling == 2.0

    def test_set_parameter_rejects_oscillator_count_change(self) -> None:
        system = self._system()
        with pytest.raises(ValueError, match="number of oscillators"):
            system.set_parameter("natural_frequencies", np.zeros(6))

    def test_set_parameter_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown Kuramoto parameter"):
            self._system().set_parameter("damping", 1.0)

    def test_reinit_returns_to_initial_state_and_time(self) -> None:
        system = self._system()
        system.step(n=5)
        assert system.current_time > 0.0
        system.reinit()
        np.testing.assert_array_equal(system.current_state, system.initial_state)
        assert system.current_time == 0.0

    def test_reinit_with_explicit_state_and_time(self) -> None:
        system = self._system()
        system.step(n=3)
        target = np.array([0.5, 0.6, 0.7, 0.8])
        system.reinit(target, time=2.0)
        np.testing.assert_array_equal(system.current_state, target)
        assert system.current_time == 2.0
        # The stored initial state is untouched, so an argument-free reinit still works.
        system.reinit()
        np.testing.assert_array_equal(system.current_state, system.initial_state)

    def test_reinit_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            self._system().reinit(np.zeros(2))

    def test_initial_state_is_a_defensive_copy(self) -> None:
        system = self._system()
        stored = system.initial_state
        stored[0] = 42.0
        assert system.initial_state[0] == 0.1

    def test_rule_value_matches_the_rule_at_current_and_given_time(self) -> None:
        system = self._system()
        parameters = system.current_parameters
        np.testing.assert_allclose(
            system.rule_value(), mean_field_phase_rule(system.current_state, parameters, 0.0)
        )
        # Autonomous rule: an explicit time gives the same velocity.
        np.testing.assert_allclose(system.rule_value(time=3.0), system.rule_value())

    def test_repr_reports_topology(self) -> None:
        assert "topology=mean_field" in repr(self._system())
        networked = KuramotoSystem.networked(
            np.zeros(3), np.zeros(3), _ring_coupling(3, 0.2), dt=0.01
        )
        assert "topology=networked" in repr(networked)


# --------------------------------------------------------------------------- #
# KuramotoSystem — stepping and trajectory
# --------------------------------------------------------------------------- #
class TestSystemStepping:
    def test_single_step_advances_state_and_time(self) -> None:
        system = KuramotoSystem.mean_field(np.array([0.1, 0.5, 0.9]), np.zeros(3), 0.6, dt=0.1)
        before = system.current_state
        returned = system.step()
        assert system.current_time == pytest.approx(0.1)
        np.testing.assert_array_equal(returned, system.current_state)
        assert not np.array_equal(returned, before)

    def test_multiple_steps_accumulate_time(self) -> None:
        system = KuramotoSystem.mean_field(np.zeros(3), np.ones(3), 0.0, dt=0.1)
        system.step(n=4)
        assert system.current_time == pytest.approx(0.4)

    def test_step_dt_override(self) -> None:
        system = KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.1)
        system.step(dt=0.25)
        assert system.current_time == pytest.approx(0.25)

    def test_step_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError, match="n must be a positive integer"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.1).step(n=0)

    def test_step_rejects_non_positive_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.1).step(dt=0.0)

    def test_trajectory_shape_and_start_row(self) -> None:
        system = KuramotoSystem.mean_field(np.array([0.1, 0.2, 0.3]), np.zeros(3), 0.4, dt=0.05)
        start = system.current_state
        path = system.trajectory(10)
        assert path.shape == (11, 3)
        np.testing.assert_array_equal(path[0], start)
        # The system is left at the final recorded row.
        np.testing.assert_array_equal(system.current_state, path[-1])
        assert system.current_time == pytest.approx(0.5)

    def test_trajectory_rejects_non_positive_steps(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.1).trajectory(0)

    def test_trajectory_rejects_non_positive_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.1).trajectory(5, dt=-1.0)

    def test_euler_step_matches_manual_formula(self) -> None:
        theta0 = np.array([0.2, 0.6, 1.0])
        omega = np.array([0.1, -0.2, 0.3])
        system = KuramotoSystem.mean_field(theta0, omega, 0.5, dt=0.01, scheme="euler")
        system.step()
        expected = theta0 + 0.01 * (omega + mean_field_force(theta0, 0.5))
        np.testing.assert_allclose(system.current_state, expected)


# --------------------------------------------------------------------------- #
# Numerical cross-checks — the load-bearing correctness evidence
# --------------------------------------------------------------------------- #
class TestNumericalEquivalence:
    def test_rk4_trajectory_matches_dedicated_networked_integrator(self) -> None:
        rng = np.random.default_rng(20260702)
        size = 6
        theta0 = rng.uniform(-np.pi, np.pi, size)
        omega = rng.normal(0.0, 0.5, size)
        coupling = _ring_coupling(size, 0.8)
        dt = 0.01
        n_steps = 200
        system = KuramotoSystem.networked(theta0, omega, coupling, dt=dt)
        reference = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        np.testing.assert_allclose(system.trajectory(n_steps), reference, rtol=0.0, atol=1e-12)

    def test_mean_field_above_critical_coupling_synchronises(self) -> None:
        rng = np.random.default_rng(11)
        size = 40
        theta0 = rng.uniform(-np.pi, np.pi, size)
        omega = rng.normal(0.0, 0.05, size)
        system = KuramotoSystem.mean_field(theta0, omega, 4.0, dt=0.01)
        start = float(order_parameter(system.current_state))
        final = system.trajectory(2000)[-1]
        assert float(order_parameter(final)) > start
        assert float(order_parameter(final)) > 0.9

    def test_reinit_makes_the_run_reproducible(self) -> None:
        system = KuramotoSystem.mean_field(
            np.array([0.1, 0.7, 1.3, 2.0]), np.array([0.2, -0.1, 0.3, 0.0]), 1.2, dt=0.02
        )
        first = system.trajectory(50)
        system.reinit()
        second = system.trajectory(50)
        np.testing.assert_array_equal(first, second)

    def test_set_parameter_changes_the_flow(self) -> None:
        theta0 = np.array([0.1, 0.5, 0.9, 1.4])
        omega = np.zeros(4)
        plain = KuramotoSystem.mean_field(theta0, omega, 1.0, dt=0.02)
        frustrated = KuramotoSystem.mean_field(theta0, omega, 1.0, dt=0.02)
        frustrated.set_parameter("frustration", 0.6)
        assert not np.allclose(plain.trajectory(100)[-1], frustrated.trajectory(100)[-1])

    @settings(max_examples=25, deadline=None)
    @given(
        coupling=st.floats(min_value=0.0, max_value=3.0),
        dt=st.floats(min_value=1e-3, max_value=5e-2),
    )
    def test_rk4_step_is_fourth_order_consistent_with_two_half_steps(
        self, coupling: float, dt: float
    ) -> None:
        theta0 = np.array([0.2, -0.4, 0.9, 1.5, -1.1])
        omega = np.array([0.1, 0.2, -0.3, 0.05, -0.15])
        full = KuramotoSystem.mean_field(theta0, omega, coupling, dt=dt)
        halved = KuramotoSystem.mean_field(theta0, omega, coupling, dt=dt / 2.0)
        full.step()
        halved.step(n=2)
        # Two half RK4 steps and one full step of a smooth flow agree to O(dt^5).
        np.testing.assert_allclose(full.current_state, halved.current_state, atol=5e-6)
