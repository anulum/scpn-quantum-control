# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SciPy solve_ivp interop tests for the Kuramoto system
"""Tests for the SciPy ``solve_ivp`` interop and the system analytic Jacobian.

Two things carry the correctness argument. The analytic rule Jacobians are
checked against a finite-difference Jacobian of the rule (the independent anchor
that ``∂f/∂θ`` is right), and ``solve_kuramoto_ivp`` at a tight tolerance is
checked against both the system's own RK4 trajectory and the dedicated
``kuramoto_rk4_trajectory`` integrator — so the composed SciPy path is shown to
integrate the same dynamics, and the stiff (implicit, Jacobian-fed) path is shown
to agree with the explicit one.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory
from scpn_quantum_control.accel.kuramoto_mean_field import mean_field_jacobian
from scpn_quantum_control.accel.kuramoto_scipy_interop import (
    KuramotoIvpSolution,
    kuramoto_ode_jacobian,
    kuramoto_ode_rhs,
    solve_kuramoto_ivp,
)
from scpn_quantum_control.accel.kuramoto_system import (
    KuramotoParameters,
    KuramotoSystem,
    mean_field_phase_rule,
    mean_field_phase_rule_jacobian,
    networked_phase_rule,
    networked_phase_rule_jacobian,
)
from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_jacobian
from scpn_quantum_control.accel.sakaguchi_kuramoto import sakaguchi_jacobian
from scpn_quantum_control.accel.sakaguchi_mean_field import sakaguchi_mean_field_jacobian


def _ring_coupling(size: int, strength: float) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    for node in range(size):
        matrix[node, (node + 1) % size] = strength
        matrix[node, (node - 1) % size] = strength
    return matrix


def _finite_difference_jacobian(
    rule: object, state: np.ndarray, parameters: KuramotoParameters, *, step: float = 1e-6
) -> np.ndarray:
    """Central finite-difference Jacobian of ``rule(state, parameters, 0)``."""

    size = state.size
    jacobian = np.zeros((size, size), dtype=np.float64)
    for column in range(size):
        perturbation = np.zeros(size, dtype=np.float64)
        perturbation[column] = step
        plus = np.asarray(rule(state + perturbation, parameters, 0.0), dtype=np.float64)  # type: ignore[operator]
        minus = np.asarray(rule(state - perturbation, parameters, 0.0), dtype=np.float64)  # type: ignore[operator]
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


# --------------------------------------------------------------------------- #
# Analytic rule Jacobians
# --------------------------------------------------------------------------- #
class TestPhaseRuleJacobians:
    def test_mean_field_jacobian_equals_force_jacobian(self) -> None:
        theta = np.array([0.2, -0.5, 1.1, 0.7])
        parameters = KuramotoParameters(np.zeros(4), 0.8)
        np.testing.assert_allclose(
            mean_field_phase_rule_jacobian(theta, parameters, 0.0),
            mean_field_jacobian(theta, 0.8),
        )

    def test_mean_field_jacobian_uses_sakaguchi_when_frustrated(self) -> None:
        theta = np.array([0.2, -0.5, 1.1, 0.7])
        parameters = KuramotoParameters(np.zeros(4), 0.8, frustration=0.3)
        np.testing.assert_allclose(
            mean_field_phase_rule_jacobian(theta, parameters, 0.0),
            sakaguchi_mean_field_jacobian(theta, 0.8, 0.3),
        )

    def test_networked_jacobian_equals_force_jacobian(self) -> None:
        theta = np.array([0.2, -0.5, 1.1])
        coupling = _ring_coupling(3, 0.6)
        parameters = KuramotoParameters(np.zeros(3), coupling)
        np.testing.assert_allclose(
            networked_phase_rule_jacobian(theta, parameters, 0.0),
            networked_kuramoto_jacobian(theta, coupling),
        )

    def test_networked_jacobian_uses_sakaguchi_when_frustrated(self) -> None:
        theta = np.array([0.2, -0.5, 1.1])
        coupling = _ring_coupling(3, 0.6)
        parameters = KuramotoParameters(np.zeros(3), coupling, frustration=0.25)
        np.testing.assert_allclose(
            networked_phase_rule_jacobian(theta, parameters, 0.0),
            sakaguchi_jacobian(theta, coupling, 0.25),
        )

    def test_mean_field_jacobian_rejects_matrix_coupling(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), _ring_coupling(3, 0.5))
        with pytest.raises(ValueError, match="scalar coupling"):
            mean_field_phase_rule_jacobian(np.zeros(3), parameters, 0.0)

    def test_networked_jacobian_rejects_scalar_coupling(self) -> None:
        parameters = KuramotoParameters(np.zeros(3), 0.5)
        with pytest.raises(ValueError, match="coupling matrix"):
            networked_phase_rule_jacobian(np.zeros(3), parameters, 0.0)

    @pytest.mark.parametrize("frustration", [0.0, 0.4])
    def test_mean_field_jacobian_matches_finite_difference(self, frustration: float) -> None:
        theta = np.array([0.3, -0.6, 0.9, 1.4, -0.2])
        parameters = KuramotoParameters(np.zeros(5), 1.1, frustration=frustration)
        np.testing.assert_allclose(
            mean_field_phase_rule_jacobian(theta, parameters, 0.0),
            _finite_difference_jacobian(mean_field_phase_rule, theta, parameters),
            rtol=1e-5,
            atol=1e-7,
        )

    @pytest.mark.parametrize("frustration", [0.0, 0.3])
    def test_networked_jacobian_matches_finite_difference(self, frustration: float) -> None:
        theta = np.array([0.3, -0.6, 0.9, 1.4])
        coupling = _ring_coupling(4, 0.7)
        parameters = KuramotoParameters(np.zeros(4), coupling, frustration=frustration)
        np.testing.assert_allclose(
            networked_phase_rule_jacobian(theta, parameters, 0.0),
            _finite_difference_jacobian(networked_phase_rule, theta, parameters),
            rtol=1e-5,
            atol=1e-7,
        )


# --------------------------------------------------------------------------- #
# KuramotoSystem rule / jacobian accessors
# --------------------------------------------------------------------------- #
class TestSystemJacobianAccessors:
    def test_factory_systems_carry_the_matching_jacobian(self) -> None:
        mean_field = KuramotoSystem.mean_field(np.zeros(3), np.zeros(3), 0.5, dt=0.01)
        assert mean_field.jacobian is mean_field_phase_rule_jacobian
        networked = KuramotoSystem.networked(
            np.zeros(3), np.zeros(3), _ring_coupling(3, 0.4), dt=0.01
        )
        assert networked.jacobian is networked_phase_rule_jacobian

    def test_custom_rule_system_has_no_jacobian(self) -> None:
        system = KuramotoSystem(
            mean_field_phase_rule, np.zeros(3), KuramotoParameters(np.zeros(3), 0.5), dt=0.01
        )
        assert system.jacobian is None

    def test_rule_property_evaluates_at_arbitrary_state(self) -> None:
        system = KuramotoSystem.mean_field(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.5, dt=0.01)
        probe = np.array([0.4, 0.5, 0.6])
        np.testing.assert_allclose(
            system.rule(probe, system.current_parameters, 0.0),
            mean_field_phase_rule(probe, system.current_parameters, 0.0),
        )

    def test_rule_jacobian_at_current_state(self) -> None:
        theta = np.array([0.2, 0.5, 0.9])
        system = KuramotoSystem.mean_field(theta, np.zeros(3), 0.7, dt=0.01)
        np.testing.assert_allclose(system.rule_jacobian(), mean_field_jacobian(theta, 0.7))

    def test_rule_jacobian_requires_a_jacobian(self) -> None:
        system = KuramotoSystem(
            mean_field_phase_rule, np.zeros(3), KuramotoParameters(np.zeros(3), 0.5), dt=0.01
        )
        with pytest.raises(ValueError, match="no analytic Jacobian"):
            system.rule_jacobian()

    def test_explicit_jacobian_argument_is_stored(self) -> None:
        system = KuramotoSystem(
            mean_field_phase_rule,
            np.zeros(3),
            KuramotoParameters(np.zeros(3), 0.5),
            dt=0.01,
            jacobian=mean_field_phase_rule_jacobian,
        )
        assert system.jacobian is mean_field_phase_rule_jacobian


# --------------------------------------------------------------------------- #
# ODE adapters
# --------------------------------------------------------------------------- #
class TestOdeAdapters:
    def test_rhs_matches_the_rule_and_ignores_internal_state(self) -> None:
        system = KuramotoSystem.mean_field(
            np.zeros(4), np.array([0.1, 0.2, -0.1, 0.3]), 0.6, dt=0.01
        )
        rhs = kuramoto_ode_rhs(system)
        probe = np.array([0.4, -0.3, 0.8, 1.2])
        np.testing.assert_allclose(
            rhs(0.0, probe), mean_field_phase_rule(probe, system.current_parameters, 0.0)
        )
        # The RHS reads the probe, not the (zero) internal state.
        assert not np.allclose(rhs(0.0, probe), system.rule_value())

    def test_jacobian_adapter_for_factory_system(self) -> None:
        system = KuramotoSystem.networked(
            np.zeros(3), np.zeros(3), _ring_coupling(3, 0.5), dt=0.01
        )
        jac = kuramoto_ode_jacobian(system)
        assert jac is not None
        probe = np.array([0.2, 0.7, -0.4])
        np.testing.assert_allclose(
            jac(0.0, probe), networked_kuramoto_jacobian(probe, _ring_coupling(3, 0.5))
        )

    def test_jacobian_adapter_none_for_custom_rule(self) -> None:
        system = KuramotoSystem(
            mean_field_phase_rule, np.zeros(3), KuramotoParameters(np.zeros(3), 0.5), dt=0.01
        )
        assert kuramoto_ode_jacobian(system) is None


# --------------------------------------------------------------------------- #
# solve_kuramoto_ivp
# --------------------------------------------------------------------------- #
class TestSolveKuramotoIvp:
    def _system(self) -> KuramotoSystem:
        rng = np.random.default_rng(20260702)
        theta0 = rng.uniform(-np.pi, np.pi, 6)
        omega = rng.normal(0.0, 0.4, 6)
        return KuramotoSystem.networked(theta0, omega, _ring_coupling(6, 0.9), dt=0.005)

    def test_returns_solution_shape_and_diagnostics(self) -> None:
        system = self._system()
        solution = solve_kuramoto_ivp(system, (0.0, 1.0))
        assert isinstance(solution, KuramotoIvpSolution)
        assert solution.success is True
        assert solution.status == 0
        assert solution.phases.shape[1] == system.dimension
        assert solution.times.shape[0] == solution.phases.shape[0]
        assert solution.function_evaluations > 0
        np.testing.assert_array_equal(solution.terminal_phases, solution.phases[-1])

    def test_does_not_mutate_the_system(self) -> None:
        system = self._system()
        before_state = system.current_state
        before_time = system.current_time
        solve_kuramoto_ivp(system, (0.0, 0.5))
        np.testing.assert_array_equal(system.current_state, before_state)
        assert system.current_time == before_time

    def test_t_eval_controls_sample_times(self) -> None:
        system = self._system()
        requested = np.linspace(0.0, 1.0, 11)
        solution = solve_kuramoto_ivp(system, (0.0, 1.0), t_eval=requested)
        np.testing.assert_allclose(solution.times, requested)

    def test_rejects_malformed_t_span(self) -> None:
        with pytest.raises(ValueError, match=r"\(t0, tf\) pair"):
            solve_kuramoto_ivp(self._system(), (0.0, 1.0, 2.0))

    def test_use_jacobian_requires_a_jacobian(self) -> None:
        system = KuramotoSystem(
            mean_field_phase_rule,
            np.zeros(4),
            KuramotoParameters(np.zeros(4), 0.5),
            dt=0.01,
        )
        with pytest.raises(ValueError, match="requires a system with an analytic Jacobian"):
            solve_kuramoto_ivp(system, (0.0, 1.0), use_jacobian=True)

    def test_terminal_phases_is_a_defensive_copy(self) -> None:
        solution = solve_kuramoto_ivp(self._system(), (0.0, 0.3))
        snapshot = solution.terminal_phases
        snapshot[0] = 123.0
        assert solution.phases[-1][0] != 123.0

    def test_matches_the_dedicated_rk4_integrator_at_tight_tolerance(self) -> None:
        rng = np.random.default_rng(7)
        size = 5
        theta0 = rng.uniform(-np.pi, np.pi, size)
        omega = rng.normal(0.0, 0.3, size)
        coupling = _ring_coupling(size, 0.8)
        system = KuramotoSystem.networked(theta0, omega, coupling, dt=0.001)
        t_end = 0.5
        reference = kuramoto_rk4_trajectory(theta0, omega, coupling, 0.001, 500)[-1]
        solution = solve_kuramoto_ivp(system, (0.0, t_end), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(solution.terminal_phases, reference, rtol=0.0, atol=1e-7)

    def test_matches_system_trajectory_at_tight_tolerance(self) -> None:
        theta0 = np.array([0.3, -0.7, 1.1, 0.4, -0.2])
        omega = np.array([0.1, 0.2, -0.1, 0.05, -0.15])
        system = KuramotoSystem.mean_field(theta0, omega, 1.2, dt=0.0005)
        reference = system.trajectory(1000)[-1]
        system.reinit()
        solution = solve_kuramoto_ivp(system, (0.0, 0.5), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(solution.terminal_phases, reference, rtol=0.0, atol=1e-6)

    def test_implicit_method_with_jacobian_agrees_with_explicit(self) -> None:
        rng = np.random.default_rng(3)
        size = 4
        theta0 = rng.uniform(-np.pi, np.pi, size)
        omega = rng.normal(0.0, 0.3, size)
        system = KuramotoSystem.networked(theta0, omega, _ring_coupling(size, 0.7), dt=0.01)
        explicit = solve_kuramoto_ivp(system, (0.0, 1.0), rtol=1e-9, atol=1e-11)
        implicit = solve_kuramoto_ivp(
            system, (0.0, 1.0), method="BDF", rtol=1e-9, atol=1e-11, use_jacobian=True
        )
        assert implicit.success is True
        assert implicit.jacobian_evaluations > 0
        np.testing.assert_allclose(
            implicit.terminal_phases, explicit.terminal_phases, rtol=0.0, atol=1e-6
        )
