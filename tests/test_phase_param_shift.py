# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Parameter Shift
"""Tests for phase/param_shift.py parameter-shift VQE gradients."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control import phase
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase import vqe_with_param_shift
from scpn_quantum_control.phase.param_shift import (
    GenericParameterShiftEvaluationPlan,
    GradientVerificationResult,
    HessianVerificationResult,
    ParamShiftVQEResult,
    multi_frequency_parameter_shift_rule,
    parameter_shift_gradient,
    parameter_shift_hessian,
    plan_generic_parameter_shift_evaluations,
    plan_quantum_gradient_backend,
    value_and_vqe_grad,
    verify_parameter_shift_gradient,
    verify_parameter_shift_hessian,
    verify_vqe_parameter_shift_gradient,
    verify_vqe_parameter_shift_hessian,
)
from scpn_quantum_control.phase.phase_vqe import PhaseVQE


def _finite_difference_gradient(
    objective,
    params: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=float)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def test_phase_param_shift_module_exports_core_gradient() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    grad = parameter_shift_gradient(objective, params, shift=np.pi / 2.0)
    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_generic_parameter_shift_plan_reports_opaque_callable_fallback() -> None:
    params = np.array([0.2, -0.4, 0.6], dtype=float)

    plan = plan_generic_parameter_shift_evaluations(params)
    payload = plan.to_dict()

    assert isinstance(plan, GenericParameterShiftEvaluationPlan)
    assert plan.parameter_count == 3
    assert plan.shift_terms == 1
    assert plan.evaluations == 6
    assert "opaque callable" in plan.fallback_reason
    assert payload["evaluations"] == 6
    assert (
        phase.plan_generic_parameter_shift_evaluations is plan_generic_parameter_shift_evaluations
    )


def test_phase_param_shift_exports_multi_frequency_rule() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    grad = parameter_shift_gradient(objective, np.array([0.4]), rule=rule)

    np.testing.assert_allclose(
        grad,
        np.array([np.cos(0.4) - 0.2 * np.sin(0.8)]),
        atol=1e-12,
    )


def test_phase_param_shift_module_exports_backend_planner() -> None:
    plan = plan_quantum_gradient_backend("statevector_simulator", n_params=2)

    assert plan.supported
    assert plan.method == "parameter_shift"


def test_phase_vqe_parameter_shift_matches_finite_difference() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(-0.3, 0.4, vqe.n_params, dtype=float)

    analytic = vqe.parameter_shift_gradient(params)
    finite_difference = _finite_difference_gradient(vqe._cost, params)

    np.testing.assert_allclose(analytic, finite_difference, atol=1e-5, rtol=1e-5)


def test_parameter_shift_verification_certificate_matches_analytic_reference() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    certificate = verify_parameter_shift_gradient(objective, params)
    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    payload = certificate.to_dict()

    assert isinstance(certificate, GradientVerificationResult)
    assert certificate.passed
    assert certificate.method == "parameter_shift_vs_central_finite_difference"
    assert certificate.parameter_shift_evaluations == 2 * params.size
    assert certificate.finite_difference_evaluations == 2 * params.size
    assert certificate.total_evaluations == 4 * params.size
    assert payload["total_evaluations"] == certificate.total_evaluations
    np.testing.assert_allclose(certificate.analytic_gradient, expected, atol=1e-12)


def test_parameter_shift_hessian_matches_coupled_analytic_reference() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: np.ndarray) -> float:
        return float(
            np.cos(values[0]) + 0.25 * np.sin(values[1]) + 0.1 * np.cos(values[0] - values[1])
        )

    expected = np.array(
        [
            [
                -np.cos(params[0]) - 0.1 * np.cos(params[0] - params[1]),
                0.1 * np.cos(params[0] - params[1]),
            ],
            [
                0.1 * np.cos(params[0] - params[1]),
                -0.25 * np.sin(params[1]) - 0.1 * np.cos(params[0] - params[1]),
            ],
        ],
        dtype=float,
    )

    hessian = parameter_shift_hessian(objective, params)

    np.testing.assert_allclose(hessian, expected, atol=1e-12)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)


def test_parameter_shift_hessian_verification_certificate_matches_reference() -> None:
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: np.ndarray) -> float:
        return float(
            np.cos(values[0]) + 0.25 * np.sin(values[1]) + 0.1 * np.cos(values[0] - values[1])
        )

    certificate = verify_parameter_shift_hessian(objective, params)
    payload = certificate.to_dict()

    assert isinstance(certificate, HessianVerificationResult)
    assert certificate.passed
    assert certificate.method == "parameter_shift_hessian_vs_central_finite_difference"
    assert certificate.parameter_shift_evaluations == 2 * params.size * params.size + 1
    assert certificate.finite_difference_evaluations == 2 * params.size * params.size + 1
    assert certificate.total_evaluations == 2 * (2 * params.size * params.size + 1)
    assert payload["total_evaluations"] == certificate.total_evaluations
    np.testing.assert_allclose(
        certificate.parameter_shift_hessian, certificate.parameter_shift_hessian.T
    )


def test_phase_vqe_gradient_verification_certificate() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(-0.3, 0.4, vqe.n_params, dtype=float)

    certificate = verify_vqe_parameter_shift_gradient(vqe, params)

    assert certificate.passed
    assert certificate.parameters.shape == (vqe.n_params,)
    assert certificate.max_abs_error < 1e-5
    assert certificate.parameter_shift_evaluations == 2 * vqe.n_params
    assert certificate.finite_difference_evaluations == 2 * vqe.n_params


def test_phase_vqe_hessian_verification_certificate() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(-0.2, 0.25, vqe.n_params, dtype=float)

    certificate = verify_vqe_parameter_shift_hessian(
        vqe,
        params,
        absolute_tolerance=5e-4,
        relative_tolerance=5e-4,
    )

    assert certificate.passed
    assert certificate.parameters.shape == (vqe.n_params,)
    assert certificate.parameter_shift_hessian.shape == (vqe.n_params, vqe.n_params)
    np.testing.assert_allclose(
        certificate.parameter_shift_hessian,
        certificate.parameter_shift_hessian.T,
        atol=1e-12,
    )


def test_gradient_verification_rejects_unsafe_finite_difference_inputs() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]))

    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_parameter_shift_gradient(objective, np.array([0.2]), finite_difference_step=0.0)

    with pytest.raises(ValueError, match="absolute_tolerance"):
        verify_parameter_shift_gradient(objective, np.array([0.2]), absolute_tolerance=-1.0)

    def non_finite_objective(values: np.ndarray) -> float:
        is_finite_base = np.isclose(values[0], 0.2, atol=0.0, rtol=0.0)
        is_finite_shift = abs(values[0] - 0.2) > 0.01
        if is_finite_base or is_finite_shift:
            return float(np.cos(values[0]))
        return float("nan")

    with pytest.raises(ValueError, match="finite-difference probes"):
        verify_parameter_shift_gradient(non_finite_objective, np.array([0.2]))


def test_hessian_verification_rejects_unsafe_inputs() -> None:
    def objective(values: np.ndarray) -> float:
        return float(np.cos(values[0]))

    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_parameter_shift_hessian(objective, np.array([0.2]), finite_difference_step=0.0)

    with pytest.raises(ValueError, match="second-order"):
        parameter_shift_hessian(objective, np.array([0.2]), shift=1e-8)

    with pytest.raises(ValueError, match="single-term"):
        parameter_shift_hessian(
            objective,
            np.array([0.2]),
            rule=multi_frequency_parameter_shift_rule([1.0, 2.0]),
        )

    def non_finite_objective(values: np.ndarray) -> float:
        if np.allclose(values, np.array([0.2]), atol=0.0, rtol=0.0):
            return float(np.cos(values[0]))
        return float("nan")

    with pytest.raises(ValueError, match="parameter-shift Hessian"):
        verify_parameter_shift_hessian(non_finite_objective, np.array([0.2]))


def test_phase_vqe_structured_gradient_metadata() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(0.1, 0.4, vqe.n_params, dtype=float)

    result = value_and_vqe_grad(vqe, params)

    assert result.method == "parameter_shift"
    assert result.gradient.shape == (vqe.n_params,)
    assert result.evaluations == 1 + 2 * vqe.n_params
    assert np.isfinite(result.value)
    assert np.all(np.isfinite(result.gradient))


def test_phase_vqe_solve_uses_gradient_aware_optimizer_for_parameter_shift() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    result = vqe.solve(maxiter=12, seed=0, gradient_method="parameter_shift")

    assert result["gradient_method"] == "parameter_shift"
    assert result["optimizer"] == "L-BFGS-B"
    assert result["n_grad_evals"] > 0
    assert np.isfinite(result["ground_energy"])
    assert np.isfinite(result["gradient_norm"])


def test_phase_vqe_rejects_unknown_gradient_method() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    with pytest.raises(ValueError, match="gradient_method"):
        vqe.solve(maxiter=5, gradient_method="finite_difference")


def test_vqe_with_param_shift_tracks_non_increasing_best_energy() -> None:
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]

    result = vqe_with_param_shift(
        K,
        omega,
        ansatz_reps=1,
        initial_params=np.linspace(-0.2, 0.2, 4, dtype=float),
        learning_rate=0.05,
        steps=8,
    )

    assert isinstance(result, ParamShiftVQEResult)
    assert result.best_energy <= result.initial_energy
    assert result.final_params.shape == (4,)
    assert result.best_params.shape == (4,)
    assert result.steps <= 8
    assert result.energy_gap is not None


def test_vqe_with_param_shift_supports_generic_callable_route() -> None:
    def objective(params: np.ndarray) -> float:
        return float(np.cos(params[0]) + np.sin(params[1]))

    result = vqe_with_param_shift(
        objective,
        n_params=2,
        initial_params=np.array([0.25, -0.5], dtype=float),
        learning_rate=0.1,
        steps=4,
    )

    as_dict = result.to_dict()
    assert result.best_energy <= result.initial_energy
    assert as_dict["energy"] == result.best_energy
    assert as_dict["optimal_params"].shape == (2,)
