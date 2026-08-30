# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Parameter Shift
"""Tests for phase/param_shift.py parameter-shift VQE gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

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
    parameter_shift_gradient_with_uncertainty,
    parameter_shift_hessian,
    plan_generic_parameter_shift_evaluations,
    plan_quantum_gradient_backend,
    validate_param_shift_convergence,
    value_and_vqe_grad,
    verify_parameter_shift_gradient,
    verify_parameter_shift_hessian,
    verify_vqe_parameter_shift_gradient,
    verify_vqe_parameter_shift_hessian,
)
from scpn_quantum_control.phase.phase_vqe import PhaseVQE

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


def _finite_difference_gradient(
    objective: ScalarObjective,
    params: FloatArray,
    *,
    step: float = 1e-6,
) -> FloatArray:
    """Estimate a scalar objective gradient with central differences."""
    gradient = np.zeros_like(params, dtype=float)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def test_phase_param_shift_module_exports_core_gradient() -> None:
    """Expose the legacy phase gradient through the public module."""
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: FloatArray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    grad = parameter_shift_gradient(objective, params, shift=np.pi / 2.0)
    expected = np.array([-np.sin(params[0]), 0.25 * np.cos(params[1])], dtype=float)
    np.testing.assert_allclose(grad, expected, atol=1e-12)


def test_generic_parameter_shift_plan_reports_opaque_callable_fallback() -> None:
    """Report independent probes when a scalar callable has no gate metadata."""
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
    """Evaluate a public multi-frequency shift rule through the phase facade."""
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: FloatArray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    grad = parameter_shift_gradient(objective, np.array([0.4]), rule=rule)

    np.testing.assert_allclose(
        grad,
        np.array([np.cos(0.4) - 0.2 * np.sin(0.8)]),
        atol=1e-12,
    )


def test_phase_param_shift_module_exports_backend_planner() -> None:
    """Expose simulator gradient planning through the phase module."""
    plan = plan_quantum_gradient_backend("statevector_simulator", n_params=2)

    assert plan.supported
    assert plan.method == "parameter_shift"


def test_phase_vqe_parameter_shift_matches_finite_difference() -> None:
    """Match the PhaseVQE public gradient against an independent derivative."""
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)
    params = np.linspace(-0.3, 0.4, vqe.n_params, dtype=float)

    analytic = vqe.parameter_shift_gradient(params)
    finite_difference = _finite_difference_gradient(vqe._cost, params)

    np.testing.assert_allclose(analytic, finite_difference, atol=1e-5, rtol=1e-5)


def test_parameter_shift_verification_certificate_matches_analytic_reference() -> None:
    """Record analytic first-order agreement and evaluation counts."""
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: FloatArray) -> float:
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
    """Match the public Hessian against a coupled analytic reference."""
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: FloatArray) -> float:
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
    """Record second-order agreement, symmetry, and evaluation counts."""
    params = np.array([0.2, -0.4], dtype=float)

    def objective(values: FloatArray) -> float:
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
    """Verify the PhaseVQE first-order certificate on a real repository model."""
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
    """Verify the PhaseVQE second-order certificate on a real model."""
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
    """Reject invalid first-order verification tolerances and probes."""

    def objective(values: FloatArray) -> float:
        return float(np.cos(values[0]))

    with pytest.raises(ValueError, match="finite_difference_step"):
        verify_parameter_shift_gradient(objective, np.array([0.2]), finite_difference_step=0.0)

    with pytest.raises(ValueError, match="absolute_tolerance"):
        verify_parameter_shift_gradient(objective, np.array([0.2]), absolute_tolerance=-1.0)

    def non_finite_objective(values: FloatArray) -> float:
        is_finite_base = np.isclose(values[0], 0.2, atol=0.0, rtol=0.0)
        is_finite_shift = abs(values[0] - 0.2) > 0.01
        if is_finite_base or is_finite_shift:
            return float(np.cos(values[0]))
        return float("nan")

    with pytest.raises(ValueError, match="finite-difference probes"):
        verify_parameter_shift_gradient(non_finite_objective, np.array([0.2]))


def test_hessian_verification_rejects_unsafe_inputs() -> None:
    """Reject invalid second-order shifts, rules, and objective values."""

    def objective(values: FloatArray) -> float:
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

    def non_finite_objective(values: FloatArray) -> float:
        if np.allclose(values, np.array([0.2]), atol=0.0, rtol=0.0):
            return float(np.cos(values[0]))
        return float("nan")

    with pytest.raises(ValueError, match="parameter-shift Hessian"):
        verify_parameter_shift_hessian(non_finite_objective, np.array([0.2]))


def test_phase_vqe_structured_gradient_metadata() -> None:
    """Return structured PhaseVQE gradient metadata and finite values."""
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
    """Route PhaseVQE parameter shifts through its gradient-aware optimiser."""
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    result = vqe.solve(maxiter=12, seed=0, gradient_method="parameter_shift")

    assert result["gradient_method"] == "parameter_shift"
    assert result["optimizer"] == "L-BFGS-B"
    assert cast(int, result["n_grad_evals"]) > 0
    assert np.isfinite(cast(float, result["ground_energy"]))
    assert np.isfinite(cast(float, result["gradient_norm"]))


def test_phase_vqe_rejects_unknown_gradient_method() -> None:
    """Reject an unsupported PhaseVQE gradient method at the public boundary."""
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    vqe = PhaseVQE(K, omega, ansatz_reps=1)

    with pytest.raises(ValueError, match="gradient_method"):
        vqe.solve(maxiter=5, gradient_method="finite_difference")


def test_vqe_with_param_shift_tracks_non_increasing_best_energy() -> None:
    """Track a non-increasing best energy for a Kuramoto-XY VQE run."""
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
    """Optimise an opaque scalar callable through the public descent route."""

    def objective(params: FloatArray) -> float:
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
    optimal_params = as_dict["optimal_params"]
    assert isinstance(optimal_params, np.ndarray)
    assert optimal_params.shape == (2,)


def test_param_shift_result_preserves_mapping_and_copy_contracts() -> None:
    """Preserve typed properties and the historical mapping-style facade."""

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]))

    result = vqe_with_param_shift(
        objective,
        n_params=1,
        initial_params=np.array([0.2]),
        steps=1,
    )
    copied = result.optimal_params
    copied[0] = 99.0

    assert result.best_params[0] != 99.0
    assert result["steps"] == result.steps
    assert tuple(key for key, _ in result.items()) == result.keys()
    assert result.values()[1] == result.best_energy


def test_parameter_shift_public_inputs_fail_closed() -> None:
    """Reject malformed vectors, counts, shifts, and parameter metadata."""

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]))

    with pytest.raises(ValueError, match="one-dimensional"):
        plan_generic_parameter_shift_evaluations(np.array([[0.2]]))
    with pytest.raises(ValueError, match="finite values"):
        plan_generic_parameter_shift_evaluations(np.array([np.nan]))
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        vqe_with_param_shift(
            objective,
            n_params=2,
            initial_params=np.array([0.2]),
            steps=1,
        )
    with pytest.raises(ValueError, match="n_params must be positive"):
        vqe_with_param_shift(objective, n_params=0, steps=1)
    with pytest.raises(ValueError, match="must match"):
        vqe_with_param_shift(objective, n_params=1, steps=1, n_iterations=2)
    with pytest.raises(ValueError, match="steps must be positive"):
        vqe_with_param_shift(objective, n_params=1, steps=0)
    with pytest.raises(ValueError, match="parameters must contain 1 entries"):
        plan_generic_parameter_shift_evaluations(np.array([0.2]), parameters=())

    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    with pytest.raises(ValueError, match="must not be overridden"):
        parameter_shift_gradient(objective, np.array([0.2]), shift=0.3, rule=rule)
    with pytest.raises(ValueError, match="finite and positive"):
        parameter_shift_gradient(objective, np.array([0.2]), shift=0.0)
    with pytest.raises(ValueError, match="denominator singular"):
        parameter_shift_gradient(objective, np.array([0.2]), shift=np.pi)

    converged = vqe_with_param_shift(
        lambda params: float(np.sum(params * 0.0)),
        n_params=1,
        initial_params=np.array([0.0]),
    )
    assert converged.converged


def test_finite_shot_parameter_shift_rejects_malformed_tensors() -> None:
    """Reject malformed finite-shot tensors through the public uncertainty API."""
    valid = np.array([0.2, 0.3])
    variances = np.array([0.01, 0.02])

    with pytest.raises(ValueError, match="shape"):
        parameter_shift_gradient_with_uncertainty(
            np.ones((1, 1, 2)), valid, variances, variances, shots=100
        )
    with pytest.raises(ValueError, match="first dimension"):
        parameter_shift_gradient_with_uncertainty(
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.ones((1, 2)),
            shots=100,
            rule=multi_frequency_parameter_shift_rule([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="at least one parameter"):
        parameter_shift_gradient_with_uncertainty(
            np.empty((1, 0)),
            np.empty((1, 0)),
            np.empty((1, 0)),
            np.empty((1, 0)),
            shots=100,
        )
    with pytest.raises(ValueError, match="finite values"):
        parameter_shift_gradient_with_uncertainty(
            np.array([np.nan, 0.3]), valid, variances, variances, shots=100
        )
    with pytest.raises(ValueError, match="positive integer or shot-count array"):
        parameter_shift_gradient_with_uncertainty(valid, valid, variances, variances, shots=True)
    with pytest.raises(ValueError, match="shots must be positive"):
        parameter_shift_gradient_with_uncertainty(valid, valid, variances, variances, shots=0)
    with pytest.raises(ValueError, match="width must match"):
        parameter_shift_gradient_with_uncertainty(
            valid, valid, variances, variances, shots=np.array([100])
        )
    with pytest.raises(ValueError, match="positive integers"):
        parameter_shift_gradient_with_uncertainty(
            valid, valid, variances, variances, shots=np.array([100.5, 100.0])
        )


def test_verification_tolerances_are_required_at_runtime() -> None:
    """Reject absent required tolerances even when an untyped caller supplies them."""

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]))

    with pytest.raises(ValueError, match="must be provided"):
        verify_parameter_shift_gradient(
            objective,
            np.array([0.2]),
            absolute_tolerance=cast(float, None),
        )
    with pytest.raises(ValueError, match="must be provided"):
        verify_parameter_shift_hessian(
            objective,
            np.array([0.2]),
            relative_tolerance=cast(float, None),
        )


def test_convergence_validator_rejects_invalid_energy_histories() -> None:
    """Reject empty and non-finite public convergence histories."""
    params = np.array([0.0])
    empty = ParamShiftVQEResult(
        initial_energy=0.0,
        final_energy=0.0,
        best_energy=0.0,
        final_params=params,
        best_params=params,
        energies=(),
        gradient_norms=(),
        steps=0,
        converged=False,
    )
    non_finite = ParamShiftVQEResult(
        initial_energy=0.0,
        final_energy=float("nan"),
        best_energy=0.0,
        final_params=params,
        best_params=params,
        energies=(0.0, float("nan")),
        gradient_norms=(),
        steps=0,
        converged=False,
    )

    with pytest.raises(ValueError, match="at least one energy"):
        validate_param_shift_convergence(empty)
    with pytest.raises(ValueError, match="finite values"):
        validate_param_shift_convergence(non_finite)


def test_convergence_validator_derives_exact_gap_and_accepts_none_energy_tolerance() -> None:
    """Derive a missing exact gap and preserve the runtime None compatibility path."""
    params = np.array([0.0])
    result = ParamShiftVQEResult(
        initial_energy=1.0,
        final_energy=0.5,
        best_energy=0.5,
        final_params=params,
        best_params=params,
        energies=(1.0, 0.5),
        gradient_norms=(),
        steps=0,
        converged=False,
        exact_energy=0.0,
    )

    diagnostics = validate_param_shift_convergence(
        result,
        energy_tolerance=cast(float, None),
        target_gap=0.5,
    )

    assert diagnostics.exact_gap == 0.5
    assert diagnostics.within_energy_tolerance


def test_vqe_with_param_shift_rejects_invalid_runtime_contracts() -> None:
    """Reject invalid optimiser settings, route arguments, and objective values."""

    def objective(params: FloatArray) -> float:
        return float(np.cos(params[0]))

    with pytest.raises(ValueError, match="learning_rate"):
        vqe_with_param_shift(objective, n_params=1, learning_rate=0.0, steps=1)
    with pytest.raises(ValueError, match="tolerance"):
        vqe_with_param_shift(objective, n_params=1, tolerance=-1.0, steps=1)
    with pytest.raises(ValueError, match="omega must be omitted"):
        vqe_with_param_shift(objective, np.array([1.0]), n_params=1, steps=1)
    with pytest.raises(ValueError, match="n_params is required"):
        vqe_with_param_shift(objective, steps=1)
    with pytest.raises(ValueError, match="omega is required"):
        vqe_with_param_shift(np.eye(2), steps=1)
    with pytest.raises(ValueError, match="finite scalar"):
        vqe_with_param_shift(
            lambda params: float("nan"),
            n_params=1,
            initial_params=np.array([0.0]),
            steps=1,
        )


def test_vqe_with_param_shift_records_backtracking_and_rejection() -> None:
    """Record a reduced accepted step and a fully rejected local-minimum step."""

    def quadratic(params: FloatArray) -> float:
        return float(params[0] ** 2)

    backtracked = vqe_with_param_shift(
        quadratic,
        n_params=1,
        initial_params=np.array([1.0]),
        learning_rate=2.0,
        steps=1,
    )

    def cusp_minimum(params: FloatArray) -> float:
        return float(abs(params[0]) + 0.1 * np.sin(params[0]))

    rejected = vqe_with_param_shift(
        cusp_minimum,
        n_params=1,
        initial_params=np.array([0.0]),
        learning_rate=1.0,
        steps=1,
    )

    assert backtracked.line_search_backtracks == (2,)
    assert backtracked.step_sizes == (0.5,)
    assert rejected.accepted_steps == 0
    assert rejected.rejected_steps == 1
