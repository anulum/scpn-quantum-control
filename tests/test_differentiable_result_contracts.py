# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable result contract tests
"""Tests for differentiable result containers and primitive metadata contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import differentiable as differentiable_facade
from scpn_quantum_control import differentiable_result_contracts as result_contracts
from scpn_quantum_control.differentiable import (
    ArmijoLineSearchResult,
    CustomDerivativeCheckResult,
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    FixedPointSensitivityResult,
    GradientCheckResult,
    GradientResult,
    HessianResult,
    HVPResult,
    ImplicitSensitivityResult,
    JacobianResult,
    JVPResult,
    NaturalGradientOptimizationResult,
    NaturalGradientResult,
    OptimizationResult,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    ShotAllocationResult,
    SparseMatrixResult,
    StochasticGradientResult,
    VJPResult,
)
from scpn_quantum_control.differentiable_stochastic_policy import (
    GradientFailurePolicy,
    StochasticGradientConfidenceInterval,
)


def _base_gradient() -> GradientResult:
    return GradientResult(
        value=1.0,
        gradient=np.array([1.0, -2.0]),
        method="exact",
        shift=1.0,
        coefficient=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )


def _natural_gradient() -> NaturalGradientResult:
    return NaturalGradientResult(
        base_gradient=_base_gradient(),
        metric=np.eye(2),
        natural_gradient=np.array([1.0, -2.0]),
        damping=0.1,
        condition_number=1.0,
    )


def _levenberg_marquardt_step() -> result_contracts.LevenbergMarquardtStep:
    return result_contracts.LevenbergMarquardtStep(
        gauss_newton=_natural_gradient(),
        step=np.array([-0.25, 0.5]),
        candidate_values=np.array([0.75, 1.5]),
        damping=0.1,
        predicted_reduction=0.25,
    )


def _levenberg_marquardt_trial() -> result_contracts.LevenbergMarquardtTrial:
    return result_contracts.LevenbergMarquardtTrial(
        step_result=_levenberg_marquardt_step(),
        candidate_residual=np.array([0.5, -0.25]),
        candidate_value=0.15625,
        actual_reduction=0.5,
        reduction_ratio=0.75,
        accepted=True,
    )


def _levenberg_marquardt_result(
    *,
    values: NDArray[np.float64] | None = None,
    residual: NDArray[np.float64] | None = None,
    value_history: tuple[float, ...] = (1.0, 0.15625),
    damping_history: tuple[float, ...] = (0.1, 0.05),
    accepted_history: tuple[bool, ...] = (True,),
    steps: int = 1,
    converged: bool = True,
    reason: str = "value_tolerance",
    best_values: NDArray[np.float64] | None = None,
    best_value: float = 0.15625,
) -> result_contracts.LevenbergMarquardtResult:
    return result_contracts.LevenbergMarquardtResult(
        values=np.array([0.75, 1.5]) if values is None else values,
        residual=np.array([0.5, -0.25]) if residual is None else residual,
        value_history=value_history,
        damping_history=damping_history,
        accepted_history=accepted_history,
        steps=steps,
        converged=converged,
        reason=reason,
        best_values=np.array([0.75, 1.5]) if best_values is None else best_values,
        best_value=best_value,
    )


def _parameter_shift_record(
    *,
    parameter_index: int = 0,
    parameter_name: str = "x",
    trainable: bool = True,
    plus_value: float = 1.0,
    minus_value: float = 0.5,
    plus_variance: float = 0.1,
    minus_variance: float = 0.2,
    plus_shots: int = 32,
    minus_shots: int = 32,
    coefficient: float = 0.5,
) -> result_contracts.ParameterShiftSampleRecord:
    """Build a self-consistent finite-shot parameter-shift evidence record."""

    gradient_contribution = coefficient * (plus_value - minus_value)
    variance_contribution = coefficient**2 * (
        plus_variance / float(plus_shots) + minus_variance / float(minus_shots)
    )
    return result_contracts.ParameterShiftSampleRecord(
        term_index=0,
        parameter_index=parameter_index,
        parameter_name=parameter_name,
        trainable=trainable,
        shift=math.pi / 2.0,
        coefficient=coefficient,
        plus_value=plus_value,
        minus_value=minus_value,
        plus_variance=plus_variance,
        minus_variance=minus_variance,
        plus_shots=plus_shots,
        minus_shots=minus_shots,
        gradient_contribution=gradient_contribution if trainable else 0.0,
        variance_contribution=variance_contribution if trainable else 0.0,
    )


def _failure_policy() -> GradientFailurePolicy:
    return GradientFailurePolicy(
        max_standard_error=0.2,
        max_confidence_radius=0.4,
        require_trainable=True,
    )


def _confidence_interval(
    *,
    shape: tuple[int, ...] = (2,),
    status: str = "failed",
    reasons: tuple[str, ...] = ("too_wide",),
) -> StochasticGradientConfidenceInterval:
    return StochasticGradientConfidenceInterval(
        lower=np.zeros(shape),
        upper=np.ones(shape),
        confidence_z=1.96,
        confidence_level=0.95,
        policy=_failure_policy(),
        status=status,
        failure_reasons=reasons,
    )


def _centered_confidence_interval(
    center: NDArray[np.float64],
    radius: NDArray[np.float64],
    *,
    status: str = "failed",
    reasons: tuple[str, ...] = ("too_wide",),
) -> StochasticGradientConfidenceInterval:
    """Build a confidence interval whose bounds are centered on a gradient."""

    return StochasticGradientConfidenceInterval(
        lower=center - radius,
        upper=center + radius,
        confidence_z=1.96,
        confidence_level=0.95,
        policy=_failure_policy(),
        status=status,
        failure_reasons=reasons,
    )


def _spsa_sample() -> result_contracts.SPSAObjectiveSample:
    return result_contracts.SPSAObjectiveSample(value=1.0, variance=0.1, shots=16)


def _spsa_probe() -> result_contracts.SPSAProbeRecord:
    return result_contracts.SPSAProbeRecord(
        repetition=0,
        perturbation=np.array([1.0, -1.0]),
        plus_parameters=np.array([0.2, 0.3]),
        minus_parameters=np.array([0.0, 0.5]),
        plus=_spsa_sample(),
        minus=_spsa_sample(),
        gradient_estimate=np.array([0.5, -0.25]),
    )


def _score_sample(index: int = 0) -> result_contracts.ScoreFunctionSampleRecord:
    return result_contracts.ScoreFunctionSampleRecord(
        index=index,
        reward=1.0,
        centred_reward=0.25,
        score=np.array([1.0, -1.0]),
        weighted_score=np.array([0.25, -0.25]),
    )


def _jvp() -> JVPResult:
    return JVPResult(
        value=np.array([1.0, 2.0]),
        jvp=np.array([0.5, -0.25]),
        tangent=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )


def _vjp() -> VJPResult:
    return VJPResult(
        value=np.array([1.0, 2.0]),
        cotangent=np.array([0.5, -0.25]),
        vjp=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )


def test_result_contract_exports_remain_facade_compatible() -> None:
    """Extracted result contracts should keep stable facade identities."""

    assert (
        differentiable_facade.DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY
        == result_contracts.DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY
    )
    assert (
        differentiable_facade.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
        == result_contracts.FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY
    )
    assert differentiable_facade.GradientResult is result_contracts.GradientResult
    assert (
        differentiable_facade.ParameterShiftSampleRecord
        is result_contracts.ParameterShiftSampleRecord
    )
    assert (
        differentiable_facade.StochasticGradientResult is result_contracts.StochasticGradientResult
    )
    assert differentiable_facade.SPSAObjectiveSample is result_contracts.SPSAObjectiveSample
    assert differentiable_facade.SPSAProbeRecord is result_contracts.SPSAProbeRecord
    assert differentiable_facade.SPSAGradientResult is result_contracts.SPSAGradientResult
    assert (
        differentiable_facade.ScoreFunctionSampleRecord
        is result_contracts.ScoreFunctionSampleRecord
    )
    assert (
        differentiable_facade.ScoreFunctionGradientResult
        is result_contracts.ScoreFunctionGradientResult
    )
    assert differentiable_facade.ShotAllocationResult is result_contracts.ShotAllocationResult
    assert differentiable_facade.OptimizationResult is result_contracts.OptimizationResult
    assert differentiable_facade.ArmijoLineSearchResult is result_contracts.ArmijoLineSearchResult
    assert differentiable_facade.GradientCheckResult is result_contracts.GradientCheckResult
    assert (
        differentiable_facade.CustomDerivativeCheckResult
        is result_contracts.CustomDerivativeCheckResult
    )
    assert differentiable_facade.JacobianResult is result_contracts.JacobianResult
    assert differentiable_facade.JVPResult is result_contracts.JVPResult
    assert differentiable_facade.VJPResult is result_contracts.VJPResult
    assert differentiable_facade.HessianResult is result_contracts.HessianResult
    assert differentiable_facade.SparseMatrixResult is result_contracts.SparseMatrixResult
    assert differentiable_facade.HVPResult is result_contracts.HVPResult
    assert differentiable_facade.NaturalGradientResult is result_contracts.NaturalGradientResult
    assert (
        differentiable_facade.NaturalGradientOptimizationResult
        is result_contracts.NaturalGradientOptimizationResult
    )
    assert differentiable_facade.LevenbergMarquardtStep is result_contracts.LevenbergMarquardtStep
    assert (
        differentiable_facade.LevenbergMarquardtTrial is result_contracts.LevenbergMarquardtTrial
    )
    assert (
        differentiable_facade.LevenbergMarquardtDampingUpdate
        is result_contracts.LevenbergMarquardtDampingUpdate
    )
    assert (
        differentiable_facade.LevenbergMarquardtResult is result_contracts.LevenbergMarquardtResult
    )
    assert (
        differentiable_facade.LeastSquaresCovarianceResult
        is result_contracts.LeastSquaresCovarianceResult
    )
    assert (
        differentiable_facade.FisherVectorProductResult
        is result_contracts.FisherVectorProductResult
    )
    assert (
        differentiable_facade.FisherConjugateGradientResult
        is result_contracts.FisherConjugateGradientResult
    )
    assert differentiable_facade.WeightedGradientResult is result_contracts.WeightedGradientResult
    assert (
        differentiable_facade.ImplicitSensitivityResult
        is result_contracts.ImplicitSensitivityResult
    )
    assert (
        differentiable_facade.FixedPointSensitivityResult
        is result_contracts.FixedPointSensitivityResult
    )


def test_levenberg_marquardt_result_contracts_normalise_extracted_records() -> None:
    """Extracted LM records should preserve numeric normalization and provenance fields."""

    step = _levenberg_marquardt_step()
    trial = _levenberg_marquardt_trial()
    update = result_contracts.LevenbergMarquardtDampingUpdate(
        trial=trial,
        next_damping=0.05,
        action="accept_decrease",
    )
    result = _levenberg_marquardt_result()

    assert np.array_equal(step.step, np.array([-0.25, 0.5]))
    assert step.predicted_reduction == pytest.approx(0.25)
    assert trial.step_result is not step
    assert trial.accepted is True
    assert update.trial is trial
    assert update.next_damping == pytest.approx(0.05)
    assert result.steps == 1
    assert result.converged is True
    assert result.value_history == (1.0, 0.15625)
    assert np.array_equal(result.best_values, np.array([0.75, 1.5]))


def test_levenberg_marquardt_step_contract_rejects_malformed_inputs() -> None:
    """LM step records should fail closed on malformed candidate data."""

    gauss_newton = _natural_gradient()
    with pytest.raises(ValueError, match="one-dimensional"):
        result_contracts.LevenbergMarquardtStep(
            gauss_newton=gauss_newton,
            step=np.array([[1.0, 2.0]]),
            candidate_values=np.array([1.0, 2.0]),
            damping=0.1,
            predicted_reduction=0.1,
        )
    with pytest.raises(ValueError, match="candidate_values shape"):
        result_contracts.LevenbergMarquardtStep(
            gauss_newton=gauss_newton,
            step=np.array([1.0, 2.0]),
            candidate_values=np.array([1.0]),
            damping=0.1,
            predicted_reduction=0.1,
        )
    with pytest.raises(ValueError, match="gradient shape"):
        result_contracts.LevenbergMarquardtStep(
            gauss_newton=gauss_newton,
            step=np.array([1.0]),
            candidate_values=np.array([1.0]),
            damping=0.1,
            predicted_reduction=0.1,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        result_contracts.LevenbergMarquardtStep(
            gauss_newton=gauss_newton,
            step=np.array([1.0, 2.0]),
            candidate_values=np.array([1.0, 2.0]),
            damping=-0.1,
            predicted_reduction=0.1,
        )
    with pytest.raises(ValueError, match="predicted_reduction"):
        result_contracts.LevenbergMarquardtStep(
            gauss_newton=gauss_newton,
            step=np.array([1.0, 2.0]),
            candidate_values=np.array([1.0, 2.0]),
            damping=0.1,
            predicted_reduction=-0.1,
        )


def test_levenberg_marquardt_trial_and_update_contracts_reject_malformed_inputs() -> None:
    """LM trial and damping records should validate record types and scalar controls."""

    step = _levenberg_marquardt_step()
    invalid_step = cast(result_contracts.LevenbergMarquardtStep, object())
    with pytest.raises(ValueError, match="LevenbergMarquardtStep"):
        result_contracts.LevenbergMarquardtTrial(
            step_result=invalid_step,
            candidate_residual=np.array([0.5, -0.25]),
            candidate_value=0.15625,
            actual_reduction=0.5,
            reduction_ratio=0.75,
            accepted=True,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        result_contracts.LevenbergMarquardtTrial(
            step_result=step,
            candidate_residual=np.array([[0.5, -0.25]]),
            candidate_value=0.15625,
            actual_reduction=0.5,
            reduction_ratio=0.75,
            accepted=True,
        )
    with pytest.raises(ValueError, match="accepted flag"):
        result_contracts.LevenbergMarquardtTrial(
            step_result=step,
            candidate_residual=np.array([0.5, -0.25]),
            candidate_value=0.15625,
            actual_reduction=0.5,
            reduction_ratio=0.75,
            accepted=cast(bool, 1),
        )

    trial = _levenberg_marquardt_trial()
    invalid_trial = cast(result_contracts.LevenbergMarquardtTrial, step)
    with pytest.raises(ValueError, match="LevenbergMarquardtTrial"):
        result_contracts.LevenbergMarquardtDampingUpdate(
            trial=invalid_trial,
            next_damping=0.05,
            action="accept_keep",
        )
    with pytest.raises(ValueError, match="next_damping"):
        result_contracts.LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=-0.05,
            action="accept_keep",
        )
    with pytest.raises(ValueError, match="known Levenberg-Marquardt action"):
        result_contracts.LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=0.05,
            action="unknown",
        )


def test_levenberg_marquardt_result_contract_rejects_malformed_inputs() -> None:
    """LM result records should validate convergence traces and best-state metadata."""

    with pytest.raises(ValueError, match="best values"):
        _levenberg_marquardt_result(best_values=np.array([0.75]))
    with pytest.raises(ValueError, match="value history"):
        _levenberg_marquardt_result(value_history=())
    with pytest.raises(ValueError, match="non-negative"):
        _levenberg_marquardt_result(steps=-1)
    with pytest.raises(ValueError, match="damping history"):
        _levenberg_marquardt_result(damping_history=(0.1, -0.05))
    with pytest.raises(ValueError, match="accepted history"):
        _levenberg_marquardt_result(accepted_history=())
    with pytest.raises(ValueError, match="initial damping"):
        _levenberg_marquardt_result(damping_history=(0.1,))
    with pytest.raises(ValueError, match="initial value"):
        _levenberg_marquardt_result(value_history=(1.0,))
    with pytest.raises(ValueError, match="best objective"):
        _levenberg_marquardt_result(best_value=0.5)
    with pytest.raises(ValueError, match="known convergence status"):
        _levenberg_marquardt_result(reason="unknown")
    with pytest.raises(ValueError, match="one-dimensional"):
        _levenberg_marquardt_result(residual=np.array([[0.5, -0.25]]))


def test_differentiable_result_contracts_cover_fail_closed_metadata_boundaries() -> None:
    """Derivative result containers should reject malformed numerical metadata."""

    gradient = GradientResult(
        value=1.0,
        gradient=np.array([1.0, -2.0]),
        method="exact",
        shift=None,
        coefficient=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    other_gradient = GradientResult(
        value=1.1,
        gradient=np.array([1.0, -2.0]),
        method="exact",
        shift=None,
        coefficient=1.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    jvp_result = JVPResult(
        value=np.array([1.0, 2.0]),
        jvp=np.array([0.5, -0.25]),
        tangent=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    vjp_result = VJPResult(
        value=np.array([1.0, 2.0]),
        cotangent=np.array([0.5, -0.25]),
        vjp=np.array([0.1, 0.2]),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    sparse_result = SparseMatrixResult(
        row_indices=np.array([0, 1]),
        column_indices=np.array([1, 0]),
        values=np.array([2.0, 3.0]),
        shape=(2, 2),
        method="coo",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    natural_gradient_result = NaturalGradientResult(
        base_gradient=gradient,
        metric=np.eye(2),
        natural_gradient=np.array([1.0, -2.0]),
        damping=0.0,
        condition_number=1.0,
    )

    valid_cases = [
        StochasticGradientResult(
            value=1.0,
            gradient=np.array([1.0, 2.0]),
            standard_error=np.array([0.1, 0.2]),
            covariance=np.diag([0.01, 0.04]),
            confidence_radius=np.array([0.2, 0.4]),
            shots=np.array([[16.0, 32.0], [16.0, 32.0]]),
            confidence_level=0.95,
            method="shot_noise_parameter_shift",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, False),
        ),
        ShotAllocationResult(
            shots=np.array([[8.0, 10.0], [8.0, 10.0]]),
            predicted_standard_error=np.array([0.25, 0.2]),
            covariance=np.eye(2),
            target_standard_error=0.3,
            total_shots=36,
            method="variance_weighted",
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        OptimizationResult(
            values=np.array([0.0, 1.0]),
            final_gradient=gradient,
            value_history=(3.0, 2.0, 1.0),
            steps=2,
            converged=True,
            reason="gradient_tolerance",
        ),
        ArmijoLineSearchResult(
            values=np.array([0.8, 1.2]),
            value=0.5,
            step_size=0.25,
            direction=np.array([-1.0, 0.5]),
            directional_derivative=-0.75,
            accepted=True,
            evaluations=2,
            value_history=(1.0, 0.5),
            reason="accepted",
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        GradientCheckResult(
            reference=gradient,
            candidate=other_gradient,
            max_abs_error=0.0,
            l2_error=0.0,
            value_delta=0.1,
            tolerance=1.0e-6,
            passed=True,
        ),
        CustomDerivativeCheckResult(
            custom_jvp=jvp_result,
            custom_vjp=vjp_result,
            reference_jvp=jvp_result,
            reference_vjp=vjp_result,
            adjoint_inner_error=0.0,
            jvp_l2_error=0.0,
            vjp_l2_error=0.0,
            tolerance=1.0e-9,
            passed=True,
        ),
        JacobianResult(
            value=np.array([1.0, 2.0]),
            jacobian=np.eye(2),
            method="exact",
            step=0.0,
            evaluations=1,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        HessianResult(
            value=1.0,
            hessian=np.eye(2),
            method="exact",
            step=1.0e-3,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        sparse_result,
        HVPResult(
            value=1.0,
            hvp=np.array([1.0, 2.0]),
            tangent=np.array([0.5, -0.5]),
            method="exact",
            step=1.0e-3,
            evaluations=4,
            parameter_names=("x", "y"),
            trainable=(True, True),
        ),
        natural_gradient_result,
        ImplicitSensitivityResult(
            sensitivity=np.eye(2),
            hessian=np.eye(2),
            cross_derivative=np.eye(2),
            damping=0.0,
            condition_number=1.0,
            method="implicit",
            parameter_names=("x", "y"),
            trainable=(True, True),
            hyperparameter_names=("a", "b"),
        ),
        FixedPointSensitivityResult(
            sensitivity=np.eye(2),
            state_jacobian=0.5 * np.eye(2),
            parameter_jacobian=np.eye(2),
            system_matrix=0.5 * np.eye(2),
            damping=0.0,
            condition_number=1.0,
            method="fixed_point",
            parameter_names=("x", "y"),
            trainable=(True, True),
            hyperparameter_names=("a", "b"),
        ),
    ]
    assert len(valid_cases) == 13
    assert sparse_result.nnz == 2
    np.testing.assert_allclose(sparse_result.to_dense(), np.array([[0.0, 2.0], [3.0, 0.0]]))

    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                confidence_radius=np.array([0.1]),
                shots=np.array([[1.0], [1.0]]),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "standard_error shape",
        ),
        (
            lambda: ShotAllocationResult(
                shots=np.array([[1.5], [2.0]]),
                predicted_standard_error=np.array([0.1]),
                covariance=np.eye(1),
                target_standard_error=0.1,
                total_shots=3,
                method="x",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "positive integer",
        ),
        (
            lambda: OptimizationResult(
                values=np.array([1.0]),
                final_gradient=gradient,
                value_history=(1.0,),
                steps=0,
                converged=True,
                reason="ok",
            ),
            "values length",
        ),
        (
            lambda: ArmijoLineSearchResult(
                values=np.array([1.0]),
                value=1.0,
                step_size=-1.0,
                direction=np.array([1.0]),
                directional_derivative=-1.0,
                accepted=True,
                evaluations=1,
                value_history=(1.0,),
                reason="accepted",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "step_size",
        ),
        (
            lambda: GradientCheckResult(
                reference=gradient,
                candidate=GradientResult(
                    value=1.0,
                    gradient=np.array([1.0]),
                    method="exact",
                    shift=None,
                    coefficient=1.0,
                    evaluations=1,
                    parameter_names=("x",),
                    trainable=(True,),
                ),
                max_abs_error=0.0,
                l2_error=0.0,
                value_delta=0.0,
                tolerance=0.0,
                passed=True,
            ),
            "matching shapes",
        ),
        (
            lambda: GradientResult(
                value=1.0,
                gradient=np.array([1.0, 0.5]),
                method="exact",
                shift=None,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(False, True),
            ),
            "gradient must be zero for non-trainable parameters",
        ),
        (
            lambda: CustomDerivativeCheckResult(
                custom_jvp=cast(Any, object()),
                custom_vjp=vjp_result,
                reference_jvp=jvp_result,
                reference_vjp=vjp_result,
                adjoint_inner_error=0.0,
                jvp_l2_error=0.0,
                vjp_l2_error=0.0,
                tolerance=0.0,
                passed=True,
            ),
            "custom_jvp",
        ),
        (
            lambda: JacobianResult(
                value=np.array([[1.0]]),
                jacobian=np.eye(1),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "one-dimensional",
        ),
        (
            lambda: JacobianResult(
                value=np.array([1.0, 2.0]),
                jacobian=np.array([[1.0, 0.5], [2.0, -0.25]]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(False, True),
            ),
            "jacobian must be zero for non-trainable parameters",
        ),
        (
            lambda: JVPResult(
                value=np.array([1.0]),
                jvp=np.array([1.0, 2.0]),
                tangent=np.array([1.0]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "shape must match",
        ),
        (
            lambda: JVPResult(
                value=np.array([1.0]),
                jvp=np.array([1.0]),
                tangent=np.array([1.0, 0.5]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "JVP tangent must be zero for non-trainable parameters",
        ),
        (
            lambda: VJPResult(
                value=np.array([1.0]),
                cotangent=np.array([1.0, 2.0]),
                vjp=np.array([1.0]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "cotangent shape",
        ),
        (
            lambda: VJPResult(
                value=np.array([1.0]),
                cotangent=np.array([1.0]),
                vjp=np.array([1.0, 0.25]),
                method="exact",
                step=0.0,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "VJP must be zero for non-trainable parameters",
        ),
        (
            lambda: HessianResult(
                value=1.0,
                hessian=np.array([[1.0, 2.0]]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "square",
        ),
        (
            lambda: HessianResult(
                value=1.0,
                hessian=np.array([[1.0, 0.5], [0.5, 0.25]]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "hessian columns must be zero for non-trainable parameters",
        ),
        (
            lambda: SparseMatrixResult(
                row_indices=np.array([0, 0]),
                column_indices=np.array([0, 0]),
                values=np.array([1.0, 2.0]),
                shape=(1, 1),
                method="coo",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "duplicate",
        ),
        (
            lambda: SparseMatrixResult(
                row_indices=np.array([0]),
                column_indices=np.array([1]),
                values=np.array([0.5]),
                shape=(1, 2),
                method="coo",
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "sparse values must be zero for non-trainable parameters",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0]),
                tangent=np.array([1.0, 2.0]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True,),
            ),
            "tangent shape",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0, 0.25]),
                tangent=np.array([1.0, 0.0]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "HVP must be zero for non-trainable parameters",
        ),
        (
            lambda: HVPResult(
                value=1.0,
                hvp=np.array([1.0, 0.0]),
                tangent=np.array([1.0, 0.25]),
                method="exact",
                step=1.0e-3,
                evaluations=1,
                parameter_names=("x", "frozen"),
                trainable=(True, False),
            ),
            "HVP tangent must be zero for non-trainable parameters",
        ),
        (
            lambda: NaturalGradientResult(
                base_gradient=gradient,
                metric=np.array([[1.0, 2.0], [0.0, 1.0]]),
                natural_gradient=np.array([1.0, -2.0]),
                damping=0.0,
                condition_number=1.0,
            ),
            "symmetric",
        ),
        (
            lambda: NaturalGradientOptimizationResult(
                values=np.array([0.0, 1.0]),
                final_gradient=gradient,
                final_natural_gradient=natural_gradient_result,
                value_history=(1.0,),
                gradient_norm_history=(1.0,),
                natural_step_norm_history=(0.5,),
                steps=0,
                converged=False,
                reason="max_steps",
                best_values=np.array([0.0, 1.0]),
                best_value=1.0,
            ),
            "one value per update step",
        ),
        (
            lambda: ImplicitSensitivityResult(
                sensitivity=np.ones((2, 1)),
                hessian=np.eye(2),
                cross_derivative=np.ones((1, 2)),
                damping=0.0,
                condition_number=1.0,
                method="implicit",
                parameter_names=("x", "y"),
                trainable=(True, True),
                hyperparameter_names=("a", "b"),
            ),
            "shape must match",
        ),
        (
            lambda: FixedPointSensitivityResult(
                sensitivity=np.eye(2),
                state_jacobian=np.ones((2, 1)),
                parameter_jacobian=np.eye(2),
                system_matrix=np.eye(2),
                damping=0.0,
                condition_number=1.0,
                method="fixed_point",
                parameter_names=("x", "y"),
                trainable=(True, True),
                hyperparameter_names=("a", "b"),
            ),
            "state_jacobian must be square",
        ),
    )
    for index, (factory, message) in enumerate(invalid_factories):
        try:
            factory()
        except ValueError as exc:
            assert re.search(message, str(exc)), f"{index}: {exc}"
        else:
            pytest.fail(f"{index}: expected ValueError matching {message!r}")


def test_shift_spsa_and_score_function_records_cover_evidence_serialisation() -> None:
    """Stochastic evidence records should validate and serialise bounded metadata."""

    shift_record = _parameter_shift_record()
    spsa_probe = _spsa_probe()
    score_records = (_score_sample(0), _score_sample(1))
    two_parameter_gradient = np.array([0.5, -0.25], dtype=np.float64)
    two_parameter_radius = np.array([0.2, 0.2], dtype=np.float64)
    interval = _centered_confidence_interval(two_parameter_gradient, two_parameter_radius)
    shift_standard_error = np.array(
        [math.sqrt(shift_record.variance_contribution)],
        dtype=np.float64,
    )
    shift_radius = 1.96 * shift_standard_error
    shift_interval = _centered_confidence_interval(
        np.array([shift_record.gradient_contribution], dtype=np.float64),
        shift_radius,
    )

    stochastic = result_contracts.StochasticGradientResult(
        value=1.0,
        gradient=np.array([shift_record.gradient_contribution], dtype=np.float64),
        standard_error=shift_standard_error,
        covariance=np.diag([shift_record.variance_contribution]),
        confidence_radius=shift_radius,
        shots=np.array([[[32.0], [32.0]]], dtype=np.float64),
        confidence_level=0.95,
        method="shot_noise_parameter_shift",
        shift=math.pi / 2.0,
        coefficient=0.5,
        evaluations=2,
        parameter_names=("x",),
        trainable=(True,),
        records=(shift_record,),
        confidence_interval=shift_interval,
        failure_policy_status="failed",
        failure_reasons=("too_wide",),
    )
    spsa = result_contracts.SPSAGradientResult(
        gradient=two_parameter_gradient,
        standard_error=np.array([0.1, 0.1]),
        covariance=np.diag([0.01, 0.01]),
        confidence_radius=two_parameter_radius,
        records=(spsa_probe,),
        perturbation_radius=0.1,
        repetitions=1,
        seed=123,
        confidence_z=1.96,
        method="seeded_spsa",
        evaluations=2,
        total_shots=32,
        parameter_names=("x", "y"),
        trainable=(True, True),
        claim_boundary="finite-shot SPSA simulation evidence",
        hardware_execution=False,
        confidence_interval=interval,
        failure_policy_status="failed",
        failure_reasons=("too_wide",),
    )
    score = result_contracts.ScoreFunctionGradientResult(
        gradient=two_parameter_gradient,
        standard_error=np.array([0.1, 0.1]),
        covariance=np.diag([0.01, 0.01]),
        confidence_radius=two_parameter_radius,
        records=score_records,
        baseline=0.75,
        sample_count=2,
        confidence_z=1.96,
        method="score_function",
        parameter_names=("x", "y"),
        trainable=(True, True),
        claim_boundary="score-function simulation evidence",
        hardware_execution=False,
        confidence_interval=interval,
        failure_policy_status="failed",
        failure_reasons=("too_wide",),
    )

    shift_record_dict = shift_record.to_dict()
    spsa_probe_dict = spsa_probe.to_dict()
    spsa_probe_plus = cast(dict[str, object], spsa_probe_dict["plus"])
    stochastic_dict = stochastic.to_dict()
    stochastic_records = cast(list[dict[str, object]], stochastic_dict["records"])
    spsa_dict = spsa.to_dict()
    spsa_interval = cast(dict[str, object], spsa_dict["confidence_interval"])
    score_dict = score.to_dict()
    score_interval = cast(dict[str, object], score_dict["confidence_interval"])

    assert shift_record_dict["parameter_name"] == "x"
    assert spsa_probe_plus["shots"] == 16
    assert score_records[0].to_dict()["index"] == 0
    assert stochastic_records[0]["plus_shots"] == 32
    assert spsa_interval["status"] == "failed"
    assert score_interval["failure_reasons"] == ["too_wide"]


def test_stochastic_gradient_rejects_moment_and_interval_inconsistency() -> None:
    """Parameter-shift stochastic evidence must bind moments to intervals."""

    gradient = np.array([0.5], dtype=np.float64)
    standard_error = np.array([0.1], dtype=np.float64)
    confidence_radius = np.array([0.2], dtype=np.float64)
    interval = _centered_confidence_interval(
        gradient,
        confidence_radius,
        status="passed",
        reasons=(),
    )

    with pytest.raises(ValueError, match="standard_error must match covariance diagonal"):
        result_contracts.StochasticGradientResult(
            value=1.0,
            gradient=gradient,
            standard_error=standard_error,
            covariance=np.array([[0.04]], dtype=np.float64),
            confidence_radius=confidence_radius,
            shots=np.array([[32.0], [32.0]], dtype=np.float64),
            confidence_level=0.95,
            method="shot_noise_parameter_shift",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("x",),
            trainable=(True,),
            confidence_interval=interval,
            failure_policy_status="passed",
        )

    bad_interval = _centered_confidence_interval(
        gradient,
        np.array([0.3], dtype=np.float64),
        status="passed",
        reasons=(),
    )
    with pytest.raises(ValueError, match="confidence_radius must match confidence_interval"):
        result_contracts.StochasticGradientResult(
            value=1.0,
            gradient=gradient,
            standard_error=standard_error,
            covariance=np.array([[0.01]], dtype=np.float64),
            confidence_radius=confidence_radius,
            shots=np.array([[32.0], [32.0]], dtype=np.float64),
            confidence_level=0.95,
            method="shot_noise_parameter_shift",
            shift=math.pi / 2.0,
            coefficient=0.5,
            evaluations=2,
            parameter_names=("x",),
            trainable=(True,),
            confidence_interval=bad_interval,
            failure_policy_status="passed",
        )


def test_spsa_and_score_results_reject_moment_inconsistency() -> None:
    """Stochastic estimator results must expose self-consistent uncertainty."""

    gradient = np.array([0.5, -0.25], dtype=np.float64)
    standard_error = np.array([0.1, 0.1], dtype=np.float64)
    confidence_radius = np.array([0.2, 0.2], dtype=np.float64)
    interval = _centered_confidence_interval(
        gradient,
        confidence_radius,
        status="passed",
        reasons=(),
    )

    with pytest.raises(ValueError, match="SPSA standard_error must match covariance diagonal"):
        result_contracts.SPSAGradientResult(
            gradient=gradient,
            standard_error=standard_error,
            covariance=np.eye(2),
            confidence_radius=confidence_radius,
            records=(_spsa_probe(),),
            perturbation_radius=0.1,
            repetitions=1,
            seed=123,
            confidence_z=1.96,
            method="seeded_spsa",
            evaluations=2,
            total_shots=32,
            parameter_names=("x", "y"),
            trainable=(True, True),
            claim_boundary="finite-shot SPSA simulation evidence",
            hardware_execution=False,
            confidence_interval=interval,
            failure_policy_status="passed",
        )

    bad_interval = _centered_confidence_interval(
        gradient,
        np.array([0.3, 0.2], dtype=np.float64),
        status="passed",
        reasons=(),
    )
    with pytest.raises(ValueError, match="score-function confidence_radius"):
        result_contracts.ScoreFunctionGradientResult(
            gradient=gradient,
            standard_error=standard_error,
            covariance=np.diag([0.01, 0.01]),
            confidence_radius=confidence_radius,
            records=(_score_sample(0), _score_sample(1)),
            baseline=0.75,
            sample_count=2,
            confidence_z=1.96,
            method="score_function",
            parameter_names=("x", "y"),
            trainable=(True, True),
            claim_boundary="score-function simulation evidence",
            hardware_execution=False,
            confidence_interval=bad_interval,
            failure_policy_status="passed",
        )


def test_result_contracts_reject_stochastic_record_and_interval_mismatches() -> None:
    """Stochastic result records should fail closed on malformed provenance links."""

    interval = _confidence_interval()
    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=True,
                parameter_index=0,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "term_index",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=-1,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "parameter_index",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=0,
                parameter_name="",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "parameter_name",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=0,
                parameter_name="x",
                trainable=cast(bool, "yes"),
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "trainable",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=0,
                parameter_name="x",
                trainable=True,
                shift=0.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "shift",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=0,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=-1.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "variances",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=0,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=0,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "shots",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.4]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                records=(cast(result_contracts.ParameterShiftSampleRecord, object()),),
            ),
            "sample records",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.4]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                records=(
                    result_contracts.ParameterShiftSampleRecord(
                        term_index=0,
                        parameter_index=2,
                        parameter_name="z",
                        trainable=True,
                        shift=1.0,
                        coefficient=1.0,
                        plus_value=1.0,
                        minus_value=0.0,
                        plus_variance=0.0,
                        minus_variance=0.0,
                        plus_shots=1,
                        minus_shots=1,
                        gradient_contribution=1.0,
                        variance_contribution=0.0,
                    ),
                ),
            ),
            "out of range",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.4]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                hardware_execution=True,
            ),
            "hardware execution",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.4]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                confidence_interval=interval,
                failure_policy_status="passed",
                failure_reasons=("too_wide",),
            ),
            "confidence_interval status",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.4]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                failure_policy_status="failed",
            ),
            "requires confidence_interval",
        ),
    )
    for index, (factory, message) in enumerate(invalid_factories):
        try:
            factory()
        except ValueError as exc:
            assert re.search(message, str(exc)), f"{index}: {exc}"
        else:
            pytest.fail(f"{index}: expected ValueError matching {message!r}")


def test_result_contracts_reject_spsa_and_score_function_metadata_mismatches() -> None:
    """SPSA and score-function records should reject malformed uncertainty metadata."""

    interval = _confidence_interval()
    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: result_contracts.SPSAObjectiveSample(value=1.0, variance=-0.1), "variance"),
        (lambda: result_contracts.SPSAObjectiveSample(value=1.0, shots=0), "shots"),
        (
            lambda: result_contracts.SPSAProbeRecord(
                repetition=True,
                perturbation=np.array([1.0]),
                plus_parameters=np.array([1.0]),
                minus_parameters=np.array([1.0]),
                plus=_spsa_sample(),
                minus=_spsa_sample(),
                gradient_estimate=np.array([1.0]),
            ),
            "repetition",
        ),
        (
            lambda: result_contracts.SPSAProbeRecord(
                repetition=0,
                perturbation=np.array([1.0, -1.0]),
                plus_parameters=np.array([1.0]),
                minus_parameters=np.array([1.0]),
                plus=_spsa_sample(),
                minus=_spsa_sample(),
                gradient_estimate=np.array([1.0]),
            ),
            "share parameter shape",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "uncertainty vectors",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "covariance shape",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([-0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "non-negative",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.0,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "perturbation_radius",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=3,
                total_shots=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "evaluations",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=None,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
                confidence_interval=interval,
                failure_policy_status="failed",
                failure_reasons=("mismatch",),
            ),
            "failure_reasons",
        ),
        (
            lambda: result_contracts.ScoreFunctionSampleRecord(
                index=True,
                reward=1.0,
                centred_reward=0.0,
                score=np.array([1.0]),
                weighted_score=np.array([1.0]),
            ),
            "index",
        ),
        (
            lambda: result_contracts.ScoreFunctionSampleRecord(
                index=0,
                reward=1.0,
                centred_reward=0.0,
                score=np.array([1.0, 2.0]),
                weighted_score=np.array([1.0]),
            ),
            "share shape",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "uncertainty vectors",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "covariance shape",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([-0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "non-negative",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0),),
                baseline=0.0,
                sample_count=1,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "sample_count",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=0.0,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "confidence_z",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
                failure_policy_status="failed",
            ),
            "requires interval",
        ),
    )
    for factory, message in invalid_factories:
        with pytest.raises(ValueError, match=message):
            factory()


def test_result_contracts_reject_remaining_stochastic_and_metadata_guards() -> None:
    """Result contracts should cover remaining stochastic and metadata guards."""

    interval = _confidence_interval()
    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=-1,
                parameter_index=0,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "term_index",
        ),
        (
            lambda: result_contracts.ParameterShiftSampleRecord(
                term_index=0,
                parameter_index=True,
                parameter_name="x",
                trainable=True,
                shift=1.0,
                coefficient=1.0,
                plus_value=1.0,
                minus_value=0.0,
                plus_variance=0.0,
                minus_variance=0.0,
                plus_shots=1,
                minus_shots=1,
                gradient_contribution=1.0,
                variance_contribution=0.0,
            ),
            "parameter_index",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "confidence_radius shape",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "covariance shape",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((1, 2, 1)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "shots shape",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([-0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "standard_error",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([-0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "confidence_radius",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.array([[np.inf, 0.0], [0.0, 1.0]]),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "covariance",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=1.0,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "confidence_level",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=0.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "shift",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=0.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "coefficient",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=-1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "evaluations",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True,),
            ),
            "trainable mask",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                confidence_interval=interval,
                failure_policy_status="failed",
                failure_reasons=("different",),
            ),
            "failure_reasons",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((1, 1)),
                predicted_standard_error=np.array([0.1]),
                covariance=np.eye(1),
                target_standard_error=0.1,
                total_shots=1,
                method="x",
                parameter_names=("x",),
                trainable=(True,),
            ),
            "shots must have shape",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "predicted_standard_error",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(1),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "covariance shape",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([-0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "predicted_standard_error",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.array([[np.inf, 0.0], [0.0, 1.0]]),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "covariance",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.0,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "target_standard_error",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=3,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "total_shots",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="",
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "method",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True,),
            ),
            "trainable",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", ""),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.ShotAllocationResult(
                shots=np.ones((2, 2)),
                predicted_standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                target_standard_error=0.1,
                total_shots=4,
                method="x",
                parameter_names=("x", "y"),
                trainable=(True, cast(bool, "yes")),
            ),
            "trainable",
        ),
    )
    for factory, message in invalid_factories:
        with pytest.raises(ValueError, match=message):
            factory()


def test_deterministic_result_contracts_reject_replacement_mismatches() -> None:
    """Deterministic result records should reject malformed replacement fields."""

    gradient = _base_gradient()
    other_gradient = replace(gradient, value=1.1)
    jvp = _jvp()
    vjp = _vjp()
    jacobian = result_contracts.JacobianResult(
        value=np.array([1.0, 2.0]),
        jacobian=np.eye(2),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    hessian = result_contracts.HessianResult(
        value=1.0,
        hessian=np.eye(2),
        method="exact",
        step=1.0e-3,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    sparse = result_contracts.SparseMatrixResult(
        row_indices=np.array([0, 1]),
        column_indices=np.array([0, 1]),
        values=np.array([1.0, 2.0]),
        shape=(2, 2),
        method="coo",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    hvp = result_contracts.HVPResult(
        value=1.0,
        hvp=np.array([1.0, 2.0]),
        tangent=np.array([0.5, -0.5]),
        method="exact",
        step=1.0e-3,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    natural = result_contracts.NaturalGradientResult(
        base_gradient=gradient,
        metric=np.eye(2),
        natural_gradient=np.array([1.0, -2.0]),
        damping=0.0,
        condition_number=1.0,
    )
    natural_opt = result_contracts.NaturalGradientOptimizationResult(
        values=np.array([0.0, 1.0]),
        final_gradient=gradient,
        final_natural_gradient=natural,
        value_history=(1.0, 0.5),
        gradient_norm_history=(1.0, 0.25),
        natural_step_norm_history=(0.5,),
        steps=1,
        converged=True,
        reason="gradient_tolerance",
        best_values=np.array([0.0, 1.0]),
        best_value=0.5,
    )
    implicit = result_contracts.ImplicitSensitivityResult(
        sensitivity=np.eye(2),
        hessian=np.eye(2),
        cross_derivative=np.eye(2),
        damping=0.0,
        condition_number=1.0,
        method="implicit",
        parameter_names=("x", "y"),
        trainable=(True, True),
        hyperparameter_names=("a", "b"),
    )
    fixed = result_contracts.FixedPointSensitivityResult(
        sensitivity=np.eye(2),
        state_jacobian=0.5 * np.eye(2),
        parameter_jacobian=np.eye(2),
        system_matrix=0.5 * np.eye(2),
        damping=0.0,
        condition_number=1.0,
        method="fixed",
        parameter_names=("x", "y"),
        trainable=(True, True),
        hyperparameter_names=("a", "b"),
    )
    opt = result_contracts.OptimizationResult(
        values=np.array([0.0, 1.0]),
        final_gradient=gradient,
        value_history=(1.0, 0.5),
        steps=1,
        converged=True,
        reason="gradient_tolerance",
    )
    line_search = result_contracts.ArmijoLineSearchResult(
        values=np.array([0.0, 1.0]),
        value=0.5,
        step_size=0.25,
        direction=np.array([-1.0, 0.5]),
        directional_derivative=-0.5,
        accepted=True,
        evaluations=1,
        value_history=(1.0, 0.5),
        reason="accepted",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    grad_check = result_contracts.GradientCheckResult(
        reference=gradient,
        candidate=other_gradient,
        max_abs_error=0.0,
        l2_error=0.0,
        value_delta=0.1,
        tolerance=1.0e-6,
        passed=True,
    )
    custom_check = result_contracts.CustomDerivativeCheckResult(
        custom_jvp=jvp,
        custom_vjp=vjp,
        reference_jvp=jvp,
        reference_vjp=vjp,
        adjoint_inner_error=0.0,
        jvp_l2_error=0.0,
        vjp_l2_error=0.0,
        tolerance=1.0e-9,
        passed=True,
    )

    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: replace(gradient, gradient=np.ones((1, 2))), "one-dimensional"),
        (lambda: replace(gradient, method=""), "method"),
        (lambda: replace(gradient, shift=0.0), "shift"),
        (lambda: replace(gradient, evaluations=-1), "evaluations"),
        (lambda: replace(gradient, trainable=(True,)), "trainable mask length"),
        (lambda: replace(gradient, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(opt, value_history=()), "value_history"),
        (lambda: replace(opt, steps=True), "steps"),
        (lambda: replace(opt, converged=cast(bool, "yes")), "converged"),
        (lambda: replace(opt, reason=""), "reason"),
        (lambda: replace(opt, best_values=np.array([1.0])), "best_values"),
        (lambda: replace(opt, best_value=2.0), "best_value"),
        (lambda: replace(line_search, direction=np.array([1.0])), "direction shape"),
        (lambda: replace(line_search, accepted=cast(bool, "yes")), "accepted"),
        (lambda: replace(line_search, evaluations=-1), "evaluations"),
        (lambda: replace(line_search, value_history=()), "value_history"),
        (lambda: replace(line_search, reason="other"), "known status"),
        (lambda: replace(line_search, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(line_search, trainable=(True,)), "trainable"),
        (lambda: replace(grad_check, max_abs_error=-1.0), "max_abs_error"),
        (lambda: replace(grad_check, l2_error=-1.0), "l2_error"),
        (lambda: replace(grad_check, value_delta=-1.0), "value_delta"),
        (lambda: replace(grad_check, tolerance=-1.0), "tolerance"),
        (lambda: replace(grad_check, passed=cast(bool, "yes")), "passed"),
        (lambda: replace(custom_check, custom_vjp=cast(VJPResult, object())), "custom_vjp"),
        (
            lambda: replace(custom_check, reference_jvp=cast(JVPResult, object())),
            "reference_jvp",
        ),
        (
            lambda: replace(custom_check, reference_vjp=cast(VJPResult, object())),
            "reference_vjp",
        ),
        (
            lambda: replace(
                custom_check,
                reference_jvp=replace(jvp, value=np.array([1.0]), jvp=np.array([1.0])),
            ),
            "JVP values",
        ),
        (
            lambda: replace(
                custom_check,
                reference_vjp=replace(vjp, value=np.array([1.0]), cotangent=np.array([1.0])),
            ),
            "VJP values",
        ),
        (lambda: replace(custom_check, adjoint_inner_error=-1.0), "errors"),
        (lambda: replace(custom_check, tolerance=-1.0), "tolerance"),
        (lambda: replace(custom_check, passed=cast(bool, "yes")), "passed"),
        (lambda: replace(jacobian, jacobian=np.ones((3, 2))), "row count"),
        (lambda: replace(jacobian, value=np.array([np.inf, 1.0])), "value"),
        (lambda: replace(jacobian, jacobian=np.array([[np.nan, 0.0], [0.0, 1.0]])), "jacobian"),
        (lambda: replace(jacobian, method=""), "method"),
        (lambda: replace(jacobian, step=-1.0), "step"),
        (lambda: replace(jacobian, evaluations=-1), "evaluations"),
        (lambda: replace(jacobian, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(jacobian, trainable=(True,)), "trainable"),
        (lambda: replace(jvp, value=np.ones((1, 1))), "JVP value"),
        (lambda: replace(jvp, tangent=np.ones((1, 1))), "tangent"),
        (lambda: replace(jvp, jvp=np.array([np.inf, 1.0])), "value and product"),
        (lambda: replace(jvp, tangent=np.array([np.inf, 1.0])), "tangent"),
        (lambda: replace(jvp, method=""), "method"),
        (lambda: replace(jvp, step=-1.0), "step"),
        (lambda: replace(jvp, evaluations=-1), "evaluations"),
        (lambda: replace(jvp, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(vjp, value=np.ones((1, 1))), "VJP value"),
        (lambda: replace(vjp, vjp=np.ones((1, 1))), "one-dimensional"),
        (lambda: replace(vjp, cotangent=np.array([np.inf, 1.0])), "value and cotangent"),
        (lambda: replace(vjp, vjp=np.array([np.inf, 1.0])), "VJP must contain"),
        (lambda: replace(vjp, method=""), "method"),
        (lambda: replace(vjp, step=-1.0), "step"),
        (lambda: replace(vjp, evaluations=-1), "evaluations"),
        (lambda: replace(vjp, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(hessian, hessian=np.array([[np.inf, 0.0], [0.0, 1.0]])), "finite"),
        (lambda: replace(hessian, method=""), "method"),
        (lambda: replace(hessian, step=0.0), "step"),
        (lambda: replace(hessian, evaluations=-1), "evaluations"),
        (lambda: replace(hessian, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(hessian, trainable=(True,)), "trainable"),
        (lambda: replace(hessian, hessian=np.array([[1.0, 2.0], [0.0, 1.0]])), "symmetric"),
        (lambda: replace(sparse, values=np.ones((1, 1))), "one-dimensional"),
        (lambda: replace(sparse, row_indices=np.array([0])), "lengths"),
        (lambda: replace(sparse, shape=(1, 1)), "inside matrix"),
        (lambda: replace(sparse, values=np.array([np.inf, 1.0])), "finite"),
        (lambda: replace(sparse, method=""), "method"),
        (lambda: replace(sparse, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(sparse, trainable=(True,)), "trainable"),
        (lambda: replace(hvp, hvp=np.ones((1, 1))), "one-dimensional"),
        (lambda: replace(hvp, hvp=np.array([np.inf, 1.0])), "contain only finite"),
        (lambda: replace(hvp, method=""), "method"),
        (lambda: replace(hvp, step=0.0), "step"),
        (lambda: replace(hvp, evaluations=-1), "evaluations"),
        (lambda: replace(hvp, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(natural, metric=np.ones((1, 2))), "square"),
        (lambda: replace(natural, metric=np.eye(3)), "dimension"),
        (lambda: replace(natural, natural_gradient=np.array([1.0])), "shape"),
        (lambda: replace(natural, metric=np.array([[np.inf, 0.0], [0.0, 1.0]])), "finite"),
        (lambda: replace(natural, metric=np.array([[1.0, 1.0], [0.0, 1.0]])), "symmetric"),
        (lambda: replace(natural, damping=-1.0), "damping"),
        (lambda: replace(natural, condition_number=0.5), "condition_number"),
        (lambda: replace(natural_opt, best_values=np.array([1.0])), "best_values"),
        (
            lambda: replace(natural_opt, final_gradient=cast(GradientResult, object())),
            "final_gradient",
        ),
        (
            lambda: replace(
                natural_opt,
                final_natural_gradient=cast(result_contracts.NaturalGradientResult, object()),
            ),
            "final_natural_gradient",
        ),
        (lambda: replace(natural_opt, value_history=()), "value_history"),
        (lambda: replace(natural_opt, gradient_norm_history=(-1.0, 0.0)), "gradient_norm"),
        (lambda: replace(natural_opt, natural_step_norm_history=(-1.0,)), "natural_step"),
        (lambda: replace(natural_opt, steps=-1), "steps"),
        (lambda: replace(natural_opt, value_history=(1.0,)), "value_history"),
        (lambda: replace(natural_opt, gradient_norm_history=(1.0,)), "gradient_norm"),
        (lambda: replace(natural_opt, natural_step_norm_history=()), "natural_step"),
        (lambda: replace(natural_opt, reason="other"), "reason"),
        (lambda: replace(natural_opt, best_value=2.0), "best_value"),
        (lambda: replace(implicit, sensitivity=np.ones((2, 2, 1))), "two-dimensional"),
        (lambda: replace(implicit, hessian=np.ones((2, 1))), "square"),
        (lambda: replace(implicit, sensitivity=np.ones((1, 2))), "shape"),
        (lambda: replace(implicit, sensitivity=np.array([[np.inf, 0.0], [0.0, 1.0]])), "finite"),
        (lambda: replace(implicit, hessian=np.array([[1.0, 1.0], [0.0, 1.0]])), "symmetric"),
        (lambda: replace(implicit, damping=-1.0), "damping"),
        (lambda: replace(implicit, condition_number=0.5), "condition_number"),
        (lambda: replace(implicit, method=""), "method"),
        (lambda: replace(implicit, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(implicit, trainable=(True,)), "trainable"),
        (lambda: replace(implicit, hyperparameter_names=("a",)), "hyperparameter_names"),
        (lambda: replace(fixed, sensitivity=np.ones((2, 2, 1))), "two-dimensional"),
        (lambda: replace(fixed, system_matrix=np.eye(3)), "system_matrix"),
        (lambda: replace(fixed, sensitivity=np.ones((1, 2))), "sensitivity shape"),
        (lambda: replace(fixed, sensitivity=np.array([[np.inf, 0.0], [0.0, 1.0]])), "finite"),
        (lambda: replace(fixed, damping=-1.0), "damping"),
        (lambda: replace(fixed, condition_number=0.5), "condition_number"),
        (lambda: replace(fixed, method=""), "method"),
        (lambda: replace(fixed, parameter_names=("x",)), "parameter_names"),
        (lambda: replace(fixed, trainable=(True,)), "trainable"),
        (lambda: replace(fixed, hyperparameter_names=("a",)), "hyperparameter_names"),
    )
    for factory, message in invalid_factories:
        with pytest.raises(ValueError, match=message):
            factory()


def test_result_contracts_reject_name_mask_and_interval_status_edges() -> None:
    """Extracted result contracts should reject name, mask, and interval edge cases."""

    interval = _confidence_interval()
    bad_shape_interval = _confidence_interval(shape=(1,))
    gradient = _base_gradient()
    jvp = _jvp()
    vjp = _vjp()
    jacobian = result_contracts.JacobianResult(
        value=np.array([1.0, 2.0]),
        jacobian=np.eye(2),
        method="exact",
        step=0.0,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    hessian = result_contracts.HessianResult(
        value=1.0,
        hessian=np.eye(2),
        method="exact",
        step=1.0e-3,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    hvp = result_contracts.HVPResult(
        value=1.0,
        hvp=np.array([1.0, 2.0]),
        tangent=np.array([0.5, -0.5]),
        method="exact",
        step=1.0e-3,
        evaluations=1,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    natural = result_contracts.NaturalGradientResult(
        base_gradient=gradient,
        metric=np.eye(2),
        natural_gradient=np.array([1.0, -2.0]),
        damping=0.0,
        condition_number=1.0,
    )
    sparse = result_contracts.SparseMatrixResult(
        row_indices=np.array([0, 1]),
        column_indices=np.array([0, 1]),
        values=np.array([1.0, 2.0]),
        shape=(2, 2),
        method="coo",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    line_search = result_contracts.ArmijoLineSearchResult(
        values=np.array([0.0, 1.0]),
        value=0.5,
        step_size=0.25,
        direction=np.array([-1.0, 0.5]),
        directional_derivative=-0.5,
        accepted=True,
        evaluations=1,
        value_history=(1.0, 0.5),
        reason="accepted",
        parameter_names=("x", "y"),
        trainable=(True, True),
    )
    implicit = result_contracts.ImplicitSensitivityResult(
        sensitivity=np.eye(2),
        hessian=np.eye(2),
        cross_derivative=np.eye(2),
        damping=0.0,
        condition_number=1.0,
        method="implicit",
        parameter_names=("x", "y"),
        trainable=(True, True),
        hyperparameter_names=("a", "b"),
    )
    fixed = result_contracts.FixedPointSensitivityResult(
        sensitivity=np.eye(2),
        state_jacobian=0.5 * np.eye(2),
        parameter_jacobian=np.eye(2),
        system_matrix=0.5 * np.eye(2),
        damping=0.0,
        condition_number=1.0,
        method="fixed",
        parameter_names=("x", "y"),
        trainable=(True, True),
        hyperparameter_names=("a", "b"),
    )

    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.array([[1.5, 1.0], [1.0, 1.0]]),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "positive integer",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", ""),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, cast(bool, "yes")),
            ),
            "trainable",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                records=(replace(_parameter_shift_record(), parameter_name="z"),),
            ),
            "parameter_name mismatch",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(False, True),
                records=(_parameter_shift_record(),),
            ),
            "trainable mismatch",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                confidence_interval=bad_shape_interval,
                failure_policy_status="failed",
                failure_reasons=("too_wide",),
            ),
            "confidence_interval shape",
        ),
        (
            lambda: result_contracts.StochasticGradientResult(
                value=1.0,
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.2, 0.3]),
                shots=np.ones((2, 2)),
                confidence_level=0.95,
                method="x",
                shift=1.0,
                coefficient=1.0,
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
                confidence_interval=interval,
                failure_policy_status="not_evaluated",
                failure_reasons=("too_wide",),
            ),
            "evaluated interval status",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=0,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=0,
                total_shots=None,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "repetitions",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=0,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "total_shots",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=None,
                parameter_names=("x",),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=None,
                parameter_names=("x", "y"),
                trainable=(True,),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "trainable",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=None,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="",
                hardware_execution=False,
            ),
            "claim_boundary",
        ),
        (
            lambda: result_contracts.SPSAGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_spsa_probe(),),
                perturbation_radius=0.1,
                repetitions=1,
                seed=1,
                confidence_z=1.96,
                method="spsa",
                evaluations=2,
                total_shots=None,
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
                confidence_interval=interval,
                failure_policy_status="not_evaluated",
                failure_reasons=("too_wide",),
            ),
            "interval status",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0),),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "records length",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x",),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True,),
                claim_boundary="simulation",
                hardware_execution=False,
            ),
            "trainable",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="",
                hardware_execution=False,
            ),
            "claim_boundary",
        ),
        (
            lambda: result_contracts.ScoreFunctionGradientResult(
                gradient=np.array([1.0, 2.0]),
                standard_error=np.array([0.1, 0.2]),
                covariance=np.eye(2),
                confidence_radius=np.array([0.1, 0.2]),
                records=(_score_sample(0), _score_sample(1)),
                baseline=0.0,
                sample_count=2,
                confidence_z=1.96,
                method="score",
                parameter_names=("x", "y"),
                trainable=(True, True),
                claim_boundary="simulation",
                hardware_execution=False,
                confidence_interval=interval,
                failure_policy_status="not_evaluated",
                failure_reasons=("too_wide",),
            ),
            "interval status",
        ),
        (lambda: replace(line_search, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(line_search, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(jacobian, jacobian=np.ones((1,))), "two-dimensional"),
        (lambda: replace(jacobian, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(jacobian, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(jvp, trainable=(True,)), "trainable"),
        (lambda: replace(jvp, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(jvp, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(vjp, trainable=(True,)), "trainable"),
        (lambda: replace(vjp, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(vjp, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(hessian, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(hessian, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(sparse, shape=(True, 2)), "shape"),
        (lambda: replace(sparse, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(sparse, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(hvp, trainable=(True,)), "trainable"),
        (lambda: replace(hvp, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(hvp, trainable=(True, cast(bool, "yes"))), "trainable"),
        (
            lambda: replace(natural, natural_gradient=np.array([np.inf, 1.0])),
            "natural_gradient",
        ),
        (lambda: replace(implicit, sensitivity=np.ones((1, 2))), "shape"),
        (
            lambda: replace(implicit, cross_derivative=np.array([[np.inf, 0.0], [0.0, 1.0]])),
            "operands",
        ),
        (lambda: replace(implicit, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(implicit, hyperparameter_names=("a", "")), "hyperparameter_names"),
        (lambda: replace(implicit, trainable=(True, cast(bool, "yes"))), "trainable"),
        (lambda: replace(fixed, sensitivity=np.ones((1, 2))), "sensitivity shape"),
        (lambda: replace(fixed, state_jacobian=np.array([[np.inf, 0.0], [0.0, 1.0]])), "operands"),
        (lambda: replace(fixed, parameter_names=("x", "")), "parameter_names"),
        (lambda: replace(fixed, hyperparameter_names=("a", "")), "hyperparameter_names"),
        (lambda: replace(fixed, trainable=(True, cast(bool, "yes"))), "trainable"),
    )
    for index, (factory, message) in enumerate(invalid_factories):
        try:
            factory()
        except ValueError as exc:
            assert re.search(message, str(exc)), f"{index}: {exc}"
        else:
            pytest.fail(f"{index}: expected ValueError matching {message!r}")


def test_fisher_and_weighted_result_contracts_validate_extracted_records() -> None:
    """Extracted Fisher and weighted-gradient records should validate metadata."""

    gradient = _base_gradient()
    covariance = result_contracts.LeastSquaresCovarianceResult(
        covariance=np.eye(2),
        standard_errors=np.array([1.0, 1.0]),
        residual_variance=0.25,
        degrees_of_freedom=3,
        condition_number=1.0,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )
    fisher_product = result_contracts.FisherVectorProductResult(
        value=np.array([1.0, -2.0]),
        tangent=np.array([0.5, 0.0]),
        product=np.array([1.5, 0.0]),
        residual_projection=np.array([0.5, -0.5]),
        damping=0.1,
        method="fisher_vector_product:test",
        evaluations=2,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )
    fisher_cg = result_contracts.FisherConjugateGradientResult(
        solution=np.array([0.25, 0.0]),
        residual_norm_history=(1.0, 0.0),
        iterations=1,
        converged=True,
        tolerance=1.0e-8,
        damping=0.1,
        parameter_names=("x", "y"),
        trainable=(True, False),
    )
    weighted = result_contracts.WeightedGradientResult(
        value=1.0,
        gradient=np.array([0.5, -1.0]),
        components=(gradient,),
        weights=np.array([1.0]),
        method="weighted:test",
        evaluations=gradient.evaluations,
        parameter_names=("x", "y"),
        trainable=(True, True),
    )

    assert covariance.degrees_of_freedom == 3
    assert fisher_product.damping == pytest.approx(0.1)
    assert fisher_cg.residual_norm_history == (1.0, 0.0)
    assert weighted.components == (gradient,)


def test_fisher_and_weighted_result_contracts_fail_closed_after_extraction() -> None:
    """Extracted Fisher and weighted records should reject malformed metadata."""

    gradient = _base_gradient()
    invalid_factories: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.ones((2, 3)),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "square matrix",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(1),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "standard_errors length",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.array([[1.0, 0.0], [0.0, np.inf]]),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "finite values",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.array([[1.0, 0.5], [0.0, 1.0]]),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "symmetric",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(2),
                residual_variance=-0.1,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "residual_variance",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=0,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "degrees_of_freedom",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=0.5,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "condition_number",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names length",
        ),
        (
            lambda: result_contracts.LeastSquaresCovarianceResult(
                covariance=np.eye(2),
                standard_errors=np.ones(2),
                residual_variance=0.0,
                degrees_of_freedom=1,
                condition_number=1.0,
                parameter_names=("x", "y"),
                trainable=(True,),
            ),
            "trainable mask length",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones((1, 2)),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "one-dimensional",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones((1, 2)),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "one-dimensional matches",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(1),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "residual_projection shape",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.array([np.inf, 1.0]),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "value and projection",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.array([np.inf, 1.0]),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "tangent and product",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=-1.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "damping",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "method",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=-1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "evaluations",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names length",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", ""),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.FisherVectorProductResult(
                value=np.ones(2),
                tangent=np.ones(2),
                product=np.ones(2),
                residual_projection=np.ones(2),
                damping=0.0,
                method="fisher",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, cast(bool, "yes")),
            ),
            "trainable mask",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones((1, 2)),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "one-dimensional",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.array([np.inf, 1.0]),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "finite values",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "residual history",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=1,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "initial residual",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(-1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "residual norms",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=-1,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "iterations",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=-1.0,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "tolerance",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=-1.0,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "damping",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names length",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True,),
            ),
            "trainable mask length",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", ""),
                trainable=(True, True),
            ),
            "parameter_names",
        ),
        (
            lambda: result_contracts.FisherConjugateGradientResult(
                solution=np.ones(2),
                residual_norm_history=(1.0,),
                iterations=0,
                converged=False,
                tolerance=1.0e-8,
                damping=0.0,
                parameter_names=("x", "y"),
                trainable=(True, cast(bool, "yes")),
            ),
            "trainable mask",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(),
                weights=np.ones(1),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "components",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.ones(2),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "weights length",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones((1, 2)),
                components=(gradient,),
                weights=np.ones(1),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "one-dimensional",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.array([np.inf, 1.0]),
                components=(gradient,),
                weights=np.ones(1),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "finite values",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.array([np.inf]),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "weights",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.ones(1),
                method="",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "method",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.ones(1),
                method="weighted",
                evaluations=-1,
                parameter_names=("x", "y"),
                trainable=(True, True),
            ),
            "evaluations",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.ones(1),
                method="weighted",
                evaluations=1,
                parameter_names=("x",),
                trainable=(True, True),
            ),
            "parameter_names length",
        ),
        (
            lambda: result_contracts.WeightedGradientResult(
                value=1.0,
                gradient=np.ones(2),
                components=(gradient,),
                weights=np.ones(1),
                method="weighted",
                evaluations=1,
                parameter_names=("x", "y"),
                trainable=(True,),
            ),
            "trainable mask length",
        ),
    )
    for factory, message in invalid_factories:
        with pytest.raises(ValueError, match=message):
            factory()


def test_primitive_identity_and_contracts_fail_closed_on_malformed_metadata() -> None:
    """Primitive registry contracts should preserve typed identity and metadata safety."""

    def batching_rule(
        _function: Callable[..., object],
        values: tuple[object, ...],
        in_axes: tuple[int | None, ...],
        _axis: int,
    ) -> object:
        return values, in_axes

    rule = CustomDerivativeRule(
        name="identity_contract_rule",
        value_fn=lambda values: np.asarray(values, dtype=np.float64),
        jvp_rule=lambda _values, tangent: np.asarray(tangent, dtype=np.float64),
        parameter_names=("theta",),
        trainable=(True,),
    )
    identity = PrimitiveIdentity("quantum", "rx", "2")
    assert identity.key == "quantum:rx@2"
    assert PrimitiveIdentity.parse(identity) is identity
    assert PrimitiveIdentity.parse("quantum:rx@2") == identity
    assert PrimitiveIdentity.parse("quantum:rx") == PrimitiveIdentity("quantum", "rx", "1")

    transform = PrimitiveTransformRule(
        identity=identity,
        derivative_rule=rule,
        batching_rule=batching_rule,
        lowering_rule=lambda lowered_rule: {"name": lowered_rule.name},
        lowering_metadata={
            "mlir_op": "scpn_diff.rx",
            "llvm": "blocked",
            "nondifferentiable_boundary": "declared_test_boundary",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args[:1],
        nondifferentiable_policy="none",
        effect="pure",
    )
    contract = PrimitiveContract.from_transform(transform)
    assert contract.identity == identity
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.rx"

    registry = CustomDerivativeRegistry()
    registry.register_transform(transform)
    assert registry.require_complete_contract(identity) == contract
    assert registry.require_effect(identity) == "pure"
    assert registry.require_nondifferentiable_policy(identity) == "none"
    assert registry.require_shape_rule(identity)((np.array([1.0]),)) == (1,)
    assert registry.require_dtype_rule(identity)((np.array([1.0]),)) == "float64"
    assert registry.require_static_argument_rule(identity)(("theta", "phi")) == ("theta",)

    malformed_cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: PrimitiveIdentity("bad namespace", "rx"), "whitespace"),
        (lambda: PrimitiveIdentity.parse("missing_namespace"), "namespace:name"),
        (
            lambda: CustomDerivativeRule(
                name="",
                value_fn=lambda values: values,
                jvp_rule=lambda _values, tangent: tangent,
            ),
            "name",
        ),
        (
            lambda: PrimitiveTransformRule(identity=cast(Any, "bad"), derivative_rule=rule),
            "PrimitiveIdentity",
        ),
        (
            lambda: PrimitiveTransformRule(
                identity=identity,
                derivative_rule=rule,
                lowering_metadata={"mlir_op": ""},
            ),
            "metadata values",
        ),
        (
            lambda: PrimitiveContract(
                identity=identity,
                derivative_rule=rule,
                batching_rule=None,
                lowering_rule=None,
                lowering_metadata={"": "bad"},
                shape_rule=None,
                dtype_rule=None,
                static_argument_rule=None,
                nondifferentiable_policy="none",
                effect="pure",
            ),
            "metadata keys",
        ),
        (lambda: CustomDerivativeRegistry().require_complete_contract(identity), "no primitive"),
    )
    for factory, message in malformed_cases:
        with pytest.raises(ValueError, match=message):
            factory()
