# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Levenberg-Marquardt residual optimization
"""Levenberg-Marquardt and Gauss-Newton residual optimization helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_custom_derivatives import value_and_custom_jacobian
from .differentiable_finite_difference import VectorObjective, value_and_finite_difference_jacobian
from .differentiable_natural_gradient import natural_gradient
from .differentiable_parameter_contracts import (
    Parameter,
    ParameterBounds,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    GradientResult,
    JacobianResult,
    LevenbergMarquardtDampingUpdate,
    LevenbergMarquardtResult,
    LevenbergMarquardtStep,
    LevenbergMarquardtTrial,
    NaturalGradientResult,
)
from .differentiable_sparse_derivatives import empirical_fisher_metric
from .differentiable_transform_helpers import (
    _as_vector_output,
    _normalise_bounds,
    _project_bounds,
)
from .program_ad_registry import CustomDerivativeRule


def gauss_newton_gradient(
    jacobian: JacobianResult,
    *,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Return the Gauss-Newton-preconditioned least-squares gradient.

    Parameters
    ----------
    jacobian:
        Residual-map Jacobian whose value is the residual vector.
    weights:
        Optional non-negative row weights with one entry per residual.
    damping:
        Non-negative diagonal damping added to the empirical-Fisher metric.
    rcond:
        Relative cutoff passed to the metric solve.

    Returns
    -------
    NaturalGradientResult
        Natural-gradient result whose vector is the trainable Gauss-Newton
        update direction before the descent sign is applied.
    """
    if not isinstance(jacobian, JacobianResult):
        raise ValueError("gauss-newton gradient requires a JacobianResult")
    jacobian_arr = jacobian.jacobian
    residual = jacobian.value
    if weights is None:
        weighted_residual = residual
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        weighted_residual = residual * weight_arr

    loss_value = 0.5 * float(residual @ weighted_residual)
    gradient = jacobian_arr.T @ weighted_residual
    base_gradient = GradientResult(
        value=loss_value,
        gradient=gradient,
        method=f"gauss_newton:{jacobian.method}",
        shift=None,
        coefficient=None,
        evaluations=jacobian.evaluations,
        parameter_names=jacobian.parameter_names,
        trainable=jacobian.trainable,
    )
    metric = empirical_fisher_metric(jacobian, weights=weights, damping=damping)
    return natural_gradient(base_gradient, metric, damping=0.0, rcond=rcond)


def custom_gauss_newton_gradient(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    weights: ArrayLike | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> NaturalGradientResult:
    """Return a Gauss-Newton update from an exact custom residual Jacobian.

    Parameters
    ----------
    rule:
        Custom derivative rule whose value function returns residuals and whose
        Jacobian rule returns the exact residual Jacobian.
    values:
        Real parameter vector.
    parameters:
        Optional parameter metadata.
    weights:
        Optional non-negative residual weights.
    damping:
        Non-negative metric damping.
    rcond:
        Relative cutoff passed to the metric solve.

    Returns
    -------
    NaturalGradientResult
        Trainable Gauss-Newton update direction with exact-Jacobian provenance.
    """
    jacobian_result = value_and_custom_jacobian(rule, values, parameters=parameters)
    return gauss_newton_gradient(
        jacobian_result,
        weights=weights,
        damping=damping,
        rcond=rcond,
    )


@dataclass(frozen=True)
class LevenbergMarquardtOptimizer:
    """Bounded Levenberg-Marquardt optimizer for residual-map objectives.

    Parameters
    ----------
    damping:
        Initial non-negative trust-region damping.
    max_steps:
        Positive maximum number of accepted or rejected LM trials.
    residual_tolerance:
        Non-negative Euclidean residual-norm convergence tolerance.
    step_tolerance:
        Non-negative trainable-step convergence tolerance.
    value_tolerance:
        Optional non-negative actual-reduction convergence tolerance.
    acceptance_threshold:
        Non-negative minimum ratio of actual to predicted reduction.
    decrease_factor:
        Multiplicative damping decrease for high-quality accepted trials.
    increase_factor:
        Multiplicative damping increase for rejected trials.
    min_damping:
        Non-negative lower damping bound.
    max_damping:
        Upper damping bound greater than or equal to ``min_damping``.
    high_quality_ratio:
        Ratio threshold for decreasing damping on accepted trials.
    finite_difference_step:
        Positive residual-Jacobian central-difference step.
    max_step_norm:
        Optional positive L2 cap for trainable LM steps.
    """

    damping: float = 1.0e-3
    max_steps: int = 100
    residual_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-8
    value_tolerance: float | None = None
    acceptance_threshold: float = 1.0e-4
    decrease_factor: float = 1.0 / 3.0
    increase_factor: float = 2.0
    min_damping: float = 1.0e-12
    max_damping: float = 1.0e12
    high_quality_ratio: float = 0.75
    finite_difference_step: float = 1.0e-6
    max_step_norm: float | None = None

    def __post_init__(self) -> None:
        """Validate and canonicalize optimizer controls."""
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        max_steps = int(self.max_steps)
        if max_steps < 1:
            raise ValueError("Levenberg-Marquardt max_steps must be positive")
        residual_tolerance = _as_real_scalar(
            "Levenberg-Marquardt residual_tolerance",
            self.residual_tolerance,
        )
        step_tolerance = _as_real_scalar(
            "Levenberg-Marquardt step_tolerance",
            self.step_tolerance,
        )
        if residual_tolerance < 0.0 or step_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt tolerances must be finite and non-negative")
        value_tolerance = (
            None
            if self.value_tolerance is None
            else _as_real_scalar("Levenberg-Marquardt value_tolerance", self.value_tolerance)
        )
        if value_tolerance is not None and value_tolerance < 0.0:
            raise ValueError("Levenberg-Marquardt value_tolerance must be finite and non-negative")
        acceptance_threshold = _as_real_scalar(
            "Levenberg-Marquardt acceptance_threshold",
            self.acceptance_threshold,
        )
        if acceptance_threshold < 0.0:
            raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
        decrease_factor = _as_real_scalar(
            "Levenberg-Marquardt decrease_factor",
            self.decrease_factor,
        )
        increase_factor = _as_real_scalar(
            "Levenberg-Marquardt increase_factor",
            self.increase_factor,
        )
        min_damping = _as_real_scalar("Levenberg-Marquardt min_damping", self.min_damping)
        max_damping = _as_real_scalar("Levenberg-Marquardt max_damping", self.max_damping)
        high_quality_ratio = _as_real_scalar(
            "Levenberg-Marquardt high_quality_ratio",
            self.high_quality_ratio,
        )
        finite_difference_step = _as_real_scalar(
            "Levenberg-Marquardt finite_difference_step",
            self.finite_difference_step,
        )
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError("decrease_factor must be finite and between 0 and 1")
        if increase_factor <= 1.0:
            raise ValueError("increase_factor must be finite and greater than 1")
        if min_damping < 0.0 or max_damping < min_damping:
            raise ValueError("LM damping bounds must be finite and ordered")
        if high_quality_ratio < 0.0:
            raise ValueError("high_quality_ratio must be finite and non-negative")
        if finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        max_step_norm = (
            None
            if self.max_step_norm is None
            else _as_real_scalar("Levenberg-Marquardt max_step_norm", self.max_step_norm)
        )
        if max_step_norm is not None and max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be finite and positive")
        object.__setattr__(self, "damping", min(max_damping, max(min_damping, damping)))
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "residual_tolerance", residual_tolerance)
        object.__setattr__(self, "step_tolerance", step_tolerance)
        object.__setattr__(self, "value_tolerance", value_tolerance)
        object.__setattr__(self, "acceptance_threshold", acceptance_threshold)
        object.__setattr__(self, "decrease_factor", decrease_factor)
        object.__setattr__(self, "increase_factor", increase_factor)
        object.__setattr__(self, "min_damping", min_damping)
        object.__setattr__(self, "max_damping", max_damping)
        object.__setattr__(self, "high_quality_ratio", high_quality_ratio)
        object.__setattr__(self, "finite_difference_step", finite_difference_step)
        object.__setattr__(self, "max_step_norm", max_step_norm)

    def minimize(
        self,
        objective: VectorObjective,
        initial_values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        bounds: Sequence[ParameterBounds] | None = None,
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
        rcond: float = 1.0e-12,
    ) -> LevenbergMarquardtResult:
        """Minimize a vector residual objective with adaptive bounded LM steps.

        Parameters
        ----------
        objective:
            Residual-map objective evaluated on real parameter vectors.
        initial_values:
            Initial parameter vector before bound projection.
        parameters:
            Optional parameter metadata for finite-difference Jacobian records.
        bounds:
            Optional box or periodic parameter bounds.
        weight_fn:
            Optional robust-weight callback evaluated on each residual vector.
        rcond:
            Relative cutoff passed through to the Gauss-Newton metric solve.

        Returns
        -------
        LevenbergMarquardtResult
            Final and best iterates, residual history, damping history, and
            convergence reason.
        """
        values = _as_parameter_array(initial_values)
        bounds_meta = _normalise_bounds(values, bounds)
        values = _project_bounds(values, bounds_meta)
        damping = self.damping
        jacobian_result = value_and_finite_difference_jacobian(
            objective,
            values,
            parameters=parameters,
            step=self.finite_difference_step,
        )
        weights = self._weights_for(jacobian_result.value, weight_fn)
        current_value = self._weighted_value(jacobian_result.value, weights)
        current_residual = jacobian_result.value
        best_values = values.copy()
        best_value = current_value
        value_history: list[float] = [current_value]
        damping_history: list[float] = [damping]
        accepted_history: list[bool] = []
        reason = "max_steps"
        converged = False

        if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
            return LevenbergMarquardtResult(
                values=values,
                residual=current_residual,
                value_history=tuple(value_history),
                damping_history=tuple(damping_history),
                accepted_history=(),
                steps=0,
                converged=True,
                reason="residual_tolerance",
                best_values=best_values,
                best_value=best_value,
            )

        for _ in range(self.max_steps):
            step_result = levenberg_marquardt_step(
                jacobian_result,
                values,
                weights=weights,
                damping=damping,
                bounds=bounds_meta,
                max_step_norm=self.max_step_norm,
                rcond=rcond,
            )
            trial = evaluate_levenberg_marquardt_step(
                objective,
                step_result,
                weights=weights,
                acceptance_threshold=self.acceptance_threshold,
            )
            update = update_levenberg_marquardt_damping(
                trial,
                decrease_factor=self.decrease_factor,
                increase_factor=self.increase_factor,
                min_damping=self.min_damping,
                max_damping=self.max_damping,
                high_quality_ratio=self.high_quality_ratio,
            )
            accepted_history.append(trial.accepted)
            trainable = np.asarray(jacobian_result.trainable, dtype=bool)
            step_norm = float(np.linalg.norm(step_result.step[trainable], ord=2))
            if trial.accepted:
                values = step_result.candidate_values
                current_residual = trial.candidate_residual
                current_value = trial.candidate_value
                if current_value < best_value:
                    best_value = current_value
                    best_values = values.copy()
                if float(np.linalg.norm(current_residual, ord=2)) <= self.residual_tolerance:
                    reason = "residual_tolerance"
                    converged = True
                elif step_norm <= self.step_tolerance:
                    reason = "step_tolerance"
                    converged = True
                elif (
                    self.value_tolerance is not None
                    and abs(trial.actual_reduction) <= self.value_tolerance
                ):
                    reason = "value_tolerance"
                    converged = True
            damping = update.next_damping
            value_history.append(current_value)
            damping_history.append(damping)
            if converged:
                break
            if trial.accepted:
                jacobian_result = value_and_finite_difference_jacobian(
                    objective,
                    values,
                    parameters=parameters,
                    step=self.finite_difference_step,
                )
                weights = self._weights_for(jacobian_result.value, weight_fn)

        return LevenbergMarquardtResult(
            values=values,
            residual=current_residual,
            value_history=tuple(value_history),
            damping_history=tuple(damping_history),
            accepted_history=tuple(accepted_history),
            steps=len(accepted_history),
            converged=converged,
            reason=reason,
            best_values=best_values,
            best_value=best_value,
        )

    @staticmethod
    def _weighted_value(
        residual: NDArray[np.float64],
        weights: NDArray[np.float64] | None,
    ) -> float:
        """Return the weighted least-squares objective value."""
        if weights is None:
            return 0.5 * float(residual @ residual)
        return 0.5 * float(residual @ (residual * weights))

    @staticmethod
    def _weights_for(
        residual: NDArray[np.float64],
        weight_fn: Callable[[NDArray[np.float64]], ArrayLike] | None,
    ) -> NDArray[np.float64] | None:
        """Return validated robust residual weights for one residual vector."""
        if weight_fn is None:
            return None
        weights = _as_real_numeric_array("LM weights", weight_fn(residual.copy()))
        if weights.ndim != 1 or weights.shape[0] != residual.size:
            raise ValueError("LM weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("LM weights must contain only finite non-negative values")
        return weights


def levenberg_marquardt_step(
    jacobian: JacobianResult,
    values: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-3,
    bounds: Sequence[ParameterBounds] | None = None,
    max_step_norm: float | None = None,
    rcond: float = 1.0e-12,
) -> LevenbergMarquardtStep:
    """Return a bounded Levenberg-Marquardt candidate for residual objectives.

    Parameters
    ----------
    jacobian:
        Residual-map Jacobian result at ``values``.
    values:
        Current real parameter vector.
    weights:
        Optional non-negative residual weights.
    damping:
        Non-negative trust-region damping.
    bounds:
        Optional box or periodic parameter bounds for the candidate.
    max_step_norm:
        Optional positive L2 cap for trainable step components.
    rcond:
        Relative cutoff passed through to the Gauss-Newton metric solve.

    Returns
    -------
    LevenbergMarquardtStep
        Candidate step, projected candidate values, predicted reduction, and
        Gauss-Newton provenance.
    """
    current_values = _as_parameter_array(values)
    if current_values.size != jacobian.jacobian.shape[1]:
        raise ValueError("values length must match Jacobian parameter dimension")
    damping_value = _as_real_scalar("Levenberg-Marquardt damping", damping)
    if damping_value < 0.0:
        raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
    max_step_norm_value = (
        None
        if max_step_norm is None
        else _as_real_scalar("Levenberg-Marquardt max_step_norm", max_step_norm)
    )
    if max_step_norm_value is not None and max_step_norm_value <= 0.0:
        raise ValueError("Levenberg-Marquardt max_step_norm must be finite and positive")

    gauss_newton = gauss_newton_gradient(
        jacobian,
        weights=weights,
        damping=damping_value,
        rcond=rcond,
    )
    step = -gauss_newton.natural_gradient.copy()
    trainable = np.asarray(jacobian.trainable, dtype=bool)
    if max_step_norm_value is not None and np.any(trainable):
        norm = float(np.linalg.norm(step[trainable], ord=2))
        if norm > max_step_norm_value:
            step[trainable] *= max_step_norm_value / norm

    candidate_values = current_values + step
    if bounds is not None:
        candidate_values = _project_bounds(
            candidate_values, _normalise_bounds(current_values, bounds)
        )
        step = candidate_values - current_values

    model_gradient = gauss_newton.base_gradient.gradient
    predicted_reduction = -float(model_gradient @ step + 0.5 * step @ gauss_newton.metric @ step)
    return LevenbergMarquardtStep(
        gauss_newton=gauss_newton,
        step=step,
        candidate_values=candidate_values,
        damping=damping_value,
        predicted_reduction=predicted_reduction,
    )


def custom_levenberg_marquardt_step(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    weights: ArrayLike | None = None,
    damping: float = 1.0e-3,
    bounds: Sequence[ParameterBounds] | None = None,
    max_step_norm: float | None = None,
    rcond: float = 1.0e-12,
) -> LevenbergMarquardtStep:
    """Return an LM candidate using an exact custom residual Jacobian.

    Parameters
    ----------
    rule:
        Custom derivative rule with an exact residual Jacobian.
    values:
        Current real parameter vector.
    parameters:
        Optional parameter metadata.
    weights:
        Optional non-negative residual weights.
    damping:
        Non-negative trust-region damping.
    bounds:
        Optional box or periodic parameter bounds for the candidate.
    max_step_norm:
        Optional positive L2 cap for trainable step components.
    rcond:
        Relative cutoff passed through to the Gauss-Newton metric solve.

    Returns
    -------
    LevenbergMarquardtStep
        Bounded LM candidate with exact-Jacobian provenance.
    """
    jacobian_result = value_and_custom_jacobian(rule, values, parameters=parameters)
    return levenberg_marquardt_step(
        jacobian_result,
        values,
        weights=weights,
        damping=damping,
        bounds=bounds,
        max_step_norm=max_step_norm,
        rcond=rcond,
    )


def evaluate_levenberg_marquardt_step(
    objective: VectorObjective,
    step_result: LevenbergMarquardtStep,
    *,
    weights: ArrayLike | None = None,
    acceptance_threshold: float = 1.0e-4,
) -> LevenbergMarquardtTrial:
    """Evaluate actual residual reduction for a Levenberg-Marquardt candidate.

    Parameters
    ----------
    objective:
        Residual-map objective to evaluate at the candidate point.
    step_result:
        Candidate step returned by ``levenberg_marquardt_step``.
    weights:
        Optional non-negative residual weights.
    acceptance_threshold:
        Non-negative minimum actual/predicted reduction ratio.

    Returns
    -------
    LevenbergMarquardtTrial
        Candidate residual, actual reduction, reduction ratio, and acceptance
        decision.
    """
    threshold = _as_real_scalar("Levenberg-Marquardt acceptance_threshold", acceptance_threshold)
    if threshold < 0.0:
        raise ValueError("Levenberg-Marquardt acceptance_threshold must be non-negative")
    candidate_residual = _as_vector_output(objective(step_result.candidate_values.copy()))
    reference_residual = step_result.gauss_newton.base_gradient.value
    if weights is None:
        candidate_value = 0.5 * float(candidate_residual @ candidate_residual)
    else:
        weight_arr = _as_real_numeric_array("weights", weights)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != candidate_residual.size:
            raise ValueError("weights must be a one-dimensional array matching residual rows")
        if not np.all(np.isfinite(weight_arr)) or np.any(weight_arr < 0.0):
            raise ValueError("weights must contain only finite non-negative values")
        candidate_value = 0.5 * float(candidate_residual @ (candidate_residual * weight_arr))
    actual_reduction = reference_residual - candidate_value
    predicted = step_result.predicted_reduction
    reduction_ratio = actual_reduction / predicted if predicted > 0.0 else 0.0
    accepted = predicted > 0.0 and reduction_ratio >= threshold
    return LevenbergMarquardtTrial(
        step_result=step_result,
        candidate_residual=candidate_residual,
        candidate_value=candidate_value,
        actual_reduction=actual_reduction,
        reduction_ratio=reduction_ratio,
        accepted=accepted,
    )


def update_levenberg_marquardt_damping(
    trial: LevenbergMarquardtTrial,
    *,
    decrease_factor: float = 1.0 / 3.0,
    increase_factor: float = 2.0,
    min_damping: float = 1.0e-12,
    max_damping: float = 1.0e12,
    high_quality_ratio: float = 0.75,
) -> LevenbergMarquardtDampingUpdate:
    """Return a bounded trust-region damping update for an LM trial.

    Parameters
    ----------
    trial:
        Evaluated LM candidate trial.
    decrease_factor:
        Multiplicative damping decrease for high-quality accepted trials.
    increase_factor:
        Multiplicative damping increase for rejected trials.
    min_damping:
        Non-negative lower damping bound.
    max_damping:
        Upper damping bound greater than or equal to ``min_damping``.
    high_quality_ratio:
        Ratio threshold for decreasing damping on accepted trials.

    Returns
    -------
    LevenbergMarquardtDampingUpdate
        Bounded next damping value and policy action.
    """
    if not isinstance(trial, LevenbergMarquardtTrial):
        raise ValueError("damping update requires a LevenbergMarquardtTrial")
    decrease = _as_real_scalar("Levenberg-Marquardt decrease_factor", decrease_factor)
    increase = _as_real_scalar("Levenberg-Marquardt increase_factor", increase_factor)
    min_value = _as_real_scalar("Levenberg-Marquardt min_damping", min_damping)
    max_value = _as_real_scalar("Levenberg-Marquardt max_damping", max_damping)
    high_quality = _as_real_scalar(
        "Levenberg-Marquardt high_quality_ratio",
        high_quality_ratio,
    )
    if not 0.0 < decrease < 1.0:
        raise ValueError("decrease_factor must be finite and between 0 and 1")
    if increase <= 1.0:
        raise ValueError("increase_factor must be finite and greater than 1")
    if min_value < 0.0:
        raise ValueError("min_damping must be finite and non-negative")
    if max_value < min_value:
        raise ValueError("max_damping must be greater than or equal to min_damping")
    if high_quality < 0.0:
        raise ValueError("high_quality_ratio must be finite and non-negative")

    current = trial.step_result.damping
    if not trial.accepted:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * increase)),
            action="reject_increase",
        )
    if trial.reduction_ratio >= high_quality:
        return LevenbergMarquardtDampingUpdate(
            trial=trial,
            next_damping=min(max_value, max(min_value, current * decrease)),
            action="accept_decrease",
        )
    return LevenbergMarquardtDampingUpdate(
        trial=trial,
        next_damping=min(max_value, max(min_value, current)),
        action="accept_keep",
    )


__all__ = [
    "LevenbergMarquardtOptimizer",
    "custom_gauss_newton_gradient",
    "custom_levenberg_marquardt_step",
    "evaluate_levenberg_marquardt_step",
    "gauss_newton_gradient",
    "levenberg_marquardt_step",
    "update_levenberg_marquardt_damping",
]
