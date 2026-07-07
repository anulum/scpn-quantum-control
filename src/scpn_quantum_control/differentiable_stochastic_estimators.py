# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stochastic differentiable estimators
"""SPSA, score-function, and shot-budget stochastic gradient helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_batch_helpers import _as_parameter_shift_sample_tensor
from .differentiable_parameter_contracts import (
    Parameter,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    ScoreFunctionGradientResult,
    ScoreFunctionSampleRecord,
    ShotAllocationResult,
    SPSAGradientResult,
    SPSAObjectiveSample,
    SPSAProbeRecord,
)
from .differentiable_stochastic_policy import (
    GradientFailurePolicy,
    gradient_confidence_interval,
)
from .differentiable_transform_helpers import _normalise_parameters


def _as_spsa_sample(value: object, *, shots: int | None) -> SPSAObjectiveSample:
    if isinstance(value, SPSAObjectiveSample):
        if shots is not None and value.variance is None:
            raise ValueError("SPSA finite-shot samples must include variance")
        if shots is not None and value.shots is None:
            return SPSAObjectiveSample(
                value=value.value,
                variance=value.variance,
                shots=shots,
                metadata=value.metadata,
            )
        return value
    if shots is not None:
        raise ValueError("SPSA finite-shot objective must return SPSAObjectiveSample")
    return SPSAObjectiveSample(value=_as_real_scalar("SPSA objective value", value))


def _call_spsa_objective(
    objective: Callable[..., object],
    values: NDArray[np.float64],
    shots: int | None,
) -> SPSAObjectiveSample:
    sample = objective(values.copy()) if shots is None else objective(values.copy(), shots)
    return _as_spsa_sample(sample, shots=shots)


def spsa_gradient_estimate(
    objective: Callable[..., object],
    values: ArrayLike,
    *,
    perturbation_radius: float = 0.1,
    repetitions: int = 1,
    seed: int = 0,
    shots: int | None = None,
    parameters: Sequence[Parameter] | None = None,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> SPSAGradientResult:
    """Estimate a scalar-objective gradient with seeded SPSA perturbations.

    Parameters
    ----------
    objective
        Scalar objective called with a copied parameter vector. When ``shots``
        is provided, the objective must accept ``(values, shots)`` and return an
        ``SPSAObjectiveSample`` carrying variance metadata.
    values
        Parameter vector at which the gradient is estimated.
    perturbation_radius
        Positive SPSA perturbation radius.
    repetitions
        Positive number of independent perturbation pairs to average.
    seed
        Non-negative random seed for deterministic perturbation generation.
    shots
        Optional positive finite-shot count per objective evaluation.
    parameters
        Optional parameter metadata. Frozen parameters are not perturbed and
        receive zero gradient entries.
    confidence_z
        Positive normal-approximation confidence multiplier.
    failure_policy
        Optional fail-closed uncertainty policy for the returned confidence
        interval.

    Returns
    -------
    SPSAGradientResult
        Seeded SPSA gradient estimate with probe records and uncertainty
        metadata.
    """
    parameter_values = _as_parameter_array(values)
    radius = _as_real_scalar("SPSA perturbation_radius", perturbation_radius)
    if radius <= 0.0:
        raise ValueError("SPSA perturbation_radius must be finite and positive")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("SPSA repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("SPSA seed must be a non-negative integer")
    if shots is not None and (isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0):
        raise ValueError("SPSA shots must be a positive integer or None")
    z_value = _as_real_scalar("SPSA confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("SPSA confidence_z must be finite and positive")

    parameter_meta = _normalise_parameters(parameter_values, parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    if not np.any(trainable):
        raise ValueError("SPSA requires at least one trainable parameter")

    rng = np.random.default_rng(seed)
    gradient_estimates = np.zeros((repetitions, parameter_values.size), dtype=np.float64)
    shot_variance = np.zeros(parameter_values.size, dtype=np.float64)
    records: list[SPSAProbeRecord] = []
    total_shots = 0

    for repetition in range(repetitions):
        raw_delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=parameter_values.size)
        perturbation = np.where(trainable, raw_delta, 0.0).astype(np.float64)
        plus_parameters = parameter_values + radius * perturbation
        minus_parameters = parameter_values - radius * perturbation
        plus = _call_spsa_objective(objective, plus_parameters, shots)
        minus = _call_spsa_objective(objective, minus_parameters, shots)
        difference = plus.value - minus.value
        estimate = np.zeros(parameter_values.size, dtype=np.float64)
        for index, is_trainable in enumerate(trainable):
            if not is_trainable:
                continue
            estimate[index] = difference / (2.0 * radius * raw_delta[index])
            if shots is not None:
                if plus.variance is None or minus.variance is None:  # pragma: no cover
                    raise ValueError("SPSA finite-shot samples must include variance")
                if plus.shots is None or minus.shots is None:  # pragma: no cover
                    raise ValueError("SPSA finite-shot samples must include shot counts")
                shot_variance[index] += (
                    plus.variance / float(plus.shots) + minus.variance / float(minus.shots)
                ) / (4.0 * radius * radius)
        gradient_estimates[repetition] = estimate
        if shots is not None:
            total_shots += int(cast(int, plus.shots)) + int(cast(int, minus.shots))
        records.append(
            SPSAProbeRecord(
                repetition=repetition,
                perturbation=perturbation,
                plus_parameters=plus_parameters,
                minus_parameters=minus_parameters,
                plus=plus,
                minus=minus,
                gradient_estimate=estimate,
            )
        )

    gradient = np.mean(gradient_estimates, axis=0)
    estimator_variance = (
        np.var(gradient_estimates, axis=0, ddof=1) / repetitions
        if repetitions > 1
        else np.zeros(parameter_values.size, dtype=np.float64)
    )
    variance = estimator_variance + shot_variance / float(repetitions * repetitions)
    variance = np.where(trainable, variance, 0.0)
    standard_error = np.sqrt(variance)
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    return SPSAGradientResult(
        gradient=gradient,
        standard_error=standard_error,
        covariance=np.diag(variance),
        confidence_radius=z_value * standard_error,
        records=tuple(records),
        perturbation_radius=radius,
        repetitions=repetitions,
        seed=seed,
        confidence_z=z_value,
        method="finite_shot_spsa" if shots is not None else "spsa",
        evaluations=2 * repetitions,
        total_shots=total_shots if shots is not None else None,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=(
            "seeded local SPSA gradient estimator for scalar objectives; "
            "finite-shot uncertainty is propagated only when objective samples "
            "provide variances and shot counts; no hardware execution"
        ),
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


def score_function_gradient_estimate(
    rewards: ArrayLike,
    score_vectors: ArrayLike,
    *,
    baseline: float = 0.0,
    parameters: Sequence[Parameter] | None = None,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> ScoreFunctionGradientResult:
    """Estimate a likelihood-ratio gradient from materialised score samples.

    Parameters
    ----------
    rewards
        One-dimensional finite reward samples.
    score_vectors
        Two-dimensional sample-by-parameter score vectors.
    baseline
        Finite scalar baseline subtracted from rewards before weighting score
        vectors.
    parameters
        Optional parameter metadata. Frozen parameter columns are zeroed in the
        gradient and covariance.
    confidence_z
        Positive normal-approximation confidence multiplier.
    failure_policy
        Optional fail-closed uncertainty policy for the returned confidence
        interval.

    Returns
    -------
    ScoreFunctionGradientResult
        Likelihood-ratio gradient estimate with per-sample records and
        covariance metadata.
    """
    reward_array = _as_parameter_array(rewards)
    scores = _as_real_numeric_array("score_vectors", score_vectors)
    if scores.ndim != 2:
        raise ValueError("score_vectors must be a two-dimensional sample-by-parameter array")
    if reward_array.size < 2:
        raise ValueError("score-function estimator requires at least two samples")
    if scores.shape[0] != reward_array.size:
        raise ValueError(
            "score_vectors row count must match reward count: "
            f"{scores.shape[0]} != {reward_array.size}"
        )
    baseline_value = _as_real_scalar("score-function baseline", baseline)
    z_value = _as_real_scalar("score-function confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("score-function confidence_z must be finite and positive")

    parameter_meta = _normalise_parameters(np.zeros(scores.shape[1], dtype=np.float64), parameters)
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    if not np.any(trainable):
        raise ValueError("score-function estimator requires at least one trainable parameter")

    centred_rewards = reward_array - baseline_value
    weighted_scores = centred_rewards[:, None] * scores
    weighted_scores[:, ~trainable] = 0.0
    gradient = np.mean(weighted_scores, axis=0)
    centred_gradients = weighted_scores - gradient
    covariance = centred_gradients.T @ centred_gradients
    covariance /= float((reward_array.size - 1) * reward_array.size)
    covariance[~trainable, :] = 0.0
    covariance[:, ~trainable] = 0.0
    standard_error = np.sqrt(np.diag(covariance))
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    records = tuple(
        ScoreFunctionSampleRecord(
            index=index,
            reward=float(reward_array[index]),
            centred_reward=float(centred_rewards[index]),
            score=scores[index],
            weighted_score=weighted_scores[index],
        )
        for index in range(reward_array.size)
    )
    return ScoreFunctionGradientResult(
        gradient=gradient,
        standard_error=standard_error,
        covariance=covariance,
        confidence_radius=z_value * standard_error,
        records=records,
        baseline=baseline_value,
        sample_count=reward_array.size,
        confidence_z=z_value,
        method="score_function_likelihood_ratio",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=(
            "materialised likelihood-ratio score-function estimator for "
            "finite scalar rewards and score vectors; no sampler autodiff, "
            "provider callback, or hardware execution"
        ),
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


def allocate_parameter_shift_shots(
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    *,
    target_standard_error: float,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    min_shots: int = 1,
    max_shots_per_evaluation: int | None = None,
) -> ShotAllocationResult:
    """Plan plus/minus shots to meet a target parameter-shift standard error.

    Parameters
    ----------
    plus_variances
        Per-parameter or per-term plus-side measurement variances.
    minus_variances
        Per-parameter or per-term minus-side measurement variances with the
        same shape as ``plus_variances``.
    target_standard_error
        Positive target standard error for each trainable gradient component.
    parameters
        Optional parameter metadata. Frozen parameters retain the minimum shot
        count and report zero predicted standard error.
    rule
        Optional single-term or multi-frequency parameter-shift rule used to
        weight variance contributions.
    min_shots
        Positive lower bound for every planned plus/minus evaluation.
    max_shots_per_evaluation
        Optional cap for each planned plus/minus evaluation.

    Returns
    -------
    ShotAllocationResult
        Shot plan and predicted covariance for the requested target.
    """
    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    term_count = len(terms)
    plus_var = _as_parameter_shift_sample_tensor(
        "plus_variances",
        plus_variances,
        term_count=term_count,
    )
    minus_var = _as_parameter_shift_sample_tensor(
        "minus_variances",
        minus_variances,
        term_count=term_count,
    )
    if minus_var.shape != plus_var.shape:
        raise ValueError("minus_variances shape must match plus_variances shape")
    if np.any(plus_var < 0.0) or np.any(minus_var < 0.0):
        raise ValueError("shot variances must be finite non-negative values")
    target = _as_real_scalar("target_standard_error", target_standard_error)
    if target <= 0.0:
        raise ValueError("target_standard_error must be finite and positive")
    if isinstance(min_shots, bool) or not isinstance(min_shots, int) or min_shots < 1:
        raise ValueError("min_shots must be a positive integer")
    if max_shots_per_evaluation is not None and (
        isinstance(max_shots_per_evaluation, bool)
        or not isinstance(max_shots_per_evaluation, int)
        or max_shots_per_evaluation < min_shots
    ):
        raise ValueError("max_shots_per_evaluation must be an integer >= min_shots")
    parameter_meta = _normalise_parameters(plus_var[0], parameters)
    shot_plan = np.full((term_count, 2, plus_var.shape[1]), float(min_shots), dtype=np.float64)
    variance = np.zeros(plus_var.shape[1], dtype=np.float64)
    target_variance = target**2

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        noises: list[tuple[int, int, float]] = []
        root_sum = 0.0
        for term_index, (_shift, coefficient) in enumerate(terms):
            coefficient_squared = coefficient**2
            plus_noise = coefficient_squared * plus_var[term_index, index]
            minus_noise = coefficient_squared * minus_var[term_index, index]
            noises.append((term_index, 0, float(plus_noise)))
            noises.append((term_index, 1, float(minus_noise)))
            root_sum += float(np.sqrt(plus_noise) + np.sqrt(minus_noise))
        if root_sum > 0.0:
            for term_index, side_index, noise in noises:
                required = np.sqrt(noise) * root_sum / target_variance
                shot_plan[term_index, side_index, index] = max(
                    float(min_shots),
                    float(np.ceil(required)),
                )
        else:
            shot_plan[:, :, index] = float(min_shots)
        if max_shots_per_evaluation is not None:
            shot_plan[:, :, index] = np.minimum(
                shot_plan[:, :, index],
                float(max_shots_per_evaluation),
            )
        for term_index, (_shift, coefficient) in enumerate(terms):
            variance[index] += coefficient**2 * (
                plus_var[term_index, index] / shot_plan[term_index, 0, index]
                + minus_var[term_index, index] / shot_plan[term_index, 1, index]
            )

    standard_error = np.sqrt(variance)
    output_shots = shot_plan[0] if shift_rule.is_single_term else shot_plan
    return ShotAllocationResult(
        shots=output_shots,
        predicted_standard_error=standard_error,
        covariance=np.diag(variance),
        target_standard_error=target,
        total_shots=int(np.sum(output_shots)),
        method="parameter_shift_target_se"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift_target_se",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )
