# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable parameter shift module
# scpn-quantum-control -- parameter-shift differentiable transforms
"""Parameter-shift transforms for scalar differentiable objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonical_encoding import canonical_digest
from .differentiable_batch_helpers import _as_parameter_shift_sample_tensor
from .differentiable_parameter_contracts import (
    Parameter,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_scalar,
    multi_frequency_parameter_shift_rule,
)
from .differentiable_result_contracts import (
    FiniteShotSampleProvenance,
    GradientResult,
    ParameterShiftSampleRecord,
    StochasticGradientResult,
)
from .differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GradientFailurePolicy,
    gradient_confidence_interval,
)
from .differentiable_transform_helpers import _as_scalar, _normalise_parameters

ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]


def parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return the parameter-shift gradient of a scalar objective.

    Parameters
    ----------
    objective:
        Scalar-valued objective evaluated on a real parameter vector.
    values:
        Initial real parameter values.
    parameters:
        Optional metadata controlling names and trainable masks.
    rule:
        Optional single- or multi-frequency parameter-shift rule.

    Returns
    -------
    numpy.ndarray
        Real gradient vector with frozen parameters set to zero.

    """
    result = value_and_parameter_shift_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
    )
    return result.gradient


def batch_parameter_shift_gradient(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> NDArray[np.float64]:
    """Return stacked parameter-shift gradients for scalar objectives.

    Parameters
    ----------
    objectives:
        Non-empty sequence of scalar-valued objectives.
    values:
        Initial real parameter values shared by every objective.
    parameters:
        Optional metadata controlling names and trainable masks.
    rule:
        Optional single- or multi-frequency parameter-shift rule.

    Returns
    -------
    numpy.ndarray
        Matrix whose rows are objective gradients.

    """
    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    rows = [
        parameter_shift_gradient(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    ]
    return np.vstack(rows)


def batch_value_and_parameter_shift_grad(
    objectives: Sequence[ScalarObjective],
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> tuple[GradientResult, ...]:
    """Return full parameter-shift results for scalar objectives.

    Parameters
    ----------
    objectives:
        Non-empty sequence of scalar-valued objectives.
    values:
        Initial real parameter values shared by every objective.
    parameters:
        Optional metadata controlling names and trainable masks.
    rule:
        Optional single- or multi-frequency parameter-shift rule.

    Returns
    -------
    tuple[GradientResult, ...]
        Per-objective value, gradient, and provenance records.

    """
    if not objectives:
        raise ValueError("objectives must contain at least one scalar objective")
    return tuple(
        value_and_parameter_shift_grad(
            objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        for objective in objectives
    )


def value_and_parameter_shift_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> GradientResult:
    """Evaluate a scalar objective and its native parameter-shift gradient.

    Parameters
    ----------
    objective:
        Scalar-valued objective evaluated on real parameter probes.
    values:
        Initial real parameter values.
    parameters:
        Optional metadata controlling names and trainable masks.
    rule:
        Optional single- or multi-frequency parameter-shift rule.

    Returns
    -------
    GradientResult
        Objective value, gradient, evaluation count, and parameter metadata.

    """
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1

    for index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        for shift, coefficient in terms:
            plus = parameter_values.copy()
            minus = parameter_values.copy()
            plus[index] += shift
            minus[index] -= shift
            plus_value = _as_scalar(objective(plus))
            minus_value = _as_scalar(objective(minus))
            evaluations += 2
            gradient[index] += coefficient * (plus_value - minus_value)

    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="parameter_shift"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift",
        shift=shift_rule.shift if shift_rule.is_single_term else None,
        coefficient=shift_rule.coefficient if shift_rule.is_single_term else None,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def parameter_shift_gradient_with_uncertainty(
    plus_values: ArrayLike,
    minus_values: ArrayLike,
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    plus_shots: ArrayLike,
    minus_shots: ArrayLike | None = None,
    *,
    sample_provenance: Mapping[str, object] | FiniteShotSampleProvenance | None = None,
    value: float = 0.0,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> StochasticGradientResult:
    """Propagate independent shot noise through parameter-shift gradients.

    Parameters
    ----------
    plus_values, minus_values:
        Shifted objective estimates for every term and parameter.
    plus_variances, minus_variances:
        Per-estimate finite-shot variances.
    plus_shots, minus_shots:
        Positive integer shot counts. When ``minus_shots`` is omitted the plus
        shot counts are reused.
    sample_provenance:
        Source metadata for the materialised plus/minus finite-shot tensors.
        The mapping or record must include ``sample_seed``, ``shot_batch_id``,
        and ``source_class``.
    value:
        Objective value associated with the gradient estimate.
    parameters:
        Optional metadata controlling names and trainable masks.
    rule:
        Optional single- or multi-frequency parameter-shift rule.
    confidence_level:
        Confidence mass associated with the returned interval.
    confidence_z:
        Positive normal-approximation multiplier for interval radii.
    failure_policy:
        Optional policy that classifies uncertainty thresholds.

    Returns
    -------
    StochasticGradientResult
        Gradient, covariance, shot provenance, confidence interval, and
        diagnostic-only claim boundary metadata.

    """
    shift_rule = rule or ParameterShiftRule()
    terms = shift_rule.terms
    term_count = len(terms)
    plus = _as_parameter_shift_sample_tensor(
        "plus_values",
        plus_values,
        term_count=term_count,
    )
    minus = _as_parameter_shift_sample_tensor(
        "minus_values",
        minus_values,
        term_count=term_count,
    )
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
    plus_count = _as_parameter_shift_sample_tensor(
        "plus_shots",
        plus_shots,
        term_count=term_count,
    )
    minus_count = (
        plus_count.copy()
        if minus_shots is None
        else _as_parameter_shift_sample_tensor(
            "minus_shots",
            minus_shots,
            term_count=term_count,
        )
    )
    if minus.shape != plus.shape:
        raise ValueError("minus_values shape must match plus_values shape")
    if plus_var.shape != plus.shape or minus_var.shape != plus.shape:
        raise ValueError("variance shapes must match plus_values shape")
    if plus_count.shape != plus.shape or minus_count.shape != plus.shape:
        raise ValueError("shot-count shapes must match plus_values shape")
    if np.any(plus_var < 0.0) or np.any(minus_var < 0.0):
        raise ValueError("shot variances must be finite non-negative values")
    if (
        not np.all(plus_count > 0.0)
        or not np.all(minus_count > 0.0)
        or not np.allclose(plus_count, np.round(plus_count))
        or not np.allclose(minus_count, np.round(minus_count))
    ):
        raise ValueError("shot counts must contain positive integers")
    confidence = _as_real_scalar("confidence_level", confidence_level)
    z_value = _as_real_scalar("confidence_z", confidence_z)
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if z_value <= 0.0:
        raise ValueError("confidence_z must be finite and positive")
    provenance = _as_sample_provenance(sample_provenance)

    parameter_meta = _normalise_parameters(plus[0], parameters)
    gradient = np.zeros(plus.shape[1], dtype=np.float64)
    variance = np.zeros(plus.shape[1], dtype=np.float64)
    records: list[ParameterShiftSampleRecord] = []
    for index, parameter in enumerate(parameter_meta):
        for term_index, (_shift, coefficient) in enumerate(terms):
            gradient_contribution = coefficient * (
                plus[term_index, index] - minus[term_index, index]
            )
            variance_contribution = coefficient**2 * (
                plus_var[term_index, index] / plus_count[term_index, index]
                + minus_var[term_index, index] / minus_count[term_index, index]
            )
            if parameter.trainable:
                gradient[index] += gradient_contribution
                variance[index] += variance_contribution
            records.append(
                ParameterShiftSampleRecord(
                    term_index=term_index,
                    parameter_index=index,
                    parameter_name=parameter.name,
                    trainable=parameter.trainable,
                    shift=_shift,
                    coefficient=coefficient,
                    plus_value=float(plus[term_index, index]),
                    minus_value=float(minus[term_index, index]),
                    plus_variance=float(plus_var[term_index, index]),
                    minus_variance=float(minus_var[term_index, index]),
                    plus_shots=int(plus_count[term_index, index]),
                    minus_shots=int(minus_count[term_index, index]),
                    sample_seed=provenance.sample_seed,
                    shot_batch_id=provenance.shot_batch_id,
                    source_class=provenance.source_class,
                    gradient_contribution=float(
                        gradient_contribution if parameter.trainable else 0.0
                    ),
                    variance_contribution=float(
                        variance_contribution if parameter.trainable else 0.0
                    ),
                )
            )
    standard_error = np.sqrt(variance)
    covariance = np.diag(variance)
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        confidence_level=confidence,
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        failure_policy=failure_policy,
    )
    shots = (
        np.vstack([plus_count[0], minus_count[0]])
        if shift_rule.is_single_term
        else np.stack([plus_count, minus_count], axis=1)
    )
    return StochasticGradientResult(
        value=value,
        gradient=gradient,
        standard_error=standard_error,
        covariance=covariance,
        confidence_radius=z_value * standard_error,
        shots=shots,
        confidence_level=confidence,
        method="parameter_shift_shot_noise"
        if shift_rule.is_single_term
        else "multi_frequency_parameter_shift_shot_noise",
        shift=shift_rule.shift if shift_rule.is_single_term else None,
        coefficient=shift_rule.coefficient if shift_rule.is_single_term else None,
        evaluations=2 * term_count * sum(parameter.trainable for parameter in parameter_meta),
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        records=tuple(records),
        claim_boundary=STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )


GENERATOR_EIGENVALUES: Mapping[str, tuple[float, ...]] = {
    "rx": (-0.5, 0.5),
    "ry": (-0.5, 0.5),
    "rz": (-0.5, 0.5),
    "phase_rotation": (0.0, 1.0),
    "controlled_phase": (0.0, 1.0),
}
"""Registered generator eigenvalues for ``exp(-i * theta * G)`` gates.

Pauli rotations use ``G = P / 2``; the phase and controlled-phase gates use the
projector onto their phased state. Any other gate name is refused.
"""

NON_SMOOTH_OPERATIONS = frozenset({"reset", "postselect", "measure", "jump"})
"""Circuit operations whose effect on the expectation is not a smooth rotation."""

QUANTUM_GRADIENT_ESTIMATOR_PLAN_SCHEMA = "quantum_gradient_estimator_plan.v1"
"""Schema and major version of a spectrum-bound estimator plan."""

_CALIBRATION_SCHEMA = "quantum_gradient_calibration.v1"
_MAX_SPECTRUM_SUMS = 64
_FREQUENCY_DECIMALS = 12
_SPECTRUM_CLAIM_BOUNDARY = (
    "spectrum-bound parameter shift: exact for expectations whose dependence on each "
    "parameter is a trigonometric polynomial with the declared generator frequencies; "
    "the objective must evaluate exactly the declared gate occurrences, observable "
    "and calibration"
)


class QuantumGradientEstimatorRefusal(ValueError):
    """An estimator request was refused before any objective evaluation.

    Attributes
    ----------
    reason:
        Stable refusal code, such as ``unsupported_generator`` or
        ``calibration_changed``.

    """

    def __init__(self, reason: str, detail: str) -> None:
        """Record the refusal code and a readable explanation.

        Parameters
        ----------
        reason:
            Stable refusal code.
        detail:
            Human-readable description.

        """
        super().__init__(f"quantum gradient estimator refused ({reason}): {detail}")
        self.reason = reason


@dataclass(frozen=True)
class GateOccurrence:
    """One appearance of a trainable parameter in a circuit.

    Attributes
    ----------
    gate:
        Registered gate name from :data:`GENERATOR_EIGENVALUES`.
    parameter:
        Index of the parameter this gate angle depends on.
    scale:
        Nonzero factor in the gate angle ``scale * theta[parameter]``.

    """

    gate: str
    parameter: int
    scale: float = 1.0


@dataclass(frozen=True)
class QuantumGradientEstimatorPlan:
    """Spectrum-bound parameter-shift plan, fixed before execution.

    Attributes
    ----------
    parameter_count:
        Number of circuit parameters.
    occurrences:
        Every gate occurrence of every parameter, in circuit order.
    observable:
        Name of the measured observable bound to this plan.
    frequencies:
        Positive generator frequencies of each parameter; empty when the
        circuit does not depend on that parameter.
    rules:
        Exact parameter-shift rule for each parameter, or ``None`` when the
        parameter has no occurrence.
    shots:
        Shots per objective evaluation, or ``None`` for exact expectations.
    calibration_sha256:
        Canonical digest of the calibration the plan was made for.
    evaluations:
        Objective evaluations the execution performs, including the value.
    digest:
        Canonical digest of the whole plan.

    """

    parameter_count: int
    occurrences: tuple[GateOccurrence, ...]
    observable: str
    frequencies: tuple[tuple[float, ...], ...]
    rules: tuple[ParameterShiftRule | None, ...]
    shots: int | None
    calibration_sha256: str
    evaluations: int
    digest: str

    @property
    def shot_budget(self) -> int | None:
        """Total shots the execution spends, or ``None`` for exact expectations."""
        return None if self.shots is None else self.shots * self.evaluations


def plan_quantum_gradient_estimator(
    occurrences: Sequence[GateOccurrence],
    parameter_count: int,
    *,
    observable: str,
    calibration: Mapping[str, object],
    shots: int | None = None,
    operations: Sequence[str] = (),
) -> QuantumGradientEstimatorPlan:
    """Derive each parameter's generator frequencies and its exact shift rule.

    A parameter that appears in several gates is shifted as one variable, so
    its frequencies are all positive differences between sums of the scaled
    generator eigenvalues of its occurrences. A single Pauli rotation gives
    the frequency ``1`` and the standard rule; a parameter shared by two
    rotations gives ``{1, 2}`` and needs the multi-frequency rule.

    Parameters
    ----------
    occurrences:
        Every gate occurrence of every parameter.
    parameter_count:
        Number of circuit parameters.
    observable:
        Non-empty observable name bound to the plan.
    calibration:
        Calibration record the expectations depend on; its canonical digest is
        bound to the plan.
    shots:
        Positive shots per evaluation, or ``None`` for exact expectations.
    operations:
        Other circuit operations; reset, postselection, measurement and jumps
        are refused because they are not smooth rotations.

    Returns
    -------
    QuantumGradientEstimatorPlan
        Frequencies, rules, budget and digest for this circuit.

    Raises
    ------
    QuantumGradientEstimatorRefusal
        With reason ``non_smooth_operation``, ``unsupported_generator``,
        ``invalid_occurrence``, ``spectrum_too_large`` or
        ``ill_conditioned_spectrum``.
    ValueError
        If the count, observable, shots or calibration are malformed.

    """
    if type(parameter_count) is not int or parameter_count < 1:
        raise ValueError("parameter_count must be a positive integer")
    if type(observable) is not str or not observable:
        raise ValueError("observable must be a non-empty string")
    if shots is not None and (type(shots) is not int or shots < 1):
        raise ValueError("shots must be a positive integer or None")
    blocked = sorted({str(operation) for operation in operations} & NON_SMOOTH_OPERATIONS)
    if blocked:
        raise QuantumGradientEstimatorRefusal(
            "non_smooth_operation",
            f"{', '.join(blocked)} is not a smooth rotation of the expectation",
        )
    bound = tuple(occurrences)
    sums: list[set[float]] = [{0.0} for _ in range(parameter_count)]
    for occurrence in bound:
        if not isinstance(occurrence, GateOccurrence):
            raise QuantumGradientEstimatorRefusal(
                "invalid_occurrence", "occurrences must be GateOccurrence records"
            )
        eigenvalues = GENERATOR_EIGENVALUES.get(occurrence.gate)
        if eigenvalues is None:
            raise QuantumGradientEstimatorRefusal(
                "unsupported_generator",
                f"gate {occurrence.gate!r} has no registered generator spectrum",
            )
        index = occurrence.parameter
        scale = occurrence.scale
        if type(index) is not int or not 0 <= index < parameter_count:
            raise QuantumGradientEstimatorRefusal(
                "invalid_occurrence", f"parameter index {index!r} is out of range"
            )
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float, np.floating))
            or not np.isfinite(scale)
            or scale == 0
        ):
            raise QuantumGradientEstimatorRefusal(
                "invalid_occurrence", "occurrence scale must be finite and nonzero"
            )
        sums[index] = {
            round(total + float(scale) * eigenvalue, _FREQUENCY_DECIMALS)
            for total in sums[index]
            for eigenvalue in eigenvalues
        }
        if len(sums[index]) > _MAX_SPECTRUM_SUMS:
            raise QuantumGradientEstimatorRefusal(
                "spectrum_too_large",
                f"parameter {index} has more than {_MAX_SPECTRUM_SUMS} eigenvalue sums",
            )
    frequencies: list[tuple[float, ...]] = []
    rules: list[ParameterShiftRule | None] = []
    used = {occurrence.parameter for occurrence in bound}
    for index in range(parameter_count):
        if index not in used:
            frequencies.append(())
            rules.append(None)
            continue
        totals = sorted(sums[index])
        spectrum = tuple(
            sorted(
                {
                    round(upper - lower, _FREQUENCY_DECIMALS)
                    for position, lower in enumerate(totals)
                    for upper in totals[position + 1 :]
                }
            )
        )
        frequencies.append(spectrum)
        if spectrum == (1.0,):
            rules.append(ParameterShiftRule())
            continue
        try:
            rules.append(multi_frequency_parameter_shift_rule(spectrum))
        except ValueError as error:
            raise QuantumGradientEstimatorRefusal(
                "ill_conditioned_spectrum", f"parameter {index}: {error}"
            ) from error
    evaluations = 1 + sum(2 * len(rule.terms) for rule in rules if rule is not None)
    calibration_sha256 = canonical_digest(_CALIBRATION_SCHEMA, dict(calibration))
    body: dict[str, object] = {
        "parameter_count": parameter_count,
        "occurrences": [
            [occurrence.gate, occurrence.parameter, float(occurrence.scale)]
            for occurrence in bound
        ],
        "observable": observable,
        "frequencies": [list(spectrum) for spectrum in frequencies],
        "rules": [None if rule is None else [list(term) for term in rule.terms] for rule in rules],
        "shots": shots,
        "calibration_sha256": calibration_sha256,
        "evaluations": evaluations,
    }
    return QuantumGradientEstimatorPlan(
        parameter_count=parameter_count,
        occurrences=bound,
        observable=observable,
        frequencies=tuple(frequencies),
        rules=tuple(rules),
        shots=shots,
        calibration_sha256=calibration_sha256,
        evaluations=evaluations,
        digest=canonical_digest(QUANTUM_GRADIENT_ESTIMATOR_PLAN_SCHEMA, body),
    )


def execute_quantum_gradient_estimator(
    plan: QuantumGradientEstimatorPlan,
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    calibration: Mapping[str, object],
    shots: int | None = None,
    parameters: Sequence[Parameter] | None = None,
) -> GradientResult:
    """Run a spectrum-bound plan after checking its calibration and budget.

    Parameters
    ----------
    plan:
        Plan from :func:`plan_quantum_gradient_estimator`.
    objective:
        Expectation of the planned circuit, observable and calibration.
    values:
        Real parameter values of length ``plan.parameter_count``.
    calibration:
        Current calibration record; it must match the planned one.
    shots:
        Shots per evaluation now in force; they must match the planned ones.
    parameters:
        Optional metadata controlling names and trainable masks.

    Returns
    -------
    GradientResult
        Value and gradient; a parameter without occurrences has an exact zero.

    Raises
    ------
    QuantumGradientEstimatorRefusal
        With reason ``calibration_changed`` or ``budget_drift``, before the
        objective is evaluated.
    ValueError
        If the plan or values are malformed.

    """
    if not isinstance(plan, QuantumGradientEstimatorPlan):
        raise ValueError("execution requires a QuantumGradientEstimatorPlan")
    if canonical_digest(_CALIBRATION_SCHEMA, dict(calibration)) != plan.calibration_sha256:
        raise QuantumGradientEstimatorRefusal(
            "calibration_changed", "the calibration differs from the planned one"
        )
    if shots != plan.shots:
        raise QuantumGradientEstimatorRefusal(
            "budget_drift", f"planned {plan.shots!r} shots per evaluation, got {shots!r}"
        )
    parameter_values = _as_parameter_array(values)
    if parameter_values.size != plan.parameter_count:
        raise ValueError("values length must match the planned parameter count")
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    gradient = np.zeros_like(parameter_values)
    base_value = _as_scalar(objective(parameter_values.copy()))
    evaluations = 1
    for index, (parameter, rule) in enumerate(zip(parameter_meta, plan.rules, strict=True)):
        if rule is None or not parameter.trainable:
            continue
        for shift, coefficient in rule.terms:
            plus = parameter_values.copy()
            minus = parameter_values.copy()
            plus[index] += shift
            minus[index] -= shift
            gradient[index] += coefficient * (
                _as_scalar(objective(plus)) - _as_scalar(objective(minus))
            )
            evaluations += 2
    return GradientResult(
        value=base_value,
        gradient=gradient,
        method="spectrum_bound_parameter_shift",
        shift=None,
        coefficient=None,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        claim_boundary=_SPECTRUM_CLAIM_BOUNDARY,
    )


def _as_sample_provenance(
    sample_provenance: Mapping[str, object] | FiniteShotSampleProvenance | None,
) -> FiniteShotSampleProvenance:
    if sample_provenance is None:
        raise ValueError(
            "finite-shot sample provenance must include sample_seed, "
            "shot_batch_id, and source_class"
        )
    if isinstance(sample_provenance, FiniteShotSampleProvenance):
        return sample_provenance
    missing_keys = {"sample_seed", "shot_batch_id", "source_class"} - set(sample_provenance)
    if missing_keys:
        joined_keys = ", ".join(sorted(missing_keys))
        raise ValueError(f"finite-shot sample provenance missing {joined_keys}")
    return FiniteShotSampleProvenance(
        sample_seed=cast(str | int, sample_provenance["sample_seed"]),
        shot_batch_id=cast(str | int, sample_provenance["shot_batch_id"]),
        source_class=str(sample_provenance["source_class"]),
    )


__all__ = [
    "GENERATOR_EIGENVALUES",
    "NON_SMOOTH_OPERATIONS",
    "QUANTUM_GRADIENT_ESTIMATOR_PLAN_SCHEMA",
    "GateOccurrence",
    "QuantumGradientEstimatorPlan",
    "QuantumGradientEstimatorRefusal",
    "ScalarObjective",
    "batch_parameter_shift_gradient",
    "batch_value_and_parameter_shift_grad",
    "execute_quantum_gradient_estimator",
    "parameter_shift_gradient",
    "parameter_shift_gradient_with_uncertainty",
    "plan_quantum_gradient_estimator",
    "value_and_parameter_shift_grad",
]
