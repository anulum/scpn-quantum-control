# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generalised parameter shift module
# scpn-quantum-control -- generalized phase parameter-shift evidence
"""Generalised parameter-shift plans for finite generator spectra.

The functions in this module sit above the scalar differentiable core. They
allow each logical parameter to declare its own positive generator-frequency
spectrum, then materialise exact shifted evaluations or finite-shot stochastic
envelopes with explicit provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable_parameter_contracts import (
    Parameter,
    ParameterShiftRule,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
    multi_frequency_parameter_shift_rule,
)
from ..differentiable_result_contracts import (
    FiniteShotSampleProvenance,
    ParameterShiftSampleRecord,
    StochasticGradientResult,
)
from ..differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GradientFailurePolicy,
    gradient_confidence_interval,
)
from ..differentiable_transform_helpers import _as_scalar, _normalise_parameters

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective: TypeAlias = Callable[[FloatArray], float | int | np.floating[Any]]

GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY = (
    "local finite-spectrum parameter-shift planning and shifted-evaluation evidence; "
    "generator spectra, coefficients, trainable masks, sample provenance, and "
    "confidence envelopes are recorded; no opaque generator inference, provider "
    "callback, hardware execution, or benchmark-promotion claim"
)


@dataclass(frozen=True)
class GeneralisedParameterShiftTerm:
    """One plus/minus shifted evaluation in a finite-spectrum plan."""

    term_index: int
    parameter_index: int
    parameter_name: str
    frequency: float
    shift: float
    coefficient: float

    def __post_init__(self) -> None:
        """Validate term identity and finite shift-rule values."""
        if isinstance(self.term_index, bool) or not isinstance(self.term_index, int):
            raise ValueError("term_index must be an integer")
        if isinstance(self.parameter_index, bool) or not isinstance(self.parameter_index, int):
            raise ValueError("parameter_index must be an integer")
        if self.term_index < 0:
            raise ValueError("term_index must be non-negative")
        if self.parameter_index < 0:
            raise ValueError("parameter_index must be non-negative")
        if not isinstance(self.parameter_name, str) or not self.parameter_name:
            raise ValueError("parameter_name must be non-empty")
        frequency = _as_real_scalar("term frequency", self.frequency)
        shift = _as_real_scalar("term shift", self.shift)
        coefficient = _as_real_scalar("term coefficient", self.coefficient)
        if frequency <= 0.0:
            raise ValueError("term frequency must be finite and positive")
        if shift <= 0.0:
            raise ValueError("term shift must be finite and positive")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)

    def shifted_parameters(
        self,
        values: ArrayLike,
        *,
        sign: int,
    ) -> FloatArray:
        """Return a copied parameter vector shifted on this term's parameter."""
        if sign not in {-1, 1}:
            raise ValueError("sign must be -1 or 1")
        parameter_values = _as_parameter_array(values)
        if self.parameter_index >= parameter_values.size:
            raise ValueError("term parameter_index is out of range for values")
        shifted = parameter_values.copy()
        shifted[self.parameter_index] += float(sign) * self.shift
        return shifted

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready shifted-term metadata."""
        return {
            "term_index": self.term_index,
            "parameter_index": self.parameter_index,
            "parameter_name": self.parameter_name,
            "frequency": self.frequency,
            "shift": self.shift,
            "coefficient": self.coefficient,
        }


@dataclass(frozen=True)
class GeneralisedParameterShiftPlan:
    """Finite-spectrum parameter-shift plan for one parameter vector."""

    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    spectra: tuple[tuple[float, ...], ...]
    terms: tuple[GeneralisedParameterShiftTerm, ...]
    method: str = "generalised_parameter_shift"
    claim_boundary: str = GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate parameter metadata, spectra, and shifted term coverage."""
        if not self.parameter_names:
            raise ValueError("parameter_names must not be empty")
        if len(self.trainable) != len(self.parameter_names):
            raise ValueError("trainable length must match parameter_names")
        if len(self.spectra) != len(self.parameter_names):
            raise ValueError("spectra length must match parameter_names")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        normalised_spectra = tuple(_as_frequency_tuple(spectrum) for spectrum in self.spectra)
        terms = tuple(self.terms)
        for term_index, term in enumerate(terms):
            if term.term_index != term_index:
                raise ValueError("terms must be stored in term_index order")
            if term.parameter_index >= len(self.parameter_names):
                raise ValueError("term parameter_index is out of range")
            if term.parameter_name != self.parameter_names[term.parameter_index]:
                raise ValueError("term parameter_name must match parameter metadata")
            if not self.trainable[term.parameter_index]:
                raise ValueError("terms must not be allocated to frozen parameters")
            if term.frequency not in normalised_spectra[term.parameter_index]:
                raise ValueError("term frequency must come from the parameter spectrum")
        claim_boundary = str(self.claim_boundary).strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if not self.method:
            raise ValueError("method must be non-empty")
        object.__setattr__(self, "spectra", normalised_spectra)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def parameter_count(self) -> int:
        """Return the number of logical parameters in the plan."""
        return len(self.parameter_names)

    @property
    def shifted_evaluations(self) -> int:
        """Return the number of plus/minus objective evaluations in the plan."""
        return 2 * len(self.terms)

    def terms_for_parameter(
        self, parameter_index: int
    ) -> tuple[GeneralisedParameterShiftTerm, ...]:
        """Return shifted terms allocated to one logical parameter."""
        if (
            isinstance(parameter_index, bool)
            or not isinstance(parameter_index, int)
            or parameter_index < 0
            or parameter_index >= self.parameter_count
        ):
            raise ValueError("parameter_index is out of range")
        return tuple(term for term in self.terms if term.parameter_index == parameter_index)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready plan metadata."""
        return {
            "parameter_names": list(self.parameter_names),
            "trainable": list(self.trainable),
            "spectra": [list(spectrum) for spectrum in self.spectra],
            "terms": [term.to_dict() for term in self.terms],
            "parameter_count": self.parameter_count,
            "shifted_evaluations": self.shifted_evaluations,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GeneralisedParameterShiftResult:
    """Exact finite-spectrum parameter-shift gradient result."""

    value: float
    gradient: FloatArray
    plan: GeneralisedParameterShiftPlan
    evaluations: int
    method: str = "generalised_parameter_shift"
    claim_boundary: str = GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate exact gradient shape and provenance metadata."""
        value = _as_real_scalar("generalised parameter-shift value", self.value)
        gradient = _as_parameter_array(self.gradient)
        if gradient.size != self.plan.parameter_count:
            raise ValueError("gradient length must match plan parameter_count")
        if self.evaluations != 1 + self.plan.shifted_evaluations:
            raise ValueError("evaluations must equal base value plus shifted evaluations")
        if self.method != self.plan.method:
            raise ValueError("result method must match plan method")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready exact-gradient evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "evaluations": self.evaluations,
            "method": self.method,
            "parameter_names": list(self.plan.parameter_names),
            "trainable": list(self.plan.trainable),
            "plan": self.plan.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class GeneralisedStochasticParameterShiftResult:
    """Finite-shot generalised parameter-shift estimate with envelopes."""

    plan: GeneralisedParameterShiftPlan
    stochastic_gradient: StochasticGradientResult
    envelope: str = "independent_shifted_mean_normal_approximation"
    claim_boundary: str = GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate plan/result alignment and stochastic evidence boundary."""
        if self.stochastic_gradient.parameter_names != self.plan.parameter_names:
            raise ValueError("stochastic parameter_names must match plan")
        if self.stochastic_gradient.trainable != self.plan.trainable:
            raise ValueError("stochastic trainable mask must match plan")
        if self.stochastic_gradient.evaluations != self.plan.shifted_evaluations:
            raise ValueError("stochastic evaluations must match shifted plan evaluations")
        if not self.envelope:
            raise ValueError("envelope must be non-empty")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    @property
    def gradient(self) -> FloatArray:
        """Return the stochastic gradient estimate."""
        return self.stochastic_gradient.gradient

    @property
    def standard_error(self) -> FloatArray:
        """Return per-parameter finite-shot standard errors."""
        return self.stochastic_gradient.standard_error

    @property
    def confidence_radius(self) -> FloatArray:
        """Return per-parameter confidence radii."""
        return self.stochastic_gradient.confidence_radius

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready stochastic-gradient evidence."""
        return {
            "plan": self.plan.to_dict(),
            "stochastic_gradient": self.stochastic_gradient.to_dict(),
            "envelope": self.envelope,
            "claim_boundary": self.claim_boundary,
        }


def plan_generalised_parameter_shift(
    values: ArrayLike,
    generator_frequencies: Sequence[ArrayLike],
    *,
    parameters: Sequence[Parameter] | None = None,
    shifts: Sequence[ArrayLike | None] | None = None,
    max_condition: float = 1.0e10,
) -> GeneralisedParameterShiftPlan:
    """Plan finite-spectrum plus/minus shifts for every trainable parameter.

    Parameters
    ----------
    values:
        One-dimensional parameter vector used only for shape validation.
    generator_frequencies:
        Per-parameter positive spectral gaps. Each trainable parameter must
        provide a non-empty finite, unique spectrum.
    parameters:
        Optional parameter names and trainable masks.
    shifts:
        Optional per-parameter custom shift grids. ``None`` entries request the
        deterministic conditioned grid from the differentiable core.
    max_condition:
        Maximum accepted condition number for multi-frequency linear systems.

    Returns
    -------
    GeneralisedParameterShiftPlan
        Auditable shifted-evaluation plan.
    """
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    spectra = _normalise_generator_frequencies(generator_frequencies, parameter_values.size)
    custom_shifts = _normalise_shift_grid_inputs(shifts, parameter_values.size)
    terms: list[GeneralisedParameterShiftTerm] = []

    for parameter_index, parameter in enumerate(parameter_meta):
        if not parameter.trainable:
            continue
        rule = _rule_for_spectrum(
            spectra[parameter_index],
            shifts=custom_shifts[parameter_index],
            max_condition=max_condition,
        )
        frequencies = cast(tuple[float, ...], rule.frequencies)
        for frequency, (shift, coefficient) in zip(frequencies, rule.terms, strict=True):
            terms.append(
                GeneralisedParameterShiftTerm(
                    term_index=len(terms),
                    parameter_index=parameter_index,
                    parameter_name=parameter.name,
                    frequency=frequency,
                    shift=shift,
                    coefficient=coefficient,
                )
            )

    return GeneralisedParameterShiftPlan(
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        spectra=spectra,
        terms=tuple(terms),
    )


def value_and_generalised_parameter_shift_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    generator_frequencies: Sequence[ArrayLike],
    *,
    parameters: Sequence[Parameter] | None = None,
    shifts: Sequence[ArrayLike | None] | None = None,
    max_condition: float = 1.0e10,
) -> GeneralisedParameterShiftResult:
    """Evaluate a scalar objective and its finite-spectrum exact gradient."""
    parameter_values = _as_parameter_array(values)
    plan = plan_generalised_parameter_shift(
        parameter_values,
        generator_frequencies,
        parameters=parameters,
        shifts=shifts,
        max_condition=max_condition,
    )
    gradient = np.zeros(parameter_values.size, dtype=np.float64)
    base_value = _as_scalar(objective(parameter_values.copy()))
    for term in plan.terms:
        plus_value = _as_scalar(objective(term.shifted_parameters(parameter_values, sign=1)))
        minus_value = _as_scalar(objective(term.shifted_parameters(parameter_values, sign=-1)))
        gradient[term.parameter_index] += term.coefficient * (plus_value - minus_value)
    return GeneralisedParameterShiftResult(
        value=base_value,
        gradient=gradient,
        plan=plan,
        evaluations=1 + plan.shifted_evaluations,
    )


def generalised_parameter_shift_gradient(
    objective: ScalarObjective,
    values: ArrayLike,
    generator_frequencies: Sequence[ArrayLike],
    *,
    parameters: Sequence[Parameter] | None = None,
    shifts: Sequence[ArrayLike | None] | None = None,
    max_condition: float = 1.0e10,
) -> FloatArray:
    """Return the exact finite-spectrum parameter-shift gradient vector."""
    return value_and_generalised_parameter_shift_grad(
        objective,
        values,
        generator_frequencies,
        parameters=parameters,
        shifts=shifts,
        max_condition=max_condition,
    ).gradient


def estimate_generalised_parameter_shift_shot_noise(
    plan: GeneralisedParameterShiftPlan,
    plus_values: ArrayLike,
    minus_values: ArrayLike,
    plus_variances: ArrayLike,
    minus_variances: ArrayLike,
    plus_shots: ArrayLike,
    minus_shots: ArrayLike | None = None,
    *,
    value: float = 0.0,
    sample_provenance: Mapping[str, object] | FiniteShotSampleProvenance | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    failure_policy: GradientFailurePolicy | None = None,
) -> GeneralisedStochasticParameterShiftResult:
    """Propagate materialised finite-shot noise through a generalised plan.

    The shifted means and variances are one-dimensional arrays aligned with
    ``plan.terms``. The returned envelope records independent shot-noise
    variance per shifted term, plus confidence bounds from the shared
    stochastic-gradient policy.
    """
    provenance = _normalise_sample_provenance(sample_provenance)
    plus = _as_term_vector("plus_values", plus_values, plan)
    minus = _as_term_vector("minus_values", minus_values, plan)
    plus_var = _as_non_negative_term_vector("plus_variances", plus_variances, plan)
    minus_var = _as_non_negative_term_vector("minus_variances", minus_variances, plan)
    plus_count = _as_positive_shot_vector("plus_shots", plus_shots, plan)
    minus_count = (
        plus_count.copy()
        if minus_shots is None
        else _as_positive_shot_vector("minus_shots", minus_shots, plan)
    )
    value_scalar = _as_real_scalar("generalised stochastic value", value)
    level = _as_real_scalar("confidence_level", confidence_level)
    if level <= 0.0 or level >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    z_value = _as_real_scalar("confidence_z", confidence_z)
    if z_value <= 0.0:
        raise ValueError("confidence_z must be finite and positive")

    gradient = np.zeros(plan.parameter_count, dtype=np.float64)
    variance = np.zeros(plan.parameter_count, dtype=np.float64)
    shot_tensor = np.ones((len(plan.terms), 2, plan.parameter_count), dtype=np.float64)
    records: list[ParameterShiftSampleRecord] = []
    for row, term in enumerate(plan.terms):
        contribution = term.coefficient * (plus[row] - minus[row])
        variance_contribution = term.coefficient**2 * (
            plus_var[row] / float(plus_count[row]) + minus_var[row] / float(minus_count[row])
        )
        gradient[term.parameter_index] += contribution
        variance[term.parameter_index] += variance_contribution
        shot_tensor[row, 0, term.parameter_index] = float(plus_count[row])
        shot_tensor[row, 1, term.parameter_index] = float(minus_count[row])
        records.append(
            ParameterShiftSampleRecord(
                term_index=row,
                parameter_index=term.parameter_index,
                parameter_name=term.parameter_name,
                trainable=True,
                shift=term.shift,
                coefficient=term.coefficient,
                plus_value=float(plus[row]),
                minus_value=float(minus[row]),
                plus_variance=float(plus_var[row]),
                minus_variance=float(minus_var[row]),
                plus_shots=int(plus_count[row]),
                minus_shots=int(minus_count[row]),
                sample_seed=provenance.sample_seed,
                shot_batch_id=provenance.shot_batch_id,
                source_class=provenance.source_class,
                gradient_contribution=float(contribution),
                variance_contribution=float(variance_contribution),
            )
        )

    standard_error = np.sqrt(variance)
    confidence_interval = gradient_confidence_interval(
        gradient,
        standard_error,
        confidence_z=z_value,
        confidence_level=level,
        trainable=plan.trainable,
        failure_policy=failure_policy,
    )
    stochastic_gradient = StochasticGradientResult(
        value=value_scalar,
        gradient=gradient,
        standard_error=standard_error,
        covariance=np.diag(variance),
        confidence_radius=z_value * standard_error,
        shots=shot_tensor,
        confidence_level=level,
        method="generalised_parameter_shift_shot_noise",
        shift=None,
        coefficient=None,
        evaluations=plan.shifted_evaluations,
        parameter_names=plan.parameter_names,
        trainable=plan.trainable,
        records=tuple(records),
        claim_boundary=STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
        hardware_execution=False,
        confidence_interval=confidence_interval,
        failure_policy_status=confidence_interval.status,
        failure_reasons=confidence_interval.failure_reasons,
    )
    return GeneralisedStochasticParameterShiftResult(
        plan=plan,
        stochastic_gradient=stochastic_gradient,
    )


def _rule_for_spectrum(
    frequencies: tuple[float, ...],
    *,
    shifts: ArrayLike | None,
    max_condition: float,
) -> ParameterShiftRule:
    if len(frequencies) == 1:
        frequency = frequencies[0]
        if shifts is not None:
            shift_values = _as_parameter_array(shifts)
            if shift_values.size != 1:
                raise ValueError("single-frequency shifts must contain exactly one value")
            shift = float(shift_values[0])
            if shift <= 0.0:
                raise ValueError("single-frequency shift must be finite and positive")
        else:
            shift = float(np.pi / (2.0 * frequency))
        denominator = float(np.sin(frequency * shift))
        if abs(denominator) <= np.finfo(np.float64).eps:
            raise ValueError("single-frequency shift system is singular or ill-conditioned")
        coefficient = 0.5 * frequency / denominator
        return ParameterShiftRule(
            shift=shift,
            coefficient=coefficient,
            shifts=(shift,),
            coefficients=(coefficient,),
            frequencies=frequencies,
        )
    return multi_frequency_parameter_shift_rule(
        frequencies,
        shifts=shifts,
        max_condition=max_condition,
    )


def _as_frequency_tuple(values: ArrayLike) -> tuple[float, ...]:
    frequencies = _as_parameter_array(values)
    if frequencies.size == 0:
        raise ValueError("generator frequencies must contain at least one value")
    if np.any(frequencies <= 0.0):
        raise ValueError("generator frequencies must contain finite positive values")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("generator frequencies must be unique")
    return tuple(float(value) for value in frequencies)


def _normalise_generator_frequencies(
    generator_frequencies: Sequence[ArrayLike],
    parameter_count: int,
) -> tuple[tuple[float, ...], ...]:
    if len(generator_frequencies) != parameter_count:
        raise ValueError("generator_frequencies length must match parameter count")
    return tuple(_as_frequency_tuple(frequencies) for frequencies in generator_frequencies)


def _normalise_shift_grid_inputs(
    shifts: Sequence[ArrayLike | None] | None,
    parameter_count: int,
) -> tuple[ArrayLike | None, ...]:
    if shifts is None:
        return tuple(None for _index in range(parameter_count))
    if len(shifts) != parameter_count:
        raise ValueError("shifts length must match parameter count")
    return tuple(shifts)


def _normalise_sample_provenance(
    sample_provenance: Mapping[str, object] | FiniteShotSampleProvenance | None,
) -> FiniteShotSampleProvenance:
    if isinstance(sample_provenance, FiniteShotSampleProvenance):
        return sample_provenance
    if sample_provenance is None:
        raise ValueError("sample_provenance is required for stochastic estimates")
    sample_seed = sample_provenance.get("sample_seed", "")
    shot_batch_id = sample_provenance.get("shot_batch_id", "")
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, str | int):
        raise ValueError("sample_provenance sample_seed must be a string or integer")
    if isinstance(shot_batch_id, bool) or not isinstance(shot_batch_id, str | int):
        raise ValueError("sample_provenance shot_batch_id must be a string or integer")
    return FiniteShotSampleProvenance(
        sample_seed=sample_seed,
        shot_batch_id=shot_batch_id,
        source_class=str(sample_provenance.get("source_class", "")),
    )


def _as_term_vector(
    name: str,
    values: ArrayLike,
    plan: GeneralisedParameterShiftPlan,
) -> FloatArray:
    vector = _as_real_numeric_array(name, values)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional term vector")
    if vector.size != len(plan.terms):
        raise ValueError(f"{name} length must match plan terms")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(vector, dtype=np.float64)


def _as_non_negative_term_vector(
    name: str,
    values: ArrayLike,
    plan: GeneralisedParameterShiftPlan,
) -> FloatArray:
    vector = _as_term_vector(name, values, plan)
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return vector


def _as_positive_shot_vector(
    name: str,
    values: ArrayLike,
    plan: GeneralisedParameterShiftPlan,
) -> NDArray[np.int64]:
    raw = _as_term_vector(name, values, plan)
    if not np.allclose(raw, np.round(raw)):
        raise ValueError(f"{name} must contain integer shot counts")
    shots = np.asarray(np.round(raw), dtype=np.int64)
    if np.any(shots <= 0):
        raise ValueError(f"{name} must contain positive shot counts")
    return shots


__all__ = [
    "GENERALISED_PARAMETER_SHIFT_CLAIM_BOUNDARY",
    "GeneralisedParameterShiftPlan",
    "GeneralisedParameterShiftResult",
    "GeneralisedParameterShiftTerm",
    "GeneralisedStochasticParameterShiftResult",
    "estimate_generalised_parameter_shift_shot_noise",
    "generalised_parameter_shift_gradient",
    "plan_generalised_parameter_shift",
    "value_and_generalised_parameter_shift_grad",
]
