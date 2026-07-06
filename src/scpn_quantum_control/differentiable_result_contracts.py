# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable derivative result contracts
"""Validated result records for native differentiable-programming transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import (
    _as_index_vector,
    _as_parameter_array,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    StochasticGradientConfidenceInterval,
)

DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY = (
    "derivative result provenance is bounded by method, evaluations, parameter metadata, "
    "and trainable mask; no hardware or provider execution is implied"
)
FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY = (
    "finite-difference diagnostic only; not analytic, parameter-shift, native-framework, "
    "whole-program AD, provider, hardware, or production benchmark evidence"
)
FINITE_SHOT_SAMPLE_SOURCE_CLASSES = (
    "caller_supplied",
    "local_simulator",
    "provider_replay",
    "provider_runtime",
    "synthetic_fixture",
)
_PARAMETER_SHIFT_RECORD_TOLERANCE = 1.0e-12


def _normalise_claim_boundary(label: str, claim_boundary: str) -> str:
    boundary = str(claim_boundary).strip()
    if not boundary:
        raise ValueError(f"{label} claim_boundary must be non-empty")
    return boundary


def _require_zero_frozen_entries(
    name: str,
    values: NDArray[np.float64],
    trainable: tuple[bool, ...],
    *,
    axis: int = -1,
) -> None:
    """Reject derivative content assigned to non-trainable parameters."""
    frozen = np.logical_not(np.asarray(trainable, dtype=bool))
    if not np.any(frozen):
        return
    selected = np.take(values, np.flatnonzero(frozen), axis=axis)
    if np.any(selected != 0.0):
        raise ValueError(f"{name} must be zero for non-trainable parameters")


def _record_values_close(actual: float, expected: float) -> bool:
    return bool(
        np.isclose(
            actual,
            expected,
            rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
            atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        )
    )


def _normalise_provenance_token(name: str, value: object) -> str:
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise ValueError(f"{name} must be a non-empty string or integer token")
    if not isinstance(value, str | int):
        raise ValueError(f"{name} must be a non-empty string or integer token")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _validate_parameter_shift_record_reconstruction(
    gradient: NDArray[np.float64],
    covariance: NDArray[np.float64],
    records: tuple[ParameterShiftSampleRecord, ...],
) -> None:
    if not records:
        return

    reconstructed_gradient = np.zeros_like(gradient)
    reconstructed_variance = np.zeros_like(gradient)
    for record in records:
        reconstructed_gradient[record.parameter_index] += record.gradient_contribution
        reconstructed_variance[record.parameter_index] += record.variance_contribution

    if not np.allclose(
        reconstructed_gradient,
        gradient,
        rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError("stochastic gradient records must reconstruct gradient")
    if not np.allclose(
        reconstructed_variance,
        np.diag(covariance),
        rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError("stochastic gradient records must reconstruct covariance diagonal")
    if not np.allclose(
        covariance,
        np.diag(np.diag(covariance)),
        rtol=0.0,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError("stochastic gradient records require diagonal independent covariance")


def _validate_stochastic_uncertainty_moments(
    label: str,
    gradient: NDArray[np.float64],
    standard_error: NDArray[np.float64],
    covariance: NDArray[np.float64],
    confidence_radius: NDArray[np.float64],
    confidence_interval: StochasticGradientConfidenceInterval | None,
) -> None:
    covariance_diagonal = np.diag(covariance)
    if np.any(covariance_diagonal < -_PARAMETER_SHIFT_RECORD_TOLERANCE):
        raise ValueError(f"{label} covariance diagonal must be non-negative")
    if not np.allclose(
        standard_error**2,
        covariance_diagonal,
        rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError(f"{label} standard_error must match covariance diagonal")
    if confidence_interval is None:
        return

    interval_center = 0.5 * (confidence_interval.lower + confidence_interval.upper)
    interval_radius = 0.5 * (confidence_interval.upper - confidence_interval.lower)
    if not np.allclose(
        interval_center,
        gradient,
        rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError(f"{label} confidence_interval must be centered on gradient")
    if not np.allclose(
        confidence_radius,
        interval_radius,
        rtol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
        atol=_PARAMETER_SHIFT_RECORD_TOLERANCE,
    ):
        raise ValueError(f"{label} confidence_radius must match confidence_interval bounds")


def _as_vector_output(value: object) -> NDArray[np.float64]:
    vector = _as_real_numeric_array("vector output", value)
    if vector.ndim != 1:
        raise ValueError("vector output must be one-dimensional")
    return vector


@dataclass(frozen=True)
class GradientResult:
    """Value, gradient, and provenance returned by a differentiable backend."""

    value: float
    gradient: NDArray[np.float64]
    method: str
    shift: float | None
    coefficient: float | None
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate scalar value, gradient shape, trainability, and provenance."""
        value = _as_real_scalar("gradient result value", self.value)
        gradient = _as_real_numeric_array("gradient", self.gradient)
        claim_boundary = _normalise_claim_boundary("gradient result", self.claim_boundary)
        if gradient.ndim != 1:
            raise ValueError("gradient must be a one-dimensional array")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("gradient must contain only finite values")
        if not self.method:
            raise ValueError("gradient method must be non-empty")
        shift = None if self.shift is None else _as_real_scalar("gradient shift", self.shift)
        coefficient = (
            None
            if self.coefficient is None
            else _as_real_scalar("gradient coefficient", self.coefficient)
        )
        if shift is not None and shift <= 0.0:
            raise ValueError("gradient shift must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("gradient", gradient, self.trainable)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class FiniteShotSampleProvenance:
    """Source metadata for materialised finite-shot gradient samples.

    Parameters
    ----------
    sample_seed:
        Non-empty string or integer token identifying the stochastic sample,
        replay, or deterministic fixture seed.
    shot_batch_id:
        Non-empty string or integer token identifying the measurement batch
        that produced the supplied plus/minus sample tensors.
    source_class:
        One of ``FINITE_SHOT_SAMPLE_SOURCE_CLASSES``. The value distinguishes
        caller-supplied arrays, local simulator rows, provider replays,
        provider runtimes, and synthetic fixtures.
    """

    sample_seed: str | int
    shot_batch_id: str | int
    source_class: str

    def __post_init__(self) -> None:
        """Validate finite-shot provenance tokens and source classification."""
        sample_seed = _normalise_provenance_token(
            "finite-shot sample provenance sample_seed",
            self.sample_seed,
        )
        shot_batch_id = _normalise_provenance_token(
            "finite-shot sample provenance shot_batch_id",
            self.shot_batch_id,
        )
        source_class = str(self.source_class).strip()
        if source_class not in FINITE_SHOT_SAMPLE_SOURCE_CLASSES:
            allowed = ", ".join(FINITE_SHOT_SAMPLE_SOURCE_CLASSES)
            raise ValueError(
                f"finite-shot sample provenance source_class must be one of {allowed}"
            )
        object.__setattr__(self, "sample_seed", sample_seed)
        object.__setattr__(self, "shot_batch_id", shot_batch_id)
        object.__setattr__(self, "source_class", source_class)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready finite-shot sample provenance."""
        return {
            "sample_seed": self.sample_seed,
            "shot_batch_id": self.shot_batch_id,
            "source_class": self.source_class,
        }


@dataclass(frozen=True)
class ParameterShiftSampleRecord:
    """One plus/minus shifted sample used in stochastic parameter-shift propagation."""

    term_index: int
    parameter_index: int
    parameter_name: str
    trainable: bool
    shift: float
    coefficient: float
    plus_value: float
    minus_value: float
    plus_variance: float
    minus_variance: float
    plus_shots: int
    minus_shots: int
    sample_seed: str | int
    shot_batch_id: str | int
    source_class: str
    gradient_contribution: float
    variance_contribution: float

    def __post_init__(self) -> None:
        """Validate one finite-shot parameter-shift contribution record."""
        if isinstance(self.term_index, bool) or not isinstance(self.term_index, int):
            raise ValueError("parameter-shift record term_index must be an integer")
        if self.term_index < 0:
            raise ValueError("parameter-shift record term_index must be non-negative")
        if isinstance(self.parameter_index, bool) or not isinstance(self.parameter_index, int):
            raise ValueError("parameter-shift record parameter_index must be an integer")
        if self.parameter_index < 0:
            raise ValueError("parameter-shift record parameter_index must be non-negative")
        if not isinstance(self.parameter_name, str) or not self.parameter_name:
            raise ValueError("parameter-shift record parameter_name must be non-empty")
        if not isinstance(self.trainable, bool):
            raise ValueError("parameter-shift record trainable must be boolean")
        shift = _as_real_scalar("parameter-shift record shift", self.shift)
        coefficient = _as_real_scalar("parameter-shift record coefficient", self.coefficient)
        plus_value = _as_real_scalar("parameter-shift record plus_value", self.plus_value)
        minus_value = _as_real_scalar("parameter-shift record minus_value", self.minus_value)
        plus_variance = _as_real_scalar("parameter-shift record plus_variance", self.plus_variance)
        minus_variance = _as_real_scalar(
            "parameter-shift record minus_variance", self.minus_variance
        )
        gradient_contribution = _as_real_scalar(
            "parameter-shift record gradient_contribution",
            self.gradient_contribution,
        )
        variance_contribution = _as_real_scalar(
            "parameter-shift record variance_contribution",
            self.variance_contribution,
        )
        if shift <= 0.0:
            raise ValueError("parameter-shift record shift must be finite and positive")
        if plus_variance < 0.0 or minus_variance < 0.0 or variance_contribution < 0.0:
            raise ValueError("parameter-shift record variances must be non-negative")
        if (
            isinstance(self.plus_shots, bool)
            or not isinstance(self.plus_shots, int)
            or self.plus_shots <= 0
            or isinstance(self.minus_shots, bool)
            or not isinstance(self.minus_shots, int)
            or self.minus_shots <= 0
        ):
            raise ValueError("parameter-shift record shots must be positive integers")
        provenance = FiniteShotSampleProvenance(
            sample_seed=self.sample_seed,
            shot_batch_id=self.shot_batch_id,
            source_class=self.source_class,
        )
        expected_gradient = coefficient * (plus_value - minus_value)
        expected_variance = coefficient**2 * (
            plus_variance / float(self.plus_shots) + minus_variance / float(self.minus_shots)
        )
        if self.trainable:
            if not _record_values_close(gradient_contribution, expected_gradient):
                raise ValueError(
                    "parameter-shift record gradient_contribution must match shifted values"
                )
            if not _record_values_close(variance_contribution, expected_variance):
                raise ValueError(
                    "parameter-shift record variance_contribution must match shifted variances"
                )
        elif gradient_contribution != 0.0 or variance_contribution != 0.0:
            raise ValueError(
                "parameter-shift record contributions must be zero for non-trainable parameters"
            )
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "plus_value", plus_value)
        object.__setattr__(self, "minus_value", minus_value)
        object.__setattr__(self, "plus_variance", plus_variance)
        object.__setattr__(self, "minus_variance", minus_variance)
        object.__setattr__(self, "sample_seed", provenance.sample_seed)
        object.__setattr__(self, "shot_batch_id", provenance.shot_batch_id)
        object.__setattr__(self, "source_class", provenance.source_class)
        object.__setattr__(self, "gradient_contribution", gradient_contribution)
        object.__setattr__(self, "variance_contribution", variance_contribution)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready shifted-sample provenance."""
        return {
            "term_index": self.term_index,
            "parameter_index": self.parameter_index,
            "parameter_name": self.parameter_name,
            "trainable": self.trainable,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "plus_value": self.plus_value,
            "minus_value": self.minus_value,
            "plus_variance": self.plus_variance,
            "minus_variance": self.minus_variance,
            "plus_shots": self.plus_shots,
            "minus_shots": self.minus_shots,
            "sample_seed": self.sample_seed,
            "shot_batch_id": self.shot_batch_id,
            "source_class": self.source_class,
            "gradient_contribution": self.gradient_contribution,
            "variance_contribution": self.variance_contribution,
        }


@dataclass(frozen=True)
class StochasticGradientResult:
    """Parameter-shift gradient with independent shot-noise uncertainty."""

    value: float
    gradient: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    confidence_radius: NDArray[np.float64]
    shots: NDArray[np.float64]
    confidence_level: float
    method: str
    shift: float | None
    coefficient: float | None
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    records: tuple[ParameterShiftSampleRecord, ...] = ()
    claim_boundary: str = STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY
    hardware_execution: bool = False
    confidence_interval: StochasticGradientConfidenceInterval | None = None
    failure_policy_status: str = "not_evaluated"
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate stochastic parameter-shift moments and provenance records."""
        value = _as_real_scalar("stochastic gradient value", self.value)
        gradient = _as_parameter_array(self.gradient)
        standard_error = _as_parameter_array(self.standard_error)
        confidence_radius = _as_parameter_array(self.confidence_radius)
        covariance = _as_real_numeric_array("stochastic gradient covariance", self.covariance)
        shots = _as_real_numeric_array("stochastic gradient shots", self.shots)
        confidence_level = _as_real_scalar(
            "stochastic gradient confidence_level",
            self.confidence_level,
        )
        shift = (
            None
            if self.shift is None
            else _as_real_scalar("stochastic gradient shift", self.shift)
        )
        coefficient = (
            None
            if self.coefficient is None
            else _as_real_scalar("stochastic gradient coefficient", self.coefficient)
        )
        if standard_error.shape != gradient.shape:
            raise ValueError("standard_error shape must match gradient shape")
        if confidence_radius.shape != gradient.shape:
            raise ValueError("confidence_radius shape must match gradient shape")
        if covariance.shape != (gradient.size, gradient.size):
            raise ValueError("covariance shape must be gradient length squared")
        if shots.shape != (2, gradient.size) and not (
            shots.ndim == 3 and shots.shape[1:] == (2, gradient.size)
        ):
            raise ValueError(
                "shots shape must be (2, gradient length) or (n_terms, 2, gradient length)"
            )
        if not np.all(shots > 0.0) or not np.allclose(shots, np.round(shots)):
            raise ValueError("shots must contain positive integer counts")
        if not np.all(np.isfinite(standard_error)) or np.any(standard_error < 0.0):
            raise ValueError("standard_error must contain finite non-negative values")
        if not np.all(np.isfinite(confidence_radius)) or np.any(confidence_radius < 0.0):
            raise ValueError("confidence_radius must contain finite non-negative values")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must contain only finite values")
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if shift is not None and shift <= 0.0:
            raise ValueError("stochastic gradient shift must be finite and positive")
        if coefficient is not None and coefficient <= 0.0:
            raise ValueError("stochastic gradient coefficient must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("stochastic gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        records = tuple(self.records)
        for record in records:
            if not isinstance(record, ParameterShiftSampleRecord):
                raise ValueError("stochastic gradient records must contain sample records")
            if record.parameter_index >= gradient.size:
                raise ValueError("stochastic gradient record parameter_index is out of range")
            if record.parameter_name != self.parameter_names[record.parameter_index]:
                raise ValueError("stochastic gradient record parameter_name mismatch")
            if record.trainable != self.trainable[record.parameter_index]:
                raise ValueError("stochastic gradient record trainable mismatch")
        _validate_parameter_shift_record_reconstruction(gradient, covariance, records)
        claim_boundary = _normalise_claim_boundary(
            "stochastic gradient result",
            self.claim_boundary,
        )
        if self.hardware_execution is not False:
            raise ValueError("stochastic gradient result must not claim hardware execution")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if self.confidence_interval is not None:
            if self.confidence_interval.lower.shape != gradient.shape:
                raise ValueError("confidence_interval shape must match gradient shape")
            if self.failure_policy_status not in {"passed", "failed"}:
                raise ValueError("failure_policy_status must match evaluated interval status")
            if self.failure_policy_status != self.confidence_interval.status:
                raise ValueError("failure_policy_status must match confidence_interval status")
            if reasons != self.confidence_interval.failure_reasons:
                raise ValueError("failure_reasons must match confidence_interval")
        elif self.failure_policy_status != "not_evaluated":
            raise ValueError("failure_policy_status requires confidence_interval")
        _validate_stochastic_uncertainty_moments(
            "stochastic gradient",
            gradient,
            standard_error,
            covariance,
            confidence_radius,
            self.confidence_interval,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "confidence_radius", confidence_radius)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "claim_boundary", claim_boundary)
        object.__setattr__(self, "failure_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready stochastic parameter-shift evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "standard_error": self.standard_error.tolist(),
            "covariance": self.covariance.tolist(),
            "confidence_radius": self.confidence_radius.tolist(),
            "shots": self.shots.tolist(),
            "confidence_level": self.confidence_level,
            "method": self.method,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "evaluations": self.evaluations,
            "parameter_names": list(self.parameter_names),
            "trainable": list(self.trainable),
            "records": [record.to_dict() for record in self.records],
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
            "confidence_interval": None
            if self.confidence_interval is None
            else self.confidence_interval.to_dict(),
            "failure_policy_status": self.failure_policy_status,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class SPSAObjectiveSample:
    """One scalar objective sample for SPSA gradient estimation."""

    value: float
    variance: float | None = None
    shots: int | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate one scalar SPSA objective sample."""
        value = _as_real_scalar("SPSA sample value", self.value)
        variance = (
            None
            if self.variance is None
            else _as_real_scalar("SPSA sample variance", self.variance)
        )
        if variance is not None and variance < 0.0:
            raise ValueError("SPSA sample variance must be non-negative")
        if self.shots is not None and (
            isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0
        ):
            raise ValueError("SPSA sample shots must be a positive integer or None")
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready sample metadata."""
        return {
            "value": self.value,
            "variance": self.variance,
            "shots": self.shots,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SPSAProbeRecord:
    """One simultaneous perturbation probe pair."""

    repetition: int
    perturbation: NDArray[np.float64]
    plus_parameters: NDArray[np.float64]
    minus_parameters: NDArray[np.float64]
    plus: SPSAObjectiveSample
    minus: SPSAObjectiveSample
    gradient_estimate: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate a simultaneous-perturbation probe pair."""
        if isinstance(self.repetition, bool) or self.repetition < 0:
            raise ValueError("SPSA repetition must be a non-negative integer")
        perturbation = _as_parameter_array(self.perturbation)
        plus_parameters = _as_parameter_array(self.plus_parameters)
        minus_parameters = _as_parameter_array(self.minus_parameters)
        gradient = _as_parameter_array(self.gradient_estimate)
        if not (
            perturbation.shape == plus_parameters.shape == minus_parameters.shape == gradient.shape
        ):
            raise ValueError("SPSA record arrays must share parameter shape")
        object.__setattr__(self, "perturbation", perturbation)
        object.__setattr__(self, "plus_parameters", plus_parameters)
        object.__setattr__(self, "minus_parameters", minus_parameters)
        object.__setattr__(self, "gradient_estimate", gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready probe evidence."""
        return {
            "repetition": self.repetition,
            "perturbation": self.perturbation.tolist(),
            "plus_parameters": self.plus_parameters.tolist(),
            "minus_parameters": self.minus_parameters.tolist(),
            "plus": self.plus.to_dict(),
            "minus": self.minus.to_dict(),
            "gradient_estimate": self.gradient_estimate.tolist(),
        }


@dataclass(frozen=True)
class SPSAGradientResult:
    """Seeded SPSA gradient estimate with optional finite-shot uncertainty."""

    gradient: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    confidence_radius: NDArray[np.float64]
    records: tuple[SPSAProbeRecord, ...]
    perturbation_radius: float
    repetitions: int
    seed: int
    confidence_z: float
    method: str
    evaluations: int
    total_shots: int | None
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str
    hardware_execution: bool
    confidence_interval: StochasticGradientConfidenceInterval | None = None
    failure_policy_status: str = "not_evaluated"
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate SPSA gradient moments, records, and failure-policy metadata."""
        gradient = _as_parameter_array(self.gradient)
        standard_error = _as_parameter_array(self.standard_error)
        covariance = _as_real_numeric_array("SPSA covariance", self.covariance)
        confidence_radius = _as_parameter_array(self.confidence_radius)
        if standard_error.shape != gradient.shape or confidence_radius.shape != gradient.shape:
            raise ValueError("SPSA uncertainty vectors must match gradient shape")
        if covariance.shape != (gradient.size, gradient.size):
            raise ValueError("SPSA covariance shape must be gradient length squared")
        if np.any(standard_error < 0.0) or np.any(confidence_radius < 0.0):
            raise ValueError("SPSA uncertainty vectors must be non-negative")
        if self.perturbation_radius <= 0.0 or not np.isfinite(self.perturbation_radius):
            raise ValueError("SPSA perturbation_radius must be finite and positive")
        if self.repetitions <= 0:
            raise ValueError("SPSA repetitions must be positive")
        if self.evaluations != 2 * self.repetitions:
            raise ValueError("SPSA evaluations must equal two per repetition")
        if self.total_shots is not None and self.total_shots <= 0:
            raise ValueError("SPSA total_shots must be positive or None")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("SPSA parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("SPSA trainable mask length must match gradient length")
        if not self.claim_boundary:
            raise ValueError("SPSA claim_boundary must be non-empty")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if self.confidence_interval is not None:
            if self.confidence_interval.lower.shape != gradient.shape:
                raise ValueError("SPSA confidence_interval shape must match gradient shape")
            if self.failure_policy_status not in {"passed", "failed"}:
                raise ValueError("SPSA failure_policy_status must match interval status")
            if self.failure_policy_status != self.confidence_interval.status:
                raise ValueError("SPSA failure_policy_status must match confidence_interval")
            if reasons != self.confidence_interval.failure_reasons:
                raise ValueError("SPSA failure_reasons must match confidence_interval")
        elif self.failure_policy_status != "not_evaluated":
            raise ValueError("SPSA failure_policy_status requires confidence_interval")
        _validate_stochastic_uncertainty_moments(
            "SPSA",
            gradient,
            standard_error,
            covariance,
            confidence_radius,
            self.confidence_interval,
        )
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "confidence_radius", confidence_radius)
        object.__setattr__(self, "failure_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready SPSA gradient evidence."""
        return {
            "gradient": self.gradient.tolist(),
            "standard_error": self.standard_error.tolist(),
            "covariance": self.covariance.tolist(),
            "confidence_radius": self.confidence_radius.tolist(),
            "records": [record.to_dict() for record in self.records],
            "perturbation_radius": self.perturbation_radius,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "confidence_z": self.confidence_z,
            "method": self.method,
            "evaluations": self.evaluations,
            "total_shots": self.total_shots,
            "parameter_names": list(self.parameter_names),
            "trainable": list(self.trainable),
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
            "confidence_interval": None
            if self.confidence_interval is None
            else self.confidence_interval.to_dict(),
            "failure_policy_status": self.failure_policy_status,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class ScoreFunctionSampleRecord:
    """One materialised likelihood-ratio gradient sample."""

    index: int
    reward: float
    centred_reward: float
    score: NDArray[np.float64]
    weighted_score: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate one likelihood-ratio sample contribution."""
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("score-function sample index must be a non-negative integer")
        reward = _as_real_scalar("score-function sample reward", self.reward)
        centred_reward = _as_real_scalar(
            "score-function sample centred_reward",
            self.centred_reward,
        )
        score = _as_parameter_array(self.score)
        weighted_score = _as_parameter_array(self.weighted_score)
        if score.shape != weighted_score.shape:
            raise ValueError("score-function sample score arrays must share shape")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "centred_reward", centred_reward)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "weighted_score", weighted_score)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready sample evidence."""
        return {
            "index": self.index,
            "reward": self.reward,
            "centred_reward": self.centred_reward,
            "score": self.score.tolist(),
            "weighted_score": self.weighted_score.tolist(),
        }


@dataclass(frozen=True)
class ScoreFunctionGradientResult:
    """Likelihood-ratio gradient estimate with empirical uncertainty."""

    gradient: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    confidence_radius: NDArray[np.float64]
    records: tuple[ScoreFunctionSampleRecord, ...]
    baseline: float
    sample_count: int
    confidence_z: float
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str
    hardware_execution: bool
    confidence_interval: StochasticGradientConfidenceInterval | None = None
    failure_policy_status: str = "not_evaluated"
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate score-function gradient moments and sample provenance."""
        gradient = _as_parameter_array(self.gradient)
        standard_error = _as_parameter_array(self.standard_error)
        covariance = _as_real_numeric_array("score-function covariance", self.covariance)
        confidence_radius = _as_parameter_array(self.confidence_radius)
        baseline = _as_real_scalar("score-function baseline", self.baseline)
        z_value = _as_real_scalar("score-function confidence_z", self.confidence_z)
        if standard_error.shape != gradient.shape or confidence_radius.shape != gradient.shape:
            raise ValueError("score-function uncertainty vectors must match gradient shape")
        if covariance.shape != (gradient.size, gradient.size):
            raise ValueError("score-function covariance shape must be gradient length squared")
        if np.any(standard_error < 0.0) or np.any(confidence_radius < 0.0):
            raise ValueError("score-function uncertainty vectors must be non-negative")
        if self.sample_count < 2:
            raise ValueError("score-function sample_count must be at least two")
        if len(self.records) != self.sample_count:
            raise ValueError("score-function records length must match sample_count")
        if z_value <= 0.0:
            raise ValueError("score-function confidence_z must be finite and positive")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("score-function parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("score-function trainable mask length must match gradient length")
        if not self.claim_boundary:
            raise ValueError("score-function claim_boundary must be non-empty")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if self.confidence_interval is not None:
            if self.confidence_interval.lower.shape != gradient.shape:
                raise ValueError("score-function confidence_interval shape must match gradient")
            if self.failure_policy_status not in {"passed", "failed"}:
                raise ValueError("score-function failure_policy_status must match interval status")
            if self.failure_policy_status != self.confidence_interval.status:
                raise ValueError(
                    "score-function failure_policy_status must match confidence_interval"
                )
            if reasons != self.confidence_interval.failure_reasons:
                raise ValueError("score-function failure_reasons must match confidence_interval")
        elif self.failure_policy_status != "not_evaluated":
            raise ValueError("score-function failure_policy_status requires interval")
        _validate_stochastic_uncertainty_moments(
            "score-function",
            gradient,
            standard_error,
            covariance,
            confidence_radius,
            self.confidence_interval,
        )
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "confidence_radius", confidence_radius)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "confidence_z", z_value)
        object.__setattr__(self, "failure_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready score-function gradient evidence."""
        return {
            "gradient": self.gradient.tolist(),
            "standard_error": self.standard_error.tolist(),
            "covariance": self.covariance.tolist(),
            "confidence_radius": self.confidence_radius.tolist(),
            "records": [record.to_dict() for record in self.records],
            "baseline": self.baseline,
            "sample_count": self.sample_count,
            "confidence_z": self.confidence_z,
            "method": self.method,
            "parameter_names": list(self.parameter_names),
            "trainable": list(self.trainable),
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
            "confidence_interval": None
            if self.confidence_interval is None
            else self.confidence_interval.to_dict(),
            "failure_policy_status": self.failure_policy_status,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class ShotAllocationResult:
    """Per-parameter shot allocation for stochastic parameter-shift gradients."""

    shots: NDArray[np.float64]
    predicted_standard_error: NDArray[np.float64]
    covariance: NDArray[np.float64]
    target_standard_error: float
    total_shots: int
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate shot-allocation shapes, totals, and parameter metadata."""
        shots = _as_real_numeric_array("shot allocation shots", self.shots)
        standard_error = _as_parameter_array(self.predicted_standard_error)
        covariance = _as_real_numeric_array("shot allocation covariance", self.covariance)
        target = _as_real_scalar(
            "shot allocation target_standard_error",
            self.target_standard_error,
        )
        if shots.ndim == 2 and shots.shape[0] == 2:
            parameter_count = int(shots.shape[1])
        elif shots.ndim == 3 and shots.shape[1] == 2:
            parameter_count = int(shots.shape[2])
        else:
            raise ValueError(
                "shot allocation shots must have shape (2, n_parameters) "
                "or (n_terms, 2, n_parameters)"
            )
        if standard_error.shape != (parameter_count,):
            raise ValueError("predicted_standard_error length must match shot columns")
        if covariance.shape != (parameter_count, parameter_count):
            raise ValueError("shot allocation covariance shape must be n_parameters squared")
        if not np.all(shots > 0.0) or not np.allclose(shots, np.round(shots)):
            raise ValueError("shot allocation shots must contain positive integer counts")
        if not np.all(np.isfinite(standard_error)) or np.any(standard_error < 0.0):
            raise ValueError("predicted_standard_error must contain finite non-negative values")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("shot allocation covariance must contain only finite values")
        if target <= 0.0:
            raise ValueError("target_standard_error must be finite and positive")
        total_shots = int(self.total_shots)
        if total_shots != int(np.sum(shots)):
            raise ValueError("total_shots must equal allocated shot sum")
        if not self.method:
            raise ValueError("shot allocation method must be non-empty")
        if len(self.parameter_names) != parameter_count:
            raise ValueError("parameter_names length must match shot columns")
        if len(self.trainable) != parameter_count:
            raise ValueError("trainable mask length must match shot columns")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "predicted_standard_error", standard_error)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "target_standard_error", target)
        object.__setattr__(self, "total_shots", total_shots)


@dataclass(frozen=True)
class OptimizationResult:
    """Bounded gradient-descent result with convergence provenance."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    value_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64] | None = None
    best_value: float | None = None

    def __post_init__(self) -> None:
        """Validate deterministic optimisation traces and best-state metadata."""
        values = _as_parameter_array(self.values)
        if values.size != self.final_gradient.gradient.size:
            raise ValueError("optimized values length must match gradient length")
        if not self.value_history:
            raise ValueError("value_history must contain at least one value")
        history = tuple(_as_real_scalar("value_history item", item) for item in self.value_history)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("optimization steps must be a non-negative integer")
        if not isinstance(self.converged, bool):
            raise ValueError("optimization converged flag must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("optimization reason must be non-empty")
        best_values = values if self.best_values is None else _as_parameter_array(self.best_values)
        if best_values.size != values.size:
            raise ValueError("best_values length must match optimized values length")
        best_value = (
            min(history)
            if self.best_value is None
            else _as_real_scalar("best_value", self.best_value)
        )
        if best_value > min(history) + 1.0e-12:
            raise ValueError("best_value must not exceed the minimum value_history entry")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", history)
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class ArmijoLineSearchResult:
    """Backtracking line-search result with sufficient-decrease provenance."""

    values: NDArray[np.float64]
    value: float
    step_size: float
    direction: NDArray[np.float64]
    directional_derivative: float
    accepted: bool
    evaluations: int
    value_history: tuple[float, ...]
    reason: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate Armijo line-search state and sufficient-decrease metadata."""
        values = _as_parameter_array(self.values)
        direction = _as_parameter_array(self.direction)
        if direction.shape != values.shape:
            raise ValueError("line-search direction shape must match values shape")
        value = _as_real_scalar("line-search value", self.value)
        step_size = _as_real_scalar("line-search step_size", self.step_size)
        if step_size < 0.0:
            raise ValueError("line-search step_size must be finite and non-negative")
        directional_derivative = _as_real_scalar(
            "line-search directional_derivative",
            self.directional_derivative,
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("line-search accepted flag must be a boolean")
        if self.evaluations < 0:
            raise ValueError("line-search evaluations must be non-negative")
        if not self.value_history:
            raise ValueError("line-search value_history must be non-empty")
        value_history = tuple(
            _as_real_scalar("line-search value history", item) for item in self.value_history
        )
        if self.reason not in {"accepted", "non_descent_direction", "max_steps"}:
            raise ValueError("line-search reason must be a known status")
        if len(self.parameter_names) != values.size:
            raise ValueError("parameter_names length must match line-search values")
        if len(self.trainable) != values.size:
            raise ValueError("trainable mask length must match line-search values")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "directional_derivative", directional_derivative)
        object.__setattr__(self, "value_history", value_history)


@dataclass(frozen=True)
class GradientCheckResult:
    """Consistency check between two differentiable gradient estimators."""

    reference: GradientResult
    candidate: GradientResult
    max_abs_error: float
    l2_error: float
    value_delta: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        """Validate gradient-check operand shapes and error metrics."""
        if self.reference.gradient.shape != self.candidate.gradient.shape:
            raise ValueError("gradient check operands must have matching shapes")
        max_abs_error = _as_real_scalar("max_abs_error", self.max_abs_error)
        l2_error = _as_real_scalar("l2_error", self.l2_error)
        value_delta = _as_real_scalar("value_delta", self.value_delta)
        tolerance = _as_real_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if l2_error < 0.0:
            raise ValueError("l2_error must be non-negative")
        if value_delta < 0.0:
            raise ValueError("value_delta must be non-negative")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("gradient check passed flag must be a boolean")
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "l2_error", l2_error)
        object.__setattr__(self, "value_delta", value_delta)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class CustomDerivativeCheckResult:
    """Consistency audit for exact custom JVP/VJP derivative rules."""

    custom_jvp: JVPResult
    custom_vjp: VJPResult
    reference_jvp: JVPResult
    reference_vjp: VJPResult
    adjoint_inner_error: float
    jvp_l2_error: float
    vjp_l2_error: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        """Validate custom JVP/VJP comparisons against reference products."""
        if not isinstance(self.custom_jvp, JVPResult):
            raise ValueError("custom_jvp must be a JVPResult")
        if not isinstance(self.custom_vjp, VJPResult):
            raise ValueError("custom_vjp must be a VJPResult")
        if not isinstance(self.reference_jvp, JVPResult):
            raise ValueError("reference_jvp must be a JVPResult")
        if not isinstance(self.reference_vjp, VJPResult):
            raise ValueError("reference_vjp must be a VJPResult")
        if self.custom_jvp.value.shape != self.reference_jvp.value.shape:
            raise ValueError("custom and reference JVP values must have matching shapes")
        if self.custom_vjp.value.shape != self.reference_vjp.value.shape:
            raise ValueError("custom and reference VJP values must have matching shapes")
        if self.custom_jvp.jvp.shape != self.reference_jvp.jvp.shape:
            raise ValueError("custom and reference JVP outputs must have matching shapes")
        if self.custom_vjp.vjp.shape != self.reference_vjp.vjp.shape:
            raise ValueError("custom and reference VJP outputs must have matching shapes")
        adjoint_inner_error = _as_real_scalar(
            "custom derivative adjoint error",
            self.adjoint_inner_error,
        )
        jvp_l2_error = _as_real_scalar("custom derivative JVP l2 error", self.jvp_l2_error)
        vjp_l2_error = _as_real_scalar("custom derivative VJP l2 error", self.vjp_l2_error)
        tolerance = _as_real_scalar("custom derivative tolerance", self.tolerance)
        if adjoint_inner_error < 0.0 or jvp_l2_error < 0.0 or vjp_l2_error < 0.0:
            raise ValueError("custom derivative errors must be non-negative")
        if tolerance < 0.0:
            raise ValueError("custom derivative tolerance must be finite and non-negative")
        if not isinstance(self.passed, bool):
            raise ValueError("custom derivative passed flag must be a boolean")
        object.__setattr__(self, "adjoint_inner_error", adjoint_inner_error)
        object.__setattr__(self, "jvp_l2_error", jvp_l2_error)
        object.__setattr__(self, "vjp_l2_error", vjp_l2_error)
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class JacobianResult:
    """Value, Jacobian, and provenance for a vector-valued objective."""

    value: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate vector value, Jacobian shape, and parameter provenance."""
        value = _as_real_numeric_array("jacobian value", self.value)
        jacobian = _as_real_numeric_array("jacobian", self.jacobian)
        claim_boundary = _normalise_claim_boundary("jacobian result", self.claim_boundary)
        if value.ndim != 1:
            raise ValueError("jacobian value must be a one-dimensional array")
        if jacobian.ndim != 2:
            raise ValueError("jacobian must be a two-dimensional array")
        if jacobian.shape[0] != value.size:
            raise ValueError("jacobian row count must match value length")
        if not np.all(np.isfinite(value)):
            raise ValueError("jacobian value must contain only finite values")
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian must contain only finite values")
        if not self.method:
            raise ValueError("jacobian method must be non-empty")
        step = _as_real_scalar("jacobian step", self.step)
        if step < 0.0:
            raise ValueError("jacobian step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("jacobian evaluations must be non-negative")
        if len(self.parameter_names) != jacobian.shape[1]:
            raise ValueError("parameter_names length must match jacobian column count")
        if len(self.trainable) != jacobian.shape[1]:
            raise ValueError("trainable mask length must match jacobian column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("jacobian", jacobian, self.trainable, axis=1)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class JVPResult:
    """Jacobian-vector product with directional finite-difference provenance."""

    value: NDArray[np.float64]
    jvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate JVP value, tangent, product, and claim boundary."""
        value = _as_real_numeric_array("JVP value", self.value)
        jvp = _as_real_numeric_array("JVP", self.jvp)
        tangent = _as_real_numeric_array("JVP tangent", self.tangent)
        claim_boundary = _normalise_claim_boundary("JVP result", self.claim_boundary)
        if value.ndim != 1:
            raise ValueError("JVP value must be a one-dimensional array")
        if jvp.shape != value.shape:
            raise ValueError("JVP shape must match value shape")
        if tangent.ndim != 1:
            raise ValueError("JVP tangent must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(jvp)):
            raise ValueError("JVP value and product must contain only finite values")
        if not np.all(np.isfinite(tangent)):
            raise ValueError("JVP tangent must contain only finite values")
        if not self.method:
            raise ValueError("JVP method must be non-empty")
        step = _as_real_scalar("JVP step", self.step)
        if step < 0.0:
            raise ValueError("JVP step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("JVP evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match tangent length")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match tangent length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("JVP tangent", tangent, self.trainable)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jvp", jvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class VJPResult:
    """Vector-Jacobian product with cotangent provenance."""

    value: NDArray[np.float64]
    cotangent: NDArray[np.float64]
    vjp: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate VJP value, cotangent, product, and claim boundary."""
        value = _as_real_numeric_array("VJP value", self.value)
        cotangent = _as_real_numeric_array("VJP cotangent", self.cotangent)
        vjp = _as_real_numeric_array("VJP", self.vjp)
        claim_boundary = _normalise_claim_boundary("VJP result", self.claim_boundary)
        if value.ndim != 1:
            raise ValueError("VJP value must be a one-dimensional array")
        if cotangent.shape != value.shape:
            raise ValueError("VJP cotangent shape must match value shape")
        if vjp.ndim != 1:
            raise ValueError("VJP must be one-dimensional")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(cotangent)):
            raise ValueError("VJP value and cotangent must contain only finite values")
        if not np.all(np.isfinite(vjp)):
            raise ValueError("VJP must contain only finite values")
        if not self.method:
            raise ValueError("VJP method must be non-empty")
        step = _as_real_scalar("VJP step", self.step)
        if step < 0.0:
            raise ValueError("VJP step must be finite and non-negative")
        if self.evaluations < 0:
            raise ValueError("VJP evaluations must be non-negative")
        if len(self.parameter_names) != vjp.size:
            raise ValueError("parameter_names length must match VJP length")
        if len(self.trainable) != vjp.size:
            raise ValueError("trainable mask length must match VJP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("VJP", vjp, self.trainable)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "cotangent", cotangent)
        object.__setattr__(self, "vjp", vjp)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class HessianResult:
    """Value, Hessian, and provenance for a scalar objective."""

    value: float
    hessian: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate scalar Hessian shape, symmetry, and trainable mask."""
        value = _as_real_scalar("hessian value", self.value)
        hessian = _as_real_numeric_array("hessian", self.hessian)
        claim_boundary = _normalise_claim_boundary("hessian result", self.claim_boundary)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("hessian must be a square two-dimensional array")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("hessian must contain only finite values")
        if not self.method:
            raise ValueError("hessian method must be non-empty")
        step = _as_real_scalar("hessian step", self.step)
        if step <= 0.0:
            raise ValueError("hessian step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("hessian evaluations must be non-negative")
        if len(self.parameter_names) != hessian.shape[1]:
            raise ValueError("parameter_names length must match hessian dimension")
        if len(self.trainable) != hessian.shape[1]:
            raise ValueError("trainable mask length must match hessian dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if not np.allclose(hessian, hessian.T, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("hessian must be symmetric")
        _require_zero_frozen_entries("hessian columns", hessian, self.trainable, axis=1)
        _require_zero_frozen_entries("hessian rows", hessian, self.trainable, axis=0)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class SparseMatrixResult:
    """Coordinate sparse derivative matrix with parameter provenance."""

    row_indices: NDArray[np.int64]
    column_indices: NDArray[np.int64]
    values: NDArray[np.float64]
    shape: tuple[int, int]
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate coordinate sparse derivative matrix metadata."""
        rows = _as_index_vector("sparse row_indices", self.row_indices)
        columns = _as_index_vector("sparse column_indices", self.column_indices)
        values = _as_real_numeric_array("sparse values", self.values)
        if values.ndim != 1:
            raise ValueError("sparse values must be one-dimensional")
        if rows.size != columns.size or rows.size != values.size:
            raise ValueError("sparse row, column, and value lengths must match")
        if (
            len(self.shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.shape)
            or self.shape[0] < 1
            or self.shape[1] < 1
        ):
            raise ValueError("sparse shape must contain two positive integer dimensions")
        if rows.size:
            if int(rows.max()) >= self.shape[0] or int(columns.max()) >= self.shape[1]:
                raise ValueError("sparse indices must be inside matrix shape")
            coordinates = set(zip(rows.tolist(), columns.tolist()))
            if len(coordinates) != rows.size:
                raise ValueError("sparse indices must not contain duplicate coordinates")
        if not np.all(np.isfinite(values)):
            raise ValueError("sparse values must contain only finite values")
        if not self.method:
            raise ValueError("sparse method must be non-empty")
        if len(self.parameter_names) != self.shape[1]:
            raise ValueError("parameter_names length must match sparse column count")
        if len(self.trainable) != self.shape[1]:
            raise ValueError("trainable mask length must match sparse column count")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        frozen_columns = {
            index for index, is_trainable in enumerate(self.trainable) if not is_trainable
        }
        if any(
            int(column) in frozen_columns and float(value) != 0.0
            for column, value in zip(columns, values, strict=True)
        ):
            raise ValueError("sparse values must be zero for non-trainable parameters")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)

    @property
    def nnz(self) -> int:
        """Number of explicitly stored non-zero entries."""
        return int(self.values.size)

    def to_dense(self) -> NDArray[np.float64]:
        """Materialise the sparse coordinate matrix as a dense array."""
        dense = np.zeros(self.shape, dtype=np.float64)
        dense[self.row_indices, self.column_indices] = self.values
        return cast(NDArray[np.float64], dense)


@dataclass(frozen=True)
class HVPResult:
    """Hessian-vector product with nested finite-difference provenance."""

    value: float
    hvp: NDArray[np.float64]
    tangent: NDArray[np.float64]
    method: str
    step: float
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    claim_boundary: str = DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate Hessian-vector product, tangent, and parameter metadata."""
        value = _as_real_scalar("HVP value", self.value)
        hvp = _as_real_numeric_array("HVP", self.hvp)
        tangent = _as_real_numeric_array("HVP tangent", self.tangent)
        claim_boundary = _normalise_claim_boundary("HVP result", self.claim_boundary)
        if hvp.ndim != 1:
            raise ValueError("HVP must be one-dimensional")
        if tangent.shape != hvp.shape:
            raise ValueError("HVP tangent shape must match HVP shape")
        if not np.all(np.isfinite(hvp)) or not np.all(np.isfinite(tangent)):
            raise ValueError("HVP and tangent must contain only finite values")
        if not self.method:
            raise ValueError("HVP method must be non-empty")
        step = _as_real_scalar("HVP step", self.step)
        if step <= 0.0:
            raise ValueError("HVP step must be finite and positive")
        if self.evaluations < 0:
            raise ValueError("HVP evaluations must be non-negative")
        if len(self.parameter_names) != hvp.size:
            raise ValueError("parameter_names length must match HVP length")
        if len(self.trainable) != hvp.size:
            raise ValueError("trainable mask length must match HVP length")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        _require_zero_frozen_entries("HVP", hvp, self.trainable)
        _require_zero_frozen_entries("HVP tangent", tangent, self.trainable)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "hvp", hvp)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "claim_boundary", claim_boundary)


@dataclass(frozen=True)
class NaturalGradientResult:
    """Metric-preconditioned gradient with solve provenance."""

    base_gradient: GradientResult
    metric: NDArray[np.float64]
    natural_gradient: NDArray[np.float64]
    damping: float
    condition_number: float

    def __post_init__(self) -> None:
        """Validate metric-preconditioned gradient solve metadata."""
        metric = _as_real_numeric_array("natural-gradient metric", self.metric)
        natural_gradient = _as_real_numeric_array("natural_gradient", self.natural_gradient)
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise ValueError("natural-gradient metric must be a square matrix")
        if metric.shape[0] != self.base_gradient.gradient.size:
            raise ValueError("natural-gradient metric dimension must match gradient length")
        if natural_gradient.shape != self.base_gradient.gradient.shape:
            raise ValueError("natural_gradient shape must match gradient shape")
        if not np.all(np.isfinite(metric)):
            raise ValueError("natural-gradient metric must contain only finite values")
        if not np.all(np.isfinite(natural_gradient)):
            raise ValueError("natural_gradient must contain only finite values")
        if not np.allclose(metric, metric.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("natural-gradient metric must be symmetric")
        damping = _as_real_scalar("natural-gradient damping", self.damping)
        if damping < 0.0:
            raise ValueError("natural-gradient damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "natural-gradient condition_number", self.condition_number
        )
        if condition_number < 1.0:
            raise ValueError("natural-gradient condition_number must be at least 1")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "natural_gradient", natural_gradient)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class NaturalGradientOptimizationResult:
    """Bounded natural-gradient optimization trace and final state."""

    values: NDArray[np.float64]
    final_gradient: GradientResult
    final_natural_gradient: NaturalGradientResult
    value_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    natural_step_norm_history: tuple[float, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64]
    best_value: float

    def __post_init__(self) -> None:
        """Validate natural-gradient optimisation history and best state."""
        values = _as_parameter_array(self.values)
        best_values = _as_parameter_array(self.best_values)
        if best_values.shape != values.shape:
            raise ValueError("natural-gradient best_values shape must match values shape")
        if not isinstance(self.final_gradient, GradientResult):
            raise ValueError("final_gradient must be a GradientResult")
        if not isinstance(self.final_natural_gradient, NaturalGradientResult):
            raise ValueError("final_natural_gradient must be a NaturalGradientResult")
        if not self.value_history:
            raise ValueError("natural-gradient value_history must be non-empty")
        value_history = tuple(
            _as_real_scalar("natural-gradient value history", value)
            for value in self.value_history
        )
        gradient_norm_history = tuple(
            _as_real_scalar("natural-gradient gradient norm history", value)
            for value in self.gradient_norm_history
        )
        step_norm_history = tuple(
            _as_real_scalar("natural-gradient step norm history", value)
            for value in self.natural_step_norm_history
        )
        if any(value < 0.0 for value in gradient_norm_history):
            raise ValueError("gradient_norm_history must contain non-negative values")
        if any(value < 0.0 for value in step_norm_history):
            raise ValueError("natural_step_norm_history must contain non-negative values")
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("natural-gradient steps must be non-negative")
        if len(value_history) != steps + 1:
            raise ValueError("value_history must include initial value plus one per step")
        if len(gradient_norm_history) != steps + 1:
            raise ValueError("gradient_norm_history must include initial value plus one per step")
        if len(step_norm_history) != steps:
            raise ValueError("natural_step_norm_history must include one value per update step")
        if self.reason not in {
            "gradient_tolerance",
            "step_tolerance",
            "value_tolerance",
            "max_steps",
        }:
            raise ValueError("natural-gradient result reason must be known")
        best_value = _as_real_scalar("natural-gradient best_value", self.best_value)
        if best_value > min(value_history) + 1.0e-12:
            raise ValueError("best_value must be no larger than the recorded minimum")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_history", value_history)
        object.__setattr__(self, "gradient_norm_history", gradient_norm_history)
        object.__setattr__(self, "natural_step_norm_history", step_norm_history)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class LevenbergMarquardtStep:
    """Bounded Levenberg-Marquardt candidate step with model diagnostics."""

    gauss_newton: NaturalGradientResult
    step: NDArray[np.float64]
    candidate_values: NDArray[np.float64]
    damping: float
    predicted_reduction: float

    def __post_init__(self) -> None:
        """Validate one Levenberg-Marquardt step proposal."""
        step = _as_real_numeric_array("Levenberg-Marquardt step", self.step)
        candidate_values = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_values",
            self.candidate_values,
        )
        if step.ndim != 1:
            raise ValueError("Levenberg-Marquardt step must be one-dimensional")
        if candidate_values.shape != step.shape:
            raise ValueError("candidate_values shape must match step shape")
        if step.shape != self.gauss_newton.base_gradient.gradient.shape:
            raise ValueError("step shape must match Gauss-Newton gradient shape")
        if not np.all(np.isfinite(step)):
            raise ValueError("Levenberg-Marquardt step must contain only finite values")
        if not np.all(np.isfinite(candidate_values)):
            raise ValueError("candidate_values must contain only finite values")
        damping = _as_real_scalar("Levenberg-Marquardt damping", self.damping)
        if damping < 0.0:
            raise ValueError("Levenberg-Marquardt damping must be finite and non-negative")
        predicted_reduction = _as_real_scalar(
            "Levenberg-Marquardt predicted_reduction",
            self.predicted_reduction,
        )
        if predicted_reduction < -1.0e-12:
            raise ValueError("predicted_reduction must be non-negative")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "candidate_values", candidate_values)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "predicted_reduction", max(0.0, predicted_reduction))


@dataclass(frozen=True)
class LevenbergMarquardtTrial:
    """Actual-vs-predicted Levenberg-Marquardt acceptance diagnostic."""

    step_result: LevenbergMarquardtStep
    candidate_residual: NDArray[np.float64]
    candidate_value: float
    actual_reduction: float
    reduction_ratio: float
    accepted: bool

    def __post_init__(self) -> None:
        """Validate one Levenberg-Marquardt trial outcome."""
        if not isinstance(self.step_result, LevenbergMarquardtStep):
            raise ValueError("step_result must be a LevenbergMarquardtStep")
        candidate_residual = _as_real_numeric_array(
            "Levenberg-Marquardt candidate_residual",
            self.candidate_residual,
        )
        if candidate_residual.ndim != 1:
            raise ValueError("candidate_residual must be one-dimensional")
        if not np.all(np.isfinite(candidate_residual)):
            raise ValueError("candidate_residual must contain only finite values")
        candidate_value = _as_real_scalar(
            "Levenberg-Marquardt candidate_value",
            self.candidate_value,
        )
        actual_reduction = _as_real_scalar(
            "Levenberg-Marquardt actual_reduction",
            self.actual_reduction,
        )
        reduction_ratio = _as_real_scalar(
            "Levenberg-Marquardt reduction_ratio",
            self.reduction_ratio,
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted flag must be a boolean")
        object.__setattr__(self, "candidate_residual", candidate_residual)
        object.__setattr__(self, "candidate_value", candidate_value)
        object.__setattr__(self, "actual_reduction", actual_reduction)
        object.__setattr__(self, "reduction_ratio", reduction_ratio)


@dataclass(frozen=True)
class LevenbergMarquardtDampingUpdate:
    """Deterministic damping update for Levenberg-Marquardt trust regions."""

    trial: LevenbergMarquardtTrial
    next_damping: float
    action: str

    def __post_init__(self) -> None:
        """Validate damping update action after an LM trial."""
        if not isinstance(self.trial, LevenbergMarquardtTrial):
            raise ValueError("trial must be a LevenbergMarquardtTrial")
        next_damping = _as_real_scalar(
            "Levenberg-Marquardt next_damping",
            self.next_damping,
        )
        if next_damping < 0.0:
            raise ValueError("next_damping must be finite and non-negative")
        if self.action not in {"accept_decrease", "accept_keep", "reject_increase"}:
            raise ValueError("damping action must be a known Levenberg-Marquardt action")
        object.__setattr__(self, "next_damping", next_damping)


@dataclass(frozen=True)
class LevenbergMarquardtResult:
    """Traceable result from a bounded Levenberg-Marquardt optimization run."""

    values: NDArray[np.float64]
    residual: NDArray[np.float64]
    value_history: tuple[float, ...]
    damping_history: tuple[float, ...]
    accepted_history: tuple[bool, ...]
    steps: int
    converged: bool
    reason: str
    best_values: NDArray[np.float64]
    best_value: float

    def __post_init__(self) -> None:
        """Validate the full Levenberg-Marquardt result trace."""
        values = _as_parameter_array(self.values)
        residual = _as_vector_output(self.residual)
        best_values = _as_parameter_array(self.best_values)
        if best_values.shape != values.shape:
            raise ValueError("LM best values must match result values shape")
        if not self.value_history:
            raise ValueError("LM value history must contain the initial objective")
        value_history = tuple(
            _as_real_scalar("LM objective history value", value) for value in self.value_history
        )
        damping_history = tuple(
            _as_real_scalar("LM damping history value", value) for value in self.damping_history
        )
        accepted_history = tuple(bool(value) for value in self.accepted_history)
        steps = int(self.steps)
        if steps < 0:
            raise ValueError("LM result steps must be non-negative")
        if any(value < 0.0 for value in damping_history):
            raise ValueError("LM damping history must contain finite non-negative values")
        if len(accepted_history) != steps:
            raise ValueError("LM accepted history length must match executed steps")
        if len(damping_history) != steps + 1:
            raise ValueError(
                "LM damping history must include initial damping plus one entry per step"
            )
        if len(value_history) != steps + 1:
            raise ValueError("LM value history must include initial value plus one entry per step")
        best_value = _as_real_scalar("LM best objective", self.best_value)
        if best_value > min(value_history) + 1.0e-12:
            raise ValueError("LM best objective must be no larger than the recorded minimum")
        if self.reason not in {
            "residual_tolerance",
            "step_tolerance",
            "value_tolerance",
            "max_steps",
        }:
            raise ValueError("LM result reason must be a known convergence status")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "value_history", value_history)
        object.__setattr__(self, "damping_history", damping_history)
        object.__setattr__(self, "accepted_history", accepted_history)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "best_values", best_values)
        object.__setattr__(self, "best_value", best_value)


@dataclass(frozen=True)
class LeastSquaresCovarianceResult:
    """Parameter uncertainty estimate from a residual-map Fisher metric."""

    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    residual_variance: float
    degrees_of_freedom: int
    condition_number: float
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate least-squares covariance and parameter uncertainties."""
        covariance = _as_real_numeric_array("least-squares covariance", self.covariance)
        standard_errors = _as_real_numeric_array(
            "least-squares standard errors",
            self.standard_errors,
        )
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("least-squares covariance must be a square matrix")
        if standard_errors.ndim != 1 or standard_errors.shape[0] != covariance.shape[0]:
            raise ValueError("standard_errors length must match covariance dimension")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("least-squares covariance must contain only finite values")
        if not np.allclose(covariance, covariance.T, atol=1.0e-10):
            raise ValueError("least-squares covariance must be symmetric")
        if not np.all(np.isfinite(standard_errors)) or np.any(standard_errors < 0.0):
            raise ValueError("standard_errors must contain finite non-negative values")
        residual_variance = _as_real_scalar(
            "least-squares residual_variance",
            self.residual_variance,
        )
        if residual_variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")
        degrees_of_freedom = int(self.degrees_of_freedom)
        if degrees_of_freedom < 1:
            raise ValueError("degrees_of_freedom must be positive")
        condition_number = _as_real_scalar(
            "least-squares condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        if len(self.parameter_names) != covariance.shape[0]:
            raise ValueError("parameter_names length must match covariance dimension")
        if len(self.trainable) != covariance.shape[0]:
            raise ValueError("trainable mask length must match covariance dimension")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "residual_variance", residual_variance)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class FisherVectorProductResult:
    """Matrix-free empirical-Fisher vector product with provenance."""

    value: NDArray[np.float64]
    tangent: NDArray[np.float64]
    product: NDArray[np.float64]
    residual_projection: NDArray[np.float64]
    damping: float
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate empirical-Fisher vector-product operands."""
        value = _as_real_numeric_array("Fisher-vector value", self.value)
        tangent = _as_real_numeric_array("Fisher-vector tangent", self.tangent)
        product = _as_real_numeric_array("Fisher-vector product", self.product)
        projection = _as_real_numeric_array(
            "Fisher-vector residual_projection",
            self.residual_projection,
        )
        if value.ndim != 1:
            raise ValueError("Fisher-vector value must be one-dimensional")
        if tangent.ndim != 1 or product.shape != tangent.shape:
            raise ValueError("Fisher-vector tangent and product must be one-dimensional matches")
        if projection.shape != value.shape:
            raise ValueError("residual_projection shape must match value shape")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(projection)):
            raise ValueError("Fisher-vector value and projection must contain only finite values")
        if not np.all(np.isfinite(tangent)) or not np.all(np.isfinite(product)):
            raise ValueError("Fisher-vector tangent and product must contain only finite values")
        damping = _as_real_scalar("Fisher-vector damping", self.damping)
        if damping < 0.0:
            raise ValueError("Fisher-vector damping must be finite and non-negative")
        if not self.method:
            raise ValueError("Fisher-vector method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("Fisher-vector evaluations must be non-negative")
        if len(self.parameter_names) != tangent.size:
            raise ValueError("parameter_names length must match Fisher-vector dimension")
        if len(self.trainable) != tangent.size:
            raise ValueError("trainable mask length must match Fisher-vector dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "residual_projection", projection)
        object.__setattr__(self, "damping", damping)


@dataclass(frozen=True)
class FisherConjugateGradientResult:
    """Matrix-free empirical-Fisher conjugate-gradient solve result."""

    solution: NDArray[np.float64]
    residual_norm_history: tuple[float, ...]
    iterations: int
    converged: bool
    tolerance: float
    damping: float
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate empirical-Fisher conjugate-gradient solve history."""
        solution = _as_real_numeric_array("Fisher-CG solution", self.solution)
        if solution.ndim != 1:
            raise ValueError("Fisher-CG solution must be one-dimensional")
        if not np.all(np.isfinite(solution)):
            raise ValueError("Fisher-CG solution must contain only finite values")
        if not self.residual_norm_history:
            raise ValueError("Fisher-CG residual history must be non-empty")
        residual_history = tuple(
            _as_real_scalar("Fisher-CG residual norm", value)
            for value in self.residual_norm_history
        )
        if any(value < 0.0 for value in residual_history):
            raise ValueError("Fisher-CG residual norms must be finite and non-negative")
        iterations = int(self.iterations)
        if iterations < 0:
            raise ValueError("Fisher-CG iterations must be non-negative")
        if len(residual_history) != iterations + 1:
            raise ValueError("Fisher-CG residual history must include initial residual")
        tolerance = _as_real_scalar("Fisher-CG tolerance", self.tolerance)
        damping = _as_real_scalar("Fisher-CG damping", self.damping)
        if tolerance < 0.0:
            raise ValueError("Fisher-CG tolerance must be finite and non-negative")
        if damping < 0.0:
            raise ValueError("Fisher-CG damping must be finite and non-negative")
        if len(self.parameter_names) != solution.size:
            raise ValueError("parameter_names length must match Fisher-CG dimension")
        if len(self.trainable) != solution.size:
            raise ValueError("trainable mask length must match Fisher-CG dimension")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "residual_norm_history", residual_history)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "damping", damping)


@dataclass(frozen=True)
class WeightedGradientResult:
    """Weighted scalarisation of multiple scalar gradient results."""

    value: float
    gradient: NDArray[np.float64]
    components: tuple[GradientResult, ...]
    weights: NDArray[np.float64]
    method: str
    evaluations: int
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate a weighted scalarisation of gradient components."""
        if not self.components:
            raise ValueError("weighted gradient components must be non-empty")
        value = _as_real_scalar("weighted gradient value", self.value)
        gradient = _as_real_numeric_array("weighted gradient", self.gradient)
        weights = _as_real_numeric_array("weighted gradient weights", self.weights)
        if gradient.ndim != 1:
            raise ValueError("weighted gradient must be a one-dimensional array")
        if weights.ndim != 1 or weights.size != len(self.components):
            raise ValueError("weights length must match weighted gradient components")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("weighted gradient must contain only finite values")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if not self.method:
            raise ValueError("weighted gradient method must be non-empty")
        if self.evaluations < 0:
            raise ValueError("weighted gradient evaluations must be non-negative")
        if len(self.parameter_names) != gradient.size:
            raise ValueError("parameter_names length must match gradient length")
        if len(self.trainable) != gradient.size:
            raise ValueError("trainable mask length must match gradient length")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class ImplicitSensitivityResult:
    """Implicit-function sensitivity for a stationary differentiable system."""

    sensitivity: NDArray[np.float64]
    hessian: NDArray[np.float64]
    cross_derivative: NDArray[np.float64]
    damping: float
    condition_number: float
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    hyperparameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate implicit-function sensitivity operands and metadata."""
        sensitivity = _as_real_numeric_array("implicit sensitivity", self.sensitivity)
        hessian = _as_real_numeric_array("implicit hessian", self.hessian)
        cross = _as_real_numeric_array("implicit cross_derivative", self.cross_derivative)
        if sensitivity.ndim != 2 or hessian.ndim != 2 or cross.ndim != 2:
            raise ValueError("implicit sensitivity operands must be two-dimensional")
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("implicit hessian must be square")
        if sensitivity.shape != cross.shape:
            raise ValueError("implicit sensitivity shape must match cross_derivative shape")
        if sensitivity.shape[0] != hessian.shape[0]:
            raise ValueError("implicit sensitivity row count must match hessian dimension")
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("implicit sensitivity must contain only finite values")
        if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(cross)):
            raise ValueError("implicit operands must contain only finite values")
        if not np.allclose(hessian, hessian.T, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError("implicit hessian must be symmetric")
        damping = _as_real_scalar("implicit damping", self.damping)
        if damping < 0.0:
            raise ValueError("implicit damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "implicit condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("implicit condition_number must be at least 1")
        if not self.method:
            raise ValueError("implicit method must be non-empty")
        if len(self.parameter_names) != hessian.shape[0]:
            raise ValueError("parameter_names length must match implicit hessian dimension")
        if len(self.trainable) != hessian.shape[0]:
            raise ValueError("trainable mask length must match implicit hessian dimension")
        if len(self.hyperparameter_names) != cross.shape[1]:
            raise ValueError("hyperparameter_names length must match cross_derivative columns")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.hyperparameter_names):
            raise ValueError("hyperparameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "cross_derivative", cross)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True)
class FixedPointSensitivityResult:
    """Implicit sensitivity for a converged fixed-point map."""

    sensitivity: NDArray[np.float64]
    state_jacobian: NDArray[np.float64]
    parameter_jacobian: NDArray[np.float64]
    system_matrix: NDArray[np.float64]
    damping: float
    condition_number: float
    method: str
    parameter_names: tuple[str, ...]
    trainable: tuple[bool, ...]
    hyperparameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fixed-point sensitivity operands and metadata."""
        sensitivity = _as_real_numeric_array("fixed-point sensitivity", self.sensitivity)
        state_jacobian = _as_real_numeric_array("fixed-point state_jacobian", self.state_jacobian)
        parameter_jacobian = _as_real_numeric_array(
            "fixed-point parameter_jacobian",
            self.parameter_jacobian,
        )
        system_matrix = _as_real_numeric_array("fixed-point system_matrix", self.system_matrix)
        if (
            sensitivity.ndim != 2
            or state_jacobian.ndim != 2
            or parameter_jacobian.ndim != 2
            or system_matrix.ndim != 2
        ):
            raise ValueError("fixed-point sensitivity operands must be two-dimensional")
        if state_jacobian.shape[0] != state_jacobian.shape[1]:
            raise ValueError("fixed-point state_jacobian must be square")
        if system_matrix.shape != state_jacobian.shape:
            raise ValueError("fixed-point system_matrix shape must match state_jacobian")
        if sensitivity.shape != parameter_jacobian.shape:
            raise ValueError("fixed-point sensitivity shape must match parameter_jacobian")
        if sensitivity.shape[0] != state_jacobian.shape[0]:
            raise ValueError("fixed-point sensitivity row count must match state dimension")
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("fixed-point sensitivity must contain only finite values")
        if (
            not np.all(np.isfinite(state_jacobian))
            or not np.all(np.isfinite(parameter_jacobian))
            or not np.all(np.isfinite(system_matrix))
        ):
            raise ValueError("fixed-point operands must contain only finite values")
        damping = _as_real_scalar("fixed-point damping", self.damping)
        if damping < 0.0:
            raise ValueError("fixed-point damping must be finite and non-negative")
        condition_number = _as_real_scalar(
            "fixed-point condition_number",
            self.condition_number,
        )
        if condition_number < 1.0:
            raise ValueError("fixed-point condition_number must be at least 1")
        if not self.method:
            raise ValueError("fixed-point method must be non-empty")
        if len(self.parameter_names) != state_jacobian.shape[0]:
            raise ValueError("parameter_names length must match fixed-point state dimension")
        if len(self.trainable) != state_jacobian.shape[0]:
            raise ValueError("trainable mask length must match fixed-point state dimension")
        if len(self.hyperparameter_names) != parameter_jacobian.shape[1]:
            raise ValueError(
                "hyperparameter_names length must match fixed-point parameter_jacobian columns"
            )
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.hyperparameter_names):
            raise ValueError("hyperparameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "state_jacobian", state_jacobian)
        object.__setattr__(self, "parameter_jacobian", parameter_jacobian)
        object.__setattr__(self, "system_matrix", system_matrix)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "condition_number", condition_number)


__all__ = [
    "DIFFERENTIABLE_RESULT_CLAIM_BOUNDARY",
    "FINITE_DIFFERENCE_DIAGNOSTIC_CLAIM_BOUNDARY",
    "FINITE_SHOT_SAMPLE_SOURCE_CLASSES",
    "FiniteShotSampleProvenance",
    "GradientResult",
    "ParameterShiftSampleRecord",
    "StochasticGradientResult",
    "SPSAObjectiveSample",
    "SPSAProbeRecord",
    "SPSAGradientResult",
    "ScoreFunctionSampleRecord",
    "ScoreFunctionGradientResult",
    "ShotAllocationResult",
    "OptimizationResult",
    "ArmijoLineSearchResult",
    "GradientCheckResult",
    "CustomDerivativeCheckResult",
    "JacobianResult",
    "JVPResult",
    "VJPResult",
    "HessianResult",
    "SparseMatrixResult",
    "HVPResult",
    "NaturalGradientResult",
    "NaturalGradientOptimizationResult",
    "LevenbergMarquardtStep",
    "LevenbergMarquardtTrial",
    "LevenbergMarquardtDampingUpdate",
    "LevenbergMarquardtResult",
    "LeastSquaresCovarianceResult",
    "FisherVectorProductResult",
    "FisherConjugateGradientResult",
    "WeightedGradientResult",
    "ImplicitSensitivityResult",
    "FixedPointSensitivityResult",
    "_normalise_claim_boundary",
    "_require_zero_frozen_entries",
]
