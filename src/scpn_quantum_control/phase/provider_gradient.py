# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Provider Gradient Execution
"""Provider-safe parameter-shift gradient execution contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import ParameterShiftRule
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend
from .hardware_gradient_policy import (
    HardwareGradientPolicy,
    HardwareGradientPolicyDecision,
    HardwareGradientRequest,
    evaluate_hardware_gradient_policy,
)

FloatArray: TypeAlias = NDArray[np.float64]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
ProviderExpectationSampler: TypeAlias = Callable[
    [FloatArray, int | None],
    "ProviderExpectationSample",
]
_FINITE_SHOT_SAMPLE_PROVENANCE_SCHEMA = "finite_shot_provider_sample_provenance_v1"
_FINITE_SHOT_SAMPLE_PROVENANCE_KEYS = frozenset({"sample_seed", "shot_batch_id", "source_class"})
_FINITE_SHOT_SAMPLE_SOURCE_CLASSES = frozenset(
    {"local_simulator", "provider_replay", "provider_runtime", "synthetic_fixture"}
)


@dataclass(frozen=True)
class ProviderExpectationSample:
    """One provider or simulator expectation sample used by parameter shift."""

    value: float
    variance: float | None = None
    shots: int | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        value = _as_finite_scalar("sample value", self.value)
        variance = (
            None if self.variance is None else _as_non_negative_scalar("variance", self.variance)
        )
        shots = _normalise_shots("shots", self.shots)
        metadata = _normalise_metadata(self.metadata)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "metadata", metadata)

    def with_default_shots(self, shots: int | None) -> ProviderExpectationSample:
        """Return a sample with default shots filled in when the sample omitted them."""
        if self.shots is not None or shots is None:
            return self
        return ProviderExpectationSample(
            value=self.value,
            variance=self.variance,
            shots=shots,
            metadata=self.metadata,
        )

    def with_metadata(self, metadata: Mapping[str, object]) -> ProviderExpectationSample:
        """Return a sample with merged JSON-safe metadata."""
        merged_metadata: dict[str, object] = dict(self.metadata or {})
        merged_metadata.update(metadata)
        return ProviderExpectationSample(
            value=self.value,
            variance=self.variance,
            shots=self.shots,
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible sample metadata."""
        return {
            "value": self.value,
            "variance": self.variance,
            "shots": self.shots,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ProviderParameterShiftRecord:
    """One plus/minus parameter-shift record for a provider callback."""

    parameter_index: int
    shift_index: int
    shift: float
    coefficient: float
    plus_parameters: FloatArray
    minus_parameters: FloatArray
    plus: ProviderExpectationSample
    minus: ProviderExpectationSample
    gradient: float
    standard_error: float
    confidence_radius: float

    def __post_init__(self) -> None:
        if isinstance(self.parameter_index, bool) or self.parameter_index < 0:
            raise ValueError("parameter_index must be a non-negative integer")
        if isinstance(self.shift_index, bool) or self.shift_index < 0:
            raise ValueError("shift_index must be a non-negative integer")
        shift = _as_positive_scalar("shift", self.shift)
        coefficient = _as_finite_scalar("coefficient", self.coefficient)
        plus_parameters = _as_finite_vector("plus_parameters", self.plus_parameters)
        minus_parameters = _as_finite_vector("minus_parameters", self.minus_parameters)
        if plus_parameters.shape != minus_parameters.shape:
            raise ValueError("plus_parameters and minus_parameters must have matching shapes")
        gradient = _as_finite_scalar("gradient", self.gradient)
        standard_error = _as_non_negative_scalar("standard_error", self.standard_error)
        confidence_radius = _as_non_negative_scalar("confidence_radius", self.confidence_radius)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "plus_parameters", plus_parameters)
        object.__setattr__(self, "minus_parameters", minus_parameters)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "confidence_radius", confidence_radius)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible shift record metadata."""
        return {
            "parameter_index": self.parameter_index,
            "shift_index": self.shift_index,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "plus_parameters": self.plus_parameters.tolist(),
            "minus_parameters": self.minus_parameters.tolist(),
            "plus": self.plus.to_dict(),
            "minus": self.minus.to_dict(),
            "gradient": self.gradient,
            "standard_error": self.standard_error,
            "confidence_radius": self.confidence_radius,
        }


@dataclass(frozen=True)
class ProviderGradientExecutionResult:
    """Provider-safe parameter-shift gradient execution result."""

    backend: str
    method: str
    values: FloatArray
    gradient: FloatArray
    standard_error: FloatArray
    confidence_radius: FloatArray
    records: tuple[ProviderParameterShiftRecord, ...]
    plan: QuantumGradientPlan
    total_evaluations: int
    total_shots: int | None
    claim_boundary: str

    def __post_init__(self) -> None:
        values = _as_finite_vector("values", self.values)
        gradient = _as_finite_vector("gradient", self.gradient)
        standard_error = _as_finite_vector("standard_error", self.standard_error)
        confidence_radius = _as_finite_vector("confidence_radius", self.confidence_radius)
        if not (values.shape == gradient.shape == standard_error.shape == confidence_radius.shape):
            raise ValueError(
                "values, gradient, standard_error, and confidence_radius shapes must match"
            )
        if self.total_evaluations <= 0:
            raise ValueError("total_evaluations must be positive")
        if self.total_shots is not None and self.total_shots <= 0:
            raise ValueError("total_shots must be positive or None")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "confidence_radius", confidence_radius)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible provider-gradient execution metadata."""
        return {
            "backend": self.backend,
            "method": self.method,
            "values": self.values.tolist(),
            "gradient": self.gradient.tolist(),
            "standard_error": self.standard_error.tolist(),
            "confidence_radius": self.confidence_radius.tolist(),
            "records": [record.to_dict() for record in self.records],
            "plan": {
                "backend": self.plan.backend,
                "family": self.plan.family,
                "method": self.plan.method,
                "supported": self.plan.supported,
                "shift_terms": self.plan.shift_terms,
                "evaluations": self.plan.evaluations,
                "shots": self.plan.shots,
                "finite_shot": self.plan.finite_shot,
                "requires_hardware_approval": self.plan.requires_hardware_approval,
                "reasons": list(self.plan.reasons),
                "alternatives": list(self.plan.alternatives),
            },
            "total_evaluations": self.total_evaluations,
            "total_shots": self.total_shots,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ProviderHardwareGradientPreparationResult:
    """Provider hardware-gradient preparation record without QPU submission."""

    provider: str
    backend: str
    method: str
    mode: str
    values: FloatArray
    decision: HardwareGradientPolicyDecision
    plan: QuantumGradientPlan | None
    rule_terms: int
    total_evaluations: int
    estimated_total_shots: int
    claim_boundary: str

    def __post_init__(self) -> None:
        values = _as_finite_vector("values", self.values)
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.backend.strip():
            raise ValueError("backend must be non-empty")
        if not self.method.strip():
            raise ValueError("method must be non-empty")
        if not self.mode.strip():
            raise ValueError("mode must be non-empty")
        if self.rule_terms <= 0:
            raise ValueError("rule_terms must be positive")
        if self.total_evaluations <= 0:
            raise ValueError("total_evaluations must be positive")
        if self.estimated_total_shots <= 0:
            raise ValueError("estimated_total_shots must be positive")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(self, "backend", self.backend.strip())
        object.__setattr__(self, "method", self.method.strip())
        object.__setattr__(self, "mode", self.mode.strip())
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    @property
    def approved(self) -> bool:
        """Whether provider hardware-gradient preparation passed policy checks."""
        return self.decision.approved

    @property
    def fail_closed(self) -> bool:
        """Whether provider hardware-gradient preparation is blocked."""
        return self.decision.fail_closed

    @property
    def gradient_available(self) -> bool:
        """Whether a gradient vector was produced by hardware execution."""
        return False

    @property
    def hardware_execution(self) -> bool:
        """Whether this record submitted or executed a hardware job."""
        return False

    @property
    def failure_reason(self) -> str:
        """Human-readable fail-closed reason from the policy decision."""
        return self.decision.failure_reason

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible hardware-preparation metadata."""
        return {
            "provider": self.provider,
            "backend": self.backend,
            "method": self.method,
            "mode": self.mode,
            "values": self.values.tolist(),
            "decision": self.decision.to_dict(),
            "plan": None if self.plan is None else _plan_to_dict(self.plan),
            "rule_terms": self.rule_terms,
            "total_evaluations": self.total_evaluations,
            "estimated_total_shots": self.estimated_total_shots,
            "claim_boundary": self.claim_boundary,
            "approved": self.approved,
            "fail_closed": self.fail_closed,
            "gradient_available": self.gradient_available,
            "hardware_execution": self.hardware_execution,
            "failure_reason": self.failure_reason,
        }


def prepare_provider_hardware_parameter_shift_gradient(
    values: ArrayLike,
    *,
    provider: str = "ibm_quantum",
    backend: str = "ibm_quantum",
    shots: int | None = None,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
    policy: HardwareGradientPolicy | None = None,
    evidence_ids: Mapping[str, str] | None = None,
    dry_run_only: bool = True,
    live_execution_ticket: str | None = None,
    allow_hardware: bool = True,
) -> ProviderHardwareGradientPreparationResult:
    """Prepare a policy-bound provider hardware-gradient request without QPU execution."""
    values_vector = _as_finite_vector("values", values)
    terms = _parameter_shift_terms(rule, shift)
    active_policy = policy or HardwareGradientPolicy()
    planned_shots = shots if shots is not None else active_policy.default_shots
    decision = evaluate_hardware_gradient_policy(
        HardwareGradientRequest(
            provider=provider,
            backend=backend,
            n_params=values_vector.size,
            shots=planned_shots,
            shift_terms=len(terms),
            allow_hardware=allow_hardware,
            evidence_ids=evidence_ids,
            dry_run_only=dry_run_only,
            live_execution_ticket=live_execution_ticket,
        ),
        policy=active_policy,
        scenario="provider_hardware_gradient_preparation",
    )
    plan = _provider_hardware_plan(
        backend=backend,
        n_params=values_vector.size,
        shift_terms=len(terms),
        shots=planned_shots,
        confidence_level=active_policy.confidence_level,
        allow_hardware=allow_hardware,
    )
    return ProviderHardwareGradientPreparationResult(
        provider=provider,
        backend=backend,
        method=_hardware_preparation_method(decision.mode, len(terms)),
        mode=decision.mode,
        values=values_vector,
        decision=decision,
        plan=plan,
        rule_terms=len(terms),
        total_evaluations=decision.evaluations,
        estimated_total_shots=decision.estimated_total_shots,
        claim_boundary=decision.claim_boundary,
    )


def execute_provider_parameter_shift_gradient(
    sampler: ProviderExpectationSampler,
    values: ArrayLike,
    *,
    backend: str = "statevector_simulator",
    shots: int | None = None,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    allow_hardware: bool = False,
) -> ProviderGradientExecutionResult:
    """Execute parameter-shift gradients through an explicit provider callback.

    The callback receives shifted parameter vectors and the planned shot count.
    Hardware-like backends remain fail-closed unless ``allow_hardware`` is set
    and the backend planner declares the route supported.
    """
    values_vector = _as_finite_vector("values", values)
    terms = _parameter_shift_terms(rule, shift)
    z_value = _as_non_negative_scalar("confidence_z", confidence_z)
    plan = plan_quantum_gradient_backend(
        backend,
        n_params=values_vector.size,
        shift_terms=len(terms),
        shots=shots,
        finite_shot=shots is not None,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )
    if plan.fail_closed:
        joined = "; ".join(plan.reasons)
        raise ValueError(f"provider gradient plan is unsupported: {joined}")

    records: list[ProviderParameterShiftRecord] = []
    gradient = np.zeros(values_vector.size, dtype=np.float64)
    variance = np.zeros(values_vector.size, dtype=np.float64)
    total_shots = 0
    saw_shots = False

    for index in range(values_vector.size):
        for shift_index, (shift_value, coefficient) in enumerate(terms):
            plus_parameters = values_vector.copy()
            minus_parameters = values_vector.copy()
            plus_parameters[index] += shift_value
            minus_parameters[index] -= shift_value
            plus = _sample_provider(sampler, plus_parameters, plan.shots)
            minus = _sample_provider(sampler, minus_parameters, plan.shots)
            if plan.finite_shot and (plus.variance is None or minus.variance is None):
                raise ValueError("finite-shot provider gradients require sample variance")
            if plan.finite_shot:
                _validate_finite_shot_sample_provenance("plus", plus)
                _validate_finite_shot_sample_provenance("minus", minus)
            plus = _attach_shift_sample_provenance(
                plus,
                parameter_index=index,
                shift_index=shift_index,
                direction="plus",
                shift=shift_value,
                coefficient=coefficient,
                shifted_parameters=plus_parameters,
                finite_shot=plan.finite_shot,
            )
            minus = _attach_shift_sample_provenance(
                minus,
                parameter_index=index,
                shift_index=shift_index,
                direction="minus",
                shift=shift_value,
                coefficient=coefficient,
                shifted_parameters=minus_parameters,
                finite_shot=plan.finite_shot,
            )

            gradient_value = coefficient * (plus.value - minus.value)
            standard_error_value = _standard_error(
                plus,
                minus,
                coefficient=coefficient,
                require_variance=plan.finite_shot,
            )
            confidence_radius_value = z_value * standard_error_value
            gradient[index] += gradient_value
            variance[index] += standard_error_value**2
            record = ProviderParameterShiftRecord(
                parameter_index=index,
                shift_index=shift_index,
                shift=shift_value,
                coefficient=coefficient,
                plus_parameters=plus_parameters,
                minus_parameters=minus_parameters,
                plus=plus,
                minus=minus,
                gradient=gradient_value,
                standard_error=standard_error_value,
                confidence_radius=confidence_radius_value,
            )
            records.append(record)
            for sample in (plus, minus):
                if sample.shots is not None:
                    saw_shots = True
                    total_shots += sample.shots
    standard_error = np.sqrt(variance).astype(np.float64, copy=False)
    confidence_radius = (z_value * standard_error).astype(np.float64, copy=False)

    return ProviderGradientExecutionResult(
        backend=plan.backend,
        method=_result_method(plan.method, len(terms)),
        values=values_vector,
        gradient=gradient,
        standard_error=standard_error,
        confidence_radius=confidence_radius,
        records=tuple(records),
        plan=plan,
        total_evaluations=2 * len(records),
        total_shots=total_shots if saw_shots else None,
        claim_boundary=(
            "provider callback parameter-shift execution with explicit sampling, "
            "shot accounting, uncertainty propagation, and fail-closed hardware policy; "
            "not a proof of provider availability or quantum advantage"
        ),
    )


def _plan_to_dict(plan: QuantumGradientPlan) -> dict[str, object]:
    return {
        "backend": plan.backend,
        "family": plan.family,
        "method": plan.method,
        "supported": plan.supported,
        "shift_terms": plan.shift_terms,
        "evaluations": plan.evaluations,
        "shots": plan.shots,
        "finite_shot": plan.finite_shot,
        "requires_hardware_approval": plan.requires_hardware_approval,
        "reasons": list(plan.reasons),
        "alternatives": list(plan.alternatives),
    }


def _provider_hardware_plan(
    *,
    backend: str,
    n_params: int,
    shift_terms: int,
    shots: int,
    confidence_level: float,
    allow_hardware: bool,
) -> QuantumGradientPlan | None:
    try:
        return plan_quantum_gradient_backend(
            backend,
            n_params=n_params,
            shift_terms=shift_terms,
            shots=shots,
            finite_shot=True,
            confidence_level=confidence_level,
            allow_hardware=allow_hardware,
        )
    except ValueError:
        return None


def _hardware_preparation_method(mode: str, rule_terms: int) -> str:
    rule_prefix = "multi_frequency_" if rule_terms > 1 else ""
    if mode == "live_ticketed":
        return f"hardware_policy_live_ticketed_{rule_prefix}parameter_shift"
    if mode == "dry_run":
        return f"hardware_policy_dry_run_{rule_prefix}parameter_shift"
    return f"hardware_policy_blocked_{rule_prefix}parameter_shift"


def _sample_provider(
    sampler: ProviderExpectationSampler,
    values: FloatArray,
    shots: int | None,
) -> ProviderExpectationSample:
    sample = sampler(values.copy(), shots)
    if not isinstance(sample, ProviderExpectationSample):
        raise ValueError("provider sampler must return ProviderExpectationSample")
    return sample.with_default_shots(shots)


def _validate_finite_shot_sample_provenance(
    direction: str,
    sample: ProviderExpectationSample,
) -> None:
    """Require sample-level provenance for finite-shot callback records."""
    metadata = sample.metadata or {}
    missing_keys = sorted(_FINITE_SHOT_SAMPLE_PROVENANCE_KEYS - metadata.keys())
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ValueError(
            f"finite-shot provider sample provenance missing {joined_keys} for {direction} sample"
        )
    source_class = metadata["source_class"]
    if not isinstance(source_class, str) or source_class not in _FINITE_SHOT_SAMPLE_SOURCE_CLASSES:
        allowed = ", ".join(sorted(_FINITE_SHOT_SAMPLE_SOURCE_CLASSES))
        raise ValueError(
            f"finite-shot provider sample provenance source_class must be one of {allowed}"
        )
    _normalise_provenance_token(
        f"finite-shot provider sample provenance {direction}.sample_seed",
        metadata["sample_seed"],
    )
    _normalise_provenance_token(
        f"finite-shot provider sample provenance {direction}.shot_batch_id",
        metadata["shot_batch_id"],
    )


def _attach_shift_sample_provenance(
    sample: ProviderExpectationSample,
    *,
    parameter_index: int,
    shift_index: int,
    direction: str,
    shift: float,
    coefficient: float,
    shifted_parameters: FloatArray,
    finite_shot: bool,
) -> ProviderExpectationSample:
    """Attach executor-owned shifted-parameter provenance to a sample."""
    return sample.with_metadata(
        {
            "sample_provenance_schema": _FINITE_SHOT_SAMPLE_PROVENANCE_SCHEMA
            if finite_shot
            else "provider_shift_sample_provenance_v1",
            "parameter_index": parameter_index,
            "shift_index": shift_index,
            "shift_direction": direction,
            "shift": shift,
            "coefficient": coefficient,
            "shifted_parameter_digest": _parameter_vector_digest(shifted_parameters),
        }
    )


def _normalise_provenance_token(name: str, value: object) -> str:
    """Return a non-empty provenance token from a JSON metadata value."""
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise ValueError(f"{name} must be a non-empty string or integer token")
    if not isinstance(value, str | int):
        raise ValueError(f"{name} must be a non-empty string or integer token")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _parameter_vector_digest(values: FloatArray) -> str:
    """Return a stable digest for a shifted parameter vector."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return f"sha256:{hashlib.sha256(contiguous.tobytes()).hexdigest()}"


def _standard_error(
    plus: ProviderExpectationSample,
    minus: ProviderExpectationSample,
    *,
    coefficient: float,
    require_variance: bool,
) -> float:
    if plus.variance is None or minus.variance is None:
        if require_variance:
            raise ValueError("finite-shot provider gradients require sample variance")
        return 0.0
    if plus.shots is None or minus.shots is None:
        if require_variance:
            raise ValueError("finite-shot provider gradients require sample shots")
        return 0.0
    return abs(coefficient) * float(
        np.sqrt((plus.variance / plus.shots) + (minus.variance / minus.shots))
    )


def _parameter_shift_terms(
    rule: ParameterShiftRule | None,
    shift: float,
) -> tuple[tuple[float, float], ...]:
    if rule is not None:
        return tuple(
            (
                _as_positive_scalar("rule.shift", shift_value),
                _as_finite_scalar("rule.coefficient", coefficient),
            )
            for shift_value, coefficient in rule.terms
        )
    shift_value = _as_positive_scalar("shift", shift)
    denominator = 2.0 * np.sin(shift_value)
    if abs(denominator) <= 1.0e-15:
        raise ValueError("shift must not make the parameter-shift denominator singular")
    return ((shift_value, float(1.0 / denominator)),)


def _result_method(plan_method: str, term_count: int) -> str:
    if term_count == 1:
        return plan_method
    if plan_method == "stochastic_parameter_shift":
        return "multi_frequency_stochastic_parameter_shift"
    if plan_method == "parameter_shift":
        return "multi_frequency_parameter_shift"
    return plan_method


def _as_finite_vector(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_non_negative_scalar(name: str, value: object) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _as_positive_scalar(name: str, value: object) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _normalise_shots(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalise_metadata(metadata: Mapping[str, object] | None) -> dict[str, JsonValue]:
    if metadata is None:
        return {}
    normalised: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        normalised[key] = _json_value(f"metadata[{key!r}]", value)
    return normalised


def _json_value(name: str, value: object) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, tuple | list):
        return [_json_value(f"{name}[]", item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            result[key] = _json_value(f"{name}.{key}", item)
        return result
    raise ValueError(f"{name} must be JSON-compatible")


__all__ = [
    "ProviderExpectationSample",
    "ProviderGradientExecutionResult",
    "ProviderParameterShiftRecord",
    "execute_provider_parameter_shift_gradient",
]
