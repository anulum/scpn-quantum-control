# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider QNode Transforms
"""Provider-callback QNode transform execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import ParameterShiftRule
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderExpectationSampler,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]

EVIDENCE_CLASS = "provider_qnode_transform_execution"
CLAIM_BOUNDARY = (
    "provider callback QNode transform execution with explicit shifted-sample "
    "records, shot accounting, uncertainty propagation, and fail-closed hardware "
    "policy; not live hardware submission or framework-native autodiff"
)
_SUPPORTED_SCALAR_TRANSFORMS = {
    "grad",
    "value_and_grad",
    "jvp",
    "vjp",
    "jacfwd",
    "jacrev",
}
_ALIASES = {
    "value_grad": "value_and_grad",
    "value-and-grad": "value_and_grad",
    "jacobian_forward": "jacfwd",
    "jacobian_reverse": "jacrev",
}


@dataclass(frozen=True)
class ProviderQNodeTransformResult:
    """Execution or fail-closed evidence for one provider-callback QNode transform."""

    transform: str
    supported: bool
    params: FloatArray | None
    value: float | None
    gradient: FloatArray | None
    standard_error: FloatArray | None
    confidence_radius: FloatArray | None
    jvp: float | None
    vjp: FloatArray | None
    jacobian: FloatArray | None
    tangent: FloatArray | None
    cotangent: FloatArray | None
    batched_params: FloatArray | None
    batched_values: FloatArray | None
    batched_gradients: FloatArray | None
    batched_standard_error: FloatArray | None
    batched_confidence_radius: FloatArray | None
    provider_gradient_result: ProviderGradientExecutionResult | None
    provider_gradient_results: tuple[ProviderGradientExecutionResult, ...]
    total_evaluations: int
    total_shots: int | None
    failure_reason: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def fail_closed(self) -> bool:
        """Return true when execution was intentionally refused."""
        return not self.supported

    @property
    def parameter_shift_evaluations(self) -> int:
        """Return shifted provider callback evaluations, excluding baseline samples."""
        return self.total_evaluations

    @property
    def batch_size(self) -> int:
        """Return provider vmap batch size."""
        return 0 if self.batched_params is None else int(self.batched_params.shape[0])

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provider QNode transform evidence."""
        return {
            "transform": self.transform,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "params": None if self.params is None else self.params.tolist(),
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "standard_error": None
            if self.standard_error is None
            else self.standard_error.tolist(),
            "confidence_radius": None
            if self.confidence_radius is None
            else self.confidence_radius.tolist(),
            "jvp": self.jvp,
            "vjp": None if self.vjp is None else self.vjp.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "tangent": None if self.tangent is None else self.tangent.tolist(),
            "cotangent": None if self.cotangent is None else self.cotangent.tolist(),
            "batched_params": (
                None if self.batched_params is None else self.batched_params.tolist()
            ),
            "batched_values": (
                None if self.batched_values is None else self.batched_values.tolist()
            ),
            "batched_gradients": (
                None if self.batched_gradients is None else self.batched_gradients.tolist()
            ),
            "batched_standard_error": None
            if self.batched_standard_error is None
            else self.batched_standard_error.tolist(),
            "batched_confidence_radius": None
            if self.batched_confidence_radius is None
            else self.batched_confidence_radius.tolist(),
            "provider_gradient_result": None
            if self.provider_gradient_result is None
            else self.provider_gradient_result.to_dict(),
            "provider_gradient_results": [
                result.to_dict() for result in self.provider_gradient_results
            ],
            "total_evaluations": self.total_evaluations,
            "total_shots": self.total_shots,
            "failure_reason": self.failure_reason,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class ProviderQNodeTransformReadinessSuiteResult:
    """Readiness evidence for supported and blocked provider QNode transforms."""

    records: tuple[ProviderQNodeTransformResult, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def record_count(self) -> int:
        """Return total records."""
        return len(self.records)

    @property
    def supported_count(self) -> int:
        """Return supported record count."""
        return sum(1 for record in self.records if record.supported)

    @property
    def fail_closed_count(self) -> int:
        """Return fail-closed record count."""
        return sum(1 for record in self.records if record.fail_closed)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return shifted provider callback evaluations across supported records."""
        return sum(record.parameter_shift_evaluations for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when supported records execute and unsafe records fail closed."""
        supported = {record.transform for record in self.records if record.supported}
        return (
            {"grad", "value_and_grad", "jvp", "jacfwd", "vmap.grad"}.issubset(supported)
            and self.fail_closed_count >= 3
            and not self.hardware_execution
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready readiness evidence."""
        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "supported_count": self.supported_count,
            "fail_closed_count": self.fail_closed_count,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "records": [record.to_dict() for record in self.records],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


def execute_provider_qnode_transform(
    transform: str,
    sampler: ProviderExpectationSampler,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    cotangent: ArrayLike | float | None = None,
    backend: str = "statevector",
    shots: int | None = None,
    rule: ParameterShiftRule | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    allow_hardware: bool = False,
) -> ProviderQNodeTransformResult:
    """Execute a supported scalar provider-callback QNode transform."""
    label = _normalise_transform(transform)
    values = _as_parameter_vector("params", params)
    if label in {"hessian", "grad.grad"}:
        return _blocked_result(
            label,
            params=values,
            reason=(
                "provider QNode curvature transforms are not implemented; "
                "use deterministic local Hessian diagnostics"
            ),
        )
    if label not in _SUPPORTED_SCALAR_TRANSFORMS:
        return _blocked_result(
            label,
            params=values,
            reason="provider QNode transform is outside the bounded scalar callback algebra",
        )

    try:
        tangent_vector = (
            _as_parameter_vector("tangent", tangent, width=values.size) if label == "jvp" else None
        )
        cotangent_vector = _as_cotangent(cotangent) if label == "vjp" else None
        gradient_result = execute_provider_parameter_shift_gradient(
            sampler,
            values,
            backend=backend,
            shots=shots,
            rule=rule,
            confidence_level=confidence_level,
            confidence_z=confidence_z,
            allow_hardware=allow_hardware,
        )
        value_sample = (
            _sample_value(sampler, values, gradient_result.plan.shots)
            if label in {"value_and_grad", "jvp", "vjp"}
            else None
        )
    except ValueError as exc:
        return _blocked_result(label, params=values, reason=str(exc))

    gradient = gradient_result.gradient
    value = None if value_sample is None else value_sample.value
    return _supported_result(
        label,
        params=values,
        value=value,
        gradient=gradient,
        standard_error=gradient_result.standard_error,
        confidence_radius=gradient_result.confidence_radius,
        jvp=None if tangent_vector is None else float(np.dot(gradient, tangent_vector)),
        vjp=None if cotangent_vector is None else cotangent_vector[0] * gradient,
        jacobian=gradient.reshape(1, -1).astype(np.float64, copy=True)
        if label in {"jacfwd", "jacrev"}
        else None,
        tangent=tangent_vector,
        cotangent=cotangent_vector,
        provider_gradient_result=gradient_result,
        provider_gradient_results=(gradient_result,),
        total_evaluations=gradient_result.total_evaluations,
        total_shots=_add_optional_shots(gradient_result.total_shots, value_sample),
    )


def execute_provider_qnode_vmap_grad(
    sampler: ProviderExpectationSampler,
    batched_params: ArrayLike,
    *,
    backend: str = "statevector",
    shots: int | None = None,
    rule: ParameterShiftRule | None = None,
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    allow_hardware: bool = False,
) -> ProviderQNodeTransformResult:
    """Execute a host-side manual vmap over provider-callback scalar gradients."""
    batch = _as_parameter_matrix("batched_params", batched_params)
    values: list[float] = []
    gradients: list[FloatArray] = []
    standard_errors: list[FloatArray] = []
    confidence_radii: list[FloatArray] = []
    results: list[ProviderGradientExecutionResult] = []
    total_shots = 0
    saw_shots = False
    total_evaluations = 0
    try:
        for row in batch:
            row_params = row.astype(np.float64, copy=True)
            gradient_result = execute_provider_parameter_shift_gradient(
                sampler,
                row_params,
                backend=backend,
                shots=shots,
                rule=rule,
                confidence_level=confidence_level,
                confidence_z=confidence_z,
                allow_hardware=allow_hardware,
            )
            value_sample = _sample_value(sampler, row_params, gradient_result.plan.shots)
            values.append(value_sample.value)
            gradients.append(gradient_result.gradient)
            standard_errors.append(gradient_result.standard_error)
            confidence_radii.append(gradient_result.confidence_radius)
            results.append(gradient_result)
            total_evaluations += gradient_result.total_evaluations
            row_shots = _add_optional_shots(gradient_result.total_shots, value_sample)
            if row_shots is not None:
                saw_shots = True
                total_shots += row_shots
    except ValueError as exc:
        return _blocked_result(
            "vmap.grad",
            batched_params=batch,
            reason=f"provider vmap(grad) failed closed: {exc}",
        )

    return _supported_result(
        "vmap.grad",
        batched_params=batch,
        batched_values=np.asarray(values, dtype=np.float64),
        batched_gradients=np.vstack(gradients).astype(np.float64, copy=True),
        batched_standard_error=np.vstack(standard_errors).astype(np.float64, copy=True),
        batched_confidence_radius=np.vstack(confidence_radii).astype(np.float64, copy=True),
        provider_gradient_results=tuple(results),
        total_evaluations=total_evaluations,
        total_shots=total_shots if saw_shots else None,
    )


def run_provider_qnode_transform_readiness_suite() -> ProviderQNodeTransformReadinessSuiteResult:
    """Run provider-callback QNode transform readiness evidence."""
    params = np.array([0.2, -0.4], dtype=np.float64)
    batch = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.6]], dtype=np.float64)
    records = (
        execute_provider_qnode_transform("grad", _readiness_sampler, params),
        execute_provider_qnode_transform(
            "value_and_grad",
            _readiness_sampler,
            params,
            backend="qasm_simulator",
            shots=400,
        ),
        execute_provider_qnode_transform(
            "jvp",
            _readiness_sampler,
            params,
            tangent=np.array([0.5, -2.0], dtype=np.float64),
        ),
        execute_provider_qnode_transform("jacfwd", _readiness_sampler, params),
        execute_provider_qnode_vmap_grad(_readiness_sampler, batch),
        execute_provider_qnode_transform(
            "grad",
            _readiness_sampler,
            params,
            backend="ibm_quantum",
            shots=1024,
        ),
        execute_provider_qnode_transform("hessian", _readiness_sampler, params),
        execute_provider_qnode_vmap_grad(
            _missing_variance_sampler,
            batch[:2],
            backend="qasm_simulator",
            shots=256,
        ),
    )
    return ProviderQNodeTransformReadinessSuiteResult(records=records)


def _supported_result(
    transform: str,
    *,
    params: FloatArray | None = None,
    value: float | None = None,
    gradient: FloatArray | None = None,
    standard_error: FloatArray | None = None,
    confidence_radius: FloatArray | None = None,
    jvp: float | None = None,
    vjp: FloatArray | None = None,
    jacobian: FloatArray | None = None,
    tangent: FloatArray | None = None,
    cotangent: FloatArray | None = None,
    batched_params: FloatArray | None = None,
    batched_values: FloatArray | None = None,
    batched_gradients: FloatArray | None = None,
    batched_standard_error: FloatArray | None = None,
    batched_confidence_radius: FloatArray | None = None,
    provider_gradient_result: ProviderGradientExecutionResult | None = None,
    provider_gradient_results: tuple[ProviderGradientExecutionResult, ...] = (),
    total_evaluations: int,
    total_shots: int | None,
) -> ProviderQNodeTransformResult:
    return ProviderQNodeTransformResult(
        transform=transform,
        supported=True,
        params=None if params is None else params.astype(np.float64, copy=True),
        value=value,
        gradient=None if gradient is None else gradient.astype(np.float64, copy=True),
        standard_error=None
        if standard_error is None
        else standard_error.astype(np.float64, copy=True),
        confidence_radius=None
        if confidence_radius is None
        else confidence_radius.astype(np.float64, copy=True),
        jvp=jvp,
        vjp=None if vjp is None else vjp.astype(np.float64, copy=True),
        jacobian=None if jacobian is None else jacobian.astype(np.float64, copy=True),
        tangent=None if tangent is None else tangent.astype(np.float64, copy=True),
        cotangent=None if cotangent is None else cotangent.astype(np.float64, copy=True),
        batched_params=None
        if batched_params is None
        else batched_params.astype(np.float64, copy=True),
        batched_values=None
        if batched_values is None
        else batched_values.astype(np.float64, copy=True),
        batched_gradients=None
        if batched_gradients is None
        else batched_gradients.astype(np.float64, copy=True),
        batched_standard_error=None
        if batched_standard_error is None
        else batched_standard_error.astype(np.float64, copy=True),
        batched_confidence_radius=None
        if batched_confidence_radius is None
        else batched_confidence_radius.astype(np.float64, copy=True),
        provider_gradient_result=provider_gradient_result,
        provider_gradient_results=provider_gradient_results,
        total_evaluations=total_evaluations,
        total_shots=total_shots,
        failure_reason="",
    )


def _blocked_result(
    transform: str,
    *,
    params: FloatArray | None = None,
    batched_params: FloatArray | None = None,
    reason: str,
) -> ProviderQNodeTransformResult:
    return ProviderQNodeTransformResult(
        transform=transform,
        supported=False,
        params=None if params is None else params.copy(),
        value=None,
        gradient=None,
        standard_error=None,
        confidence_radius=None,
        jvp=None,
        vjp=None,
        jacobian=None,
        tangent=None,
        cotangent=None,
        batched_params=None if batched_params is None else batched_params.copy(),
        batched_values=None,
        batched_gradients=None,
        batched_standard_error=None,
        batched_confidence_radius=None,
        provider_gradient_result=None,
        provider_gradient_results=(),
        total_evaluations=0,
        total_shots=None,
        failure_reason=reason,
    )


def _sample_value(
    sampler: ProviderExpectationSampler,
    params: FloatArray,
    shots: int | None,
) -> ProviderExpectationSample:
    sample = sampler(params.copy(), shots)
    if not isinstance(sample, ProviderExpectationSample):
        raise ValueError("provider sampler must return ProviderExpectationSample")
    return sample.with_default_shots(shots)


def _add_optional_shots(
    gradient_shots: int | None,
    value_sample: ProviderExpectationSample | None,
) -> int | None:
    value_shots = None if value_sample is None else value_sample.shots
    if gradient_shots is None and value_shots is None:
        return None
    return (0 if gradient_shots is None else gradient_shots) + (
        0 if value_shots is None else value_shots
    )


def _normalise_transform(transform: str) -> str:
    key = transform.strip().lower().replace("-", "_").replace(".", "_")
    if not key:
        raise ValueError("transform must be non-empty")
    return _ALIASES.get(key, key)


def _as_parameter_vector(
    name: str,
    values: ArrayLike | None,
    *,
    width: int | None = None,
) -> FloatArray:
    if values is None:
        raise ValueError(f"{name} must be provided")
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _as_parameter_matrix(name: str, values: ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty batch and parameter axes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, matrix.astype(np.float64, copy=True))


def _as_cotangent(values: ArrayLike | float | None) -> FloatArray:
    if values is None:
        raise ValueError("cotangent must be provided")
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or vector.shape != (1,):
        raise ValueError("cotangent must be a scalar or one-element vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("cotangent must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _readiness_objective(params: FloatArray) -> float:
    return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))


def _readiness_sampler(params: FloatArray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=_readiness_objective(params),
        variance=None if shots is None else 0.04,
        shots=shots,
        metadata={
            "route": "provider_qnode_readiness",
            "sample_seed": "provider-qnode-readiness-seed",
            "shot_batch_id": "provider-qnode-readiness-batch",
            "source_class": "synthetic_fixture",
        },
    )


def _missing_variance_sampler(params: FloatArray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=_readiness_objective(params),
        shots=shots,
        metadata={
            "route": "provider_qnode_missing_variance",
            "sample_seed": "provider-qnode-missing-variance-seed",
            "shot_batch_id": "provider-qnode-missing-variance-batch",
            "source_class": "synthetic_fixture",
        },
    )


__all__ = [
    "ProviderQNodeTransformReadinessSuiteResult",
    "ProviderQNodeTransformResult",
    "execute_provider_qnode_transform",
    "execute_provider_qnode_vmap_grad",
    "run_provider_qnode_transform_readiness_suite",
]
