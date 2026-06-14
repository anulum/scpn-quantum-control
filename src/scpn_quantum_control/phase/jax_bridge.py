# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase JAX Bridge
"""Optional JAX interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    value_and_parameter_shift_grad,
)
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class PhaseJAXParameterShiftResult:
    """Result from the optional JAX phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    method: str
    evaluations: int
    jit_requested: bool
    jitted: bool
    host_callback: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable JAX interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "host_callback": self.host_callback,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PhaseJAXGradientAgreementResult:
    """Agreement report between SCPN and JAX-style gradient callables."""

    value: float
    scpn_gradient: FloatArray
    jax_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable JAX gradient agreement metadata."""
        return {
            "value": self.value,
            "scpn_gradient": self.scpn_gradient.tolist(),
            "jax_gradient": self.jax_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PhaseJAXNativeQNNGradientResult:
    """Native JAX autodiff agreement for the bounded phase-QNN classifier."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_requested: bool
    jitted: bool
    method: str = "jax_native_bounded_phase_qnn_value_and_grad"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready native JAX bounded-QNN gradient evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "method": self.method,
        }


@dataclass(frozen=True)
class PhaseJAXCustomVJPQNNGradientResult:
    """JAX custom-VJP evidence for the bounded phase-QNN classifier."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    custom_vjp: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_requested: bool
    jitted: bool
    method: str = "jax_custom_vjp_bounded_phase_qnn_value_and_grad"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready custom-VJP bounded-QNN gradient evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "custom_vjp": self.custom_vjp,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "method": self.method,
        }


def _load_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX is unavailable; install scpn-quantum-control[jax]") from exc
    return jax, jnp


def is_phase_jax_available() -> bool:
    """Return whether the optional phase JAX adapter can import JAX."""
    try:
        _load_jax()
    except ImportError:
        return False
    return True


def _as_parameter_vector(name: str, values: object, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_scalar(name: str, value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array or a single-column matrix")
    if vector.shape != (n_samples,):
        raise ValueError("features and labels must have the same sample count")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in the closed interval [0, 1]")
    return vector.astype(np.float64, copy=True)


def _require_jax_callback_support(jax_module: Any) -> None:
    if not hasattr(jax_module, "jit"):
        raise RuntimeError("JAX JIT is unavailable in the active JAX module")
    if not hasattr(jax_module, "pure_callback"):
        raise RuntimeError("JAX pure_callback is required for JIT-wrapped host gradients")
    if not hasattr(jax_module, "ShapeDtypeStruct"):
        raise RuntimeError("JAX ShapeDtypeStruct is required for JIT-wrapped host gradients")


def _require_jax_custom_vjp_support(jax_module: Any) -> None:
    custom_vjp = getattr(jax_module, "custom_vjp", None)
    if not callable(custom_vjp):
        raise RuntimeError("JAX custom_vjp is required for the bounded-QNN custom VJP route")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for the bounded-QNN custom VJP route")


def _jax_bounded_qnn_loss(
    jnp: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    probabilities = 0.5 * (1.0 - jnp.cos(feature_tensor + parameter_tensor[None, :]))
    predictions = jnp.mean(probabilities, axis=1)
    residual = predictions - label_tensor
    return jnp.mean(residual * residual)


def _jax_bounded_qnn_gradient(
    jnp: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    feature_count = feature_tensor.shape[1]
    probabilities = 0.5 * (1.0 - jnp.cos(feature_tensor + parameter_tensor[None, :]))
    predictions = jnp.mean(probabilities, axis=1)
    residual = predictions - label_tensor
    probability_derivative = 0.5 * jnp.sin(feature_tensor + parameter_tensor[None, :])
    prediction_derivative = probability_derivative / feature_count
    return jnp.mean(2.0 * residual[:, None] * prediction_derivative, axis=0)


def jax_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    jit: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXParameterShiftResult:
    """Return parameter-shift value and gradient through the optional JAX bridge.

    This adapter is an interop boundary, not a native quantum-kernel compiler.
    Without ``jit`` it imports JAX, accepts JAX-like inputs, and executes the
    repository's deterministic parameter-shift rule on the host. With ``jit`` it
    wraps that host execution in ``jax.pure_callback`` and ``jax.jit`` so JAX
    workflows can stage the call explicitly while provenance still reports
    ``host_callback=True``.
    """
    parameter_values = _as_parameter_vector("values", values)
    jax_module, jnp = _load_jax()
    shift_terms = len((rule or ParameterShiftRule()).terms)
    last_result: GradientResult | None = None

    def evaluate(raw_values: object) -> tuple[np.ndarray, np.ndarray]:
        nonlocal last_result
        raw_array = _as_parameter_vector(
            "JAX callback values", raw_values, width=parameter_values.size
        )
        result: GradientResult = value_and_parameter_shift_grad(
            objective,
            raw_array,
            parameters=parameters,
            rule=rule,
        )
        last_result = result
        return (
            np.asarray(result.value, dtype=np.float64),
            result.gradient.astype(np.float64, copy=False),
        )

    if jit:
        _require_jax_callback_support(jax_module)
        value_shape = jax_module.ShapeDtypeStruct((), np.dtype(np.float64))
        gradient_shape = jax_module.ShapeDtypeStruct(parameter_values.shape, np.dtype(np.float64))

        def wrapped(raw_values: object) -> tuple[object, object]:
            return cast(
                tuple[object, object],
                jax_module.pure_callback(
                    evaluate,
                    (value_shape, gradient_shape),
                    raw_values,
                ),
            )

        value_obj, gradient_obj = jax_module.jit(wrapped)(jnp.asarray(parameter_values))
        jitted = True
        host_callback = True
    else:
        value_obj, gradient_obj = evaluate(jnp.asarray(parameter_values))
        jitted = False
        host_callback = False

    gradient = _as_parameter_vector(
        "JAX parameter-shift gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    if last_result is None:
        raise RuntimeError("JAX parameter-shift callback did not execute")
    return PhaseJAXParameterShiftResult(
        value=_as_scalar("JAX parameter-shift value", value_obj),
        gradient=gradient,
        method=last_result.method,
        evaluations=last_result.evaluations,
        jit_requested=jit,
        jitted=jitted,
        host_callback=host_callback,
        shift_terms=shift_terms,
    )


def check_jax_parameter_shift_agreement(
    objective: ScalarObjective,
    jax_gradient: GradientCallable,
    values: ArrayLike,
    *,
    tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXGradientAgreementResult:
    """Compare SCPN parameter-shift gradients with a JAX-derived gradient callable.

    ``jax_gradient`` is caller-supplied so the bridge can verify agreement with
    ``jax.grad`` or equivalent JAX code without claiming automatic conversion of
    every SCPN objective into a native JAX quantum kernel.
    """
    _load_jax()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_gradient = _as_parameter_vector(
        "JAX gradient",
        jax_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXGradientAgreementResult(
        value=float(scpn.value),
        scpn_gradient=scpn.gradient.copy(),
        jax_gradient=external_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
    )


def jax_native_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXNativeQNNGradientResult:
    """Differentiate the bounded phase-QNN loss with native JAX autodiff.

    This route is intentionally narrower than arbitrary simulator autodiff. It
    expresses the repository's bounded phase-QNN classifier loss directly in
    JAX tensor operations, obtains a native ``value_and_grad`` result, and
    records agreement against the existing multi-frequency parameter-shift
    gradient. It does not use ``pure_callback`` and does not claim conversion of
    arbitrary quantum programs into JAX kernels.
    """
    jax_module, jnp = _load_jax()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    value_and_grad_fn = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad_fn):
        raise RuntimeError("JAX value_and_grad is required for native QNN autodiff")
    executable = value_and_grad_fn(loss_fn)
    jitted = False
    if jit:
        jit_fn = getattr(jax_module, "jit", None)
        if not callable(jit_fn):
            raise RuntimeError("JAX JIT is unavailable in the active JAX module")
        executable = jit_fn(executable)
        jitted = True

    loss_obj, gradient_obj = executable(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX native QNN gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    loss = _as_scalar("JAX native QNN loss", loss_obj)
    if abs(loss - reference_loss) > max(tolerance_value, 1e-12):
        raise RuntimeError("JAX native QNN loss disagrees with parameter-shift loss")
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXNativeQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


def jax_custom_vjp_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXCustomVJPQNNGradientResult:
    """Differentiate the bounded phase-QNN loss through a JAX custom VJP.

    The primal is the same bounded phase-QNN MSE loss used by
    ``jax_native_qnn_value_and_grad``. The VJP rule is registered explicitly
    and returns the mathematically equivalent bounded-QNN derivative, then the
    result is checked against the repository's multi-frequency parameter-shift
    reference. This route is still intentionally narrow: it does not expose
    arbitrary simulator autodiff, provider callbacks, or hardware execution.
    """
    jax_module, jnp = _load_jax()
    _require_jax_custom_vjp_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    @jax_module.custom_vjp
    def loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    def loss_fwd(raw_params: object) -> tuple[object, object]:
        parameter_tensor = jnp.asarray(raw_params)
        return (
            _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor),
            parameter_tensor,
        )

    def loss_bwd(parameter_tensor: object, cotangent: object) -> tuple[object]:
        gradient = _jax_bounded_qnn_gradient(
            jnp,
            feature_tensor,
            label_tensor,
            jnp.asarray(parameter_tensor),
        )
        return (jnp.asarray(cotangent) * gradient,)

    loss_fn.defvjp(loss_fwd, loss_bwd)

    executable = jax_module.value_and_grad(loss_fn)
    jitted = False
    if jit:
        jit_fn = getattr(jax_module, "jit", None)
        if not callable(jit_fn):
            raise RuntimeError("JAX JIT is unavailable in the active JAX module")
        executable = jit_fn(executable)
        jitted = True

    loss_obj, gradient_obj = executable(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX custom VJP QNN gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    loss = _as_scalar("JAX custom VJP QNN loss", loss_obj)
    if abs(loss - reference_loss) > max(tolerance_value, 1e-12):
        raise RuntimeError("JAX custom VJP QNN loss disagrees with parameter-shift loss")
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXCustomVJPQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        custom_vjp=True,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


__all__ = [
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXParameterShiftResult",
    "check_jax_parameter_shift_agreement",
    "is_phase_jax_available",
    "jax_custom_vjp_qnn_value_and_grad",
    "jax_native_qnn_value_and_grad",
    "jax_parameter_shift_value_and_grad",
]
