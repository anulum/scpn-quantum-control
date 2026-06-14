# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase TensorFlow Bridge
"""Optional TensorFlow interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

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


@dataclass(frozen=True)
class PhaseTensorFlowParameterShiftResult:
    """Result from the optional TensorFlow phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    tensorflow_value: Any
    tensorflow_gradient: Any
    method: str
    evaluations: int
    host_boundary: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "host_boundary": self.host_boundary,
            "shift_terms": self.shift_terms,
            "tensorflow_value_type": type(self.tensorflow_value).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowQNNGradientResult:
    """Tensor-ready bounded phase-QNN gradient evidence for TensorFlow workflows."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    tensorflow_parameter_shift_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    method: str = "tensorflow_bounded_phase_qnn_analytic_value_and_grad"
    host_boundary: bool = False
    native_framework_autodiff: bool = False
    analytic_framework_gradient: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow bounded-QNN gradient metadata."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "method": self.method,
            "host_boundary": self.host_boundary,
            "native_framework_autodiff": self.native_framework_autodiff,
            "analytic_framework_gradient": self.analytic_framework_gradient,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
            "tensorflow_parameter_shift_gradient_type": type(
                self.tensorflow_parameter_shift_gradient,
            ).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowGradientTapeCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow ``GradientTape``."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    gradient_tape_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_gradient_tape_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow ``GradientTape`` evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "gradient_tape_supported": self.gradient_tape_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowFunctionCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow ``tf.function``."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    function_supported: bool
    gradient_tape_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_function_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow ``tf.function`` evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "function_supported": self.function_supported,
            "gradient_tape_supported": self.gradient_tape_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowXLACompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow XLA JIT."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    function_supported: bool
    gradient_tape_supported: bool
    xla_compile_requested: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_xla_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow XLA compatibility evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "function_supported": self.function_supported,
            "gradient_tape_supported": self.gradient_tape_supported,
            "xla_compile_requested": self.xla_compile_requested,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


def _load_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is unavailable; install scpn-quantum-control[tensorflow]"
        ) from exc
    return tf


def is_phase_tensorflow_available() -> bool:
    """Return whether the optional phase TensorFlow bridge can import TensorFlow."""
    try:
        _load_tensorflow()
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


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if vector.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a non-negative finite float")
    return tolerance


def _tensorflow_values_to_numpy(values: object) -> FloatArray:
    candidate = values
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return _as_parameter_vector("values", candidate)


def _tensorflow_tensor(tensorflow_module: Any, values: object) -> Any:
    convert = getattr(tensorflow_module, "convert_to_tensor", None)
    if not callable(convert):
        raise RuntimeError("TensorFlow module does not expose convert_to_tensor")
    dtype = getattr(tensorflow_module, "float64", None)
    if dtype is None:
        return convert(values)
    return convert(values, dtype=dtype)


def _tensorflow_variable(tensorflow_module: Any, values: object) -> Any:
    variable = getattr(tensorflow_module, "Variable", None)
    if not callable(variable):
        raise RuntimeError("TensorFlow module does not expose Variable")
    dtype = getattr(tensorflow_module, "float64", None)
    if dtype is None:
        return variable(values)
    return variable(values, dtype=dtype)


def _tensorflow_gradient_tape(tensorflow_module: Any) -> Any:
    tape = getattr(tensorflow_module, "GradientTape", None)
    if not callable(tape):
        raise RuntimeError("TensorFlow module does not expose GradientTape")
    return tape


def _tensorflow_function(tensorflow_module: Any) -> Any:
    function = getattr(tensorflow_module, "function", None)
    if not callable(function):
        raise RuntimeError("TensorFlow module does not expose tf.function")
    return function


def _tensorflow_values_to_float(values: object) -> float:
    candidate = values
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    scalar = np.asarray(candidate, dtype=float)
    if scalar.shape not in ((), (1,)):
        raise ValueError(f"TensorFlow scalar value must be scalar-like, got {scalar.shape}")
    value = float(scalar.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("TensorFlow scalar value must be finite")
    return value


def _tensorflow_bounded_qnn_loss_tensor(
    tensorflow_module: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    cos = getattr(tensorflow_module, "cos", None)
    reduce_mean = getattr(tensorflow_module, "reduce_mean", None)
    if not callable(cos) or not callable(reduce_mean):
        raise RuntimeError("TensorFlow module does not expose cos and reduce_mean")
    shifted = feature_tensor + parameter_tensor
    probabilities = 0.5 * (1.0 - cos(shifted))
    predictions = reduce_mean(probabilities, axis=1)
    residual = predictions - label_tensor
    return reduce_mean(residual * residual)


def tensorflow_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike | object,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseTensorFlowParameterShiftResult:
    """Return phase parameter-shift value and gradient as NumPy and TensorFlow tensors.

    This is a host-boundary bridge. The quantum expectation is evaluated through
    SCPN's deterministic parameter-shift rule and the result is converted to
    TensorFlow tensors for framework pipelines. It does not claim native
    TensorFlow autodiff through a quantum simulator.
    """
    tensorflow_module = _load_tensorflow()
    parameter_values = _tensorflow_values_to_numpy(values)
    result: GradientResult = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    gradient = _as_parameter_vector(
        "TensorFlow parameter-shift gradient",
        result.gradient,
        width=parameter_values.size,
    )
    return PhaseTensorFlowParameterShiftResult(
        value=float(result.value),
        gradient=gradient,
        tensorflow_value=_tensorflow_tensor(
            tensorflow_module,
            np.asarray(result.value, dtype=np.float64),
        ),
        tensorflow_gradient=_tensorflow_tensor(tensorflow_module, gradient),
        method=result.method,
        evaluations=result.evaluations,
        host_boundary=True,
        shift_terms=shift_terms,
    )


def tensorflow_bounded_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowQNNGradientResult:
    """Return bounded phase-QNN loss and gradient as NumPy plus TensorFlow tensors.

    This route is narrower than arbitrary TensorFlow autodiff through a quantum
    simulator. It evaluates the bounded classifier's analytic tensor-gradient
    formula and verifies it against the canonical SCPN parameter-shift gradient
    before returning TensorFlow tensors.
    """
    tensorflow_module = _load_tensorflow()
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _tensorflow_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)

    shifted = feature_matrix + parameter_values[None, :]
    probabilities = 0.5 * (1.0 - np.cos(shifted))
    predictions = np.mean(probabilities, axis=1)
    residual = predictions - label_vector
    loss = float(np.mean(residual * residual))
    scale = 1.0 / float(feature_matrix.shape[1])
    gradient = (2.0 / float(feature_matrix.shape[0])) * np.sum(
        residual[:, None] * (0.5 * np.sin(shifted) * scale),
        axis=0,
    )
    gradient = _as_parameter_vector(
        "TensorFlow bounded phase-QNN gradient",
        gradient,
        width=parameter_values.size,
    )

    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    if abs(loss - reference_loss) > tolerance_value:
        raise RuntimeError(
            "TensorFlow bounded phase-QNN tensor loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        reference_gradient,
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    passed = bool(max_abs_error <= tolerance_value)
    return PhaseTensorFlowQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        tensorflow_loss=_tensorflow_tensor(tensorflow_module, np.asarray(loss, dtype=np.float64)),
        tensorflow_gradient=_tensorflow_tensor(tensorflow_module, gradient),
        tensorflow_parameter_shift_gradient=_tensorflow_tensor(
            tensorflow_module,
            reference_gradient,
        ),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=passed,
    )


def run_tensorflow_gradient_tape_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowGradientTapeCompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow ``GradientTape``.

    The audited route is the bounded classifier loss only. It does not expose
    arbitrary TensorFlow autodiff through SCPN simulator kernels or provider
    hardware execution.
    """
    tensorflow_module = _load_tensorflow()
    tape_factory = _tensorflow_gradient_tape(tensorflow_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _tensorflow_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _tensorflow_tensor(tensorflow_module, feature_matrix)
    label_tensor = _tensorflow_tensor(tensorflow_module, label_vector)
    parameter_tensor = _tensorflow_variable(tensorflow_module, parameter_values)
    with tape_factory() as tape:
        watch = getattr(tape, "watch", None)
        if callable(watch):
            watch(parameter_tensor)
        tensorflow_loss = _tensorflow_bounded_qnn_loss_tensor(
            tensorflow_module,
            feature_tensor,
            label_tensor,
            parameter_tensor,
        )
    gradient_method = getattr(tape, "gradient", None)
    if not callable(gradient_method):
        raise RuntimeError("TensorFlow GradientTape does not expose gradient")
    tensorflow_gradient = gradient_method(tensorflow_loss, parameter_tensor)
    if tensorflow_gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    gradient = _as_parameter_vector(
        "TensorFlow GradientTape bounded phase-QNN gradient",
        _tensorflow_values_to_numpy(tensorflow_gradient),
        width=parameter_values.size,
    )
    loss = _tensorflow_values_to_float(tensorflow_loss)
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    if abs(loss - reference_loss) > tolerance_value:
        raise RuntimeError(
            "TensorFlow GradientTape bounded phase-QNN loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient_values = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        reference_gradient_values,
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTensorFlowGradientTapeCompatibilityResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        tensorflow_loss=tensorflow_loss,
        tensorflow_gradient=tensorflow_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        gradient_tape_supported=True,
    )


def run_tensorflow_function_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowFunctionCompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow ``tf.function``.

    The traced route is the bounded classifier loss only. It does not claim XLA,
    Keras integration, arbitrary simulator tracing, or provider execution.
    """
    tensorflow_module = _load_tensorflow()
    function = _tensorflow_function(tensorflow_module)
    tape_factory = _tensorflow_gradient_tape(tensorflow_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _tensorflow_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _tensorflow_tensor(tensorflow_module, feature_matrix)
    label_tensor = _tensorflow_tensor(tensorflow_module, label_vector)
    parameter_tensor = _tensorflow_variable(tensorflow_module, parameter_values)

    def loss_fn(candidate_params: object) -> object:
        return _tensorflow_bounded_qnn_loss_tensor(
            tensorflow_module,
            feature_tensor,
            label_tensor,
            candidate_params,
        )

    traced_loss_fn = function(loss_fn)
    with tape_factory() as tape:
        watch = getattr(tape, "watch", None)
        if callable(watch):
            watch(parameter_tensor)
        tensorflow_loss = traced_loss_fn(parameter_tensor)
    gradient_method = getattr(tape, "gradient", None)
    if not callable(gradient_method):
        raise RuntimeError("TensorFlow GradientTape does not expose gradient")
    tensorflow_gradient = gradient_method(tensorflow_loss, parameter_tensor)
    if tensorflow_gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    gradient = _as_parameter_vector(
        "TensorFlow tf.function bounded phase-QNN gradient",
        _tensorflow_values_to_numpy(tensorflow_gradient),
        width=parameter_values.size,
    )
    loss = _tensorflow_values_to_float(tensorflow_loss)
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    if abs(loss - reference_loss) > tolerance_value:
        raise RuntimeError(
            "TensorFlow tf.function bounded phase-QNN loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_qnn_classifier_gradient(
            feature_matrix,
            label_vector,
            parameter_values,
        ),
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTensorFlowFunctionCompatibilityResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        tensorflow_loss=tensorflow_loss,
        tensorflow_gradient=tensorflow_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        function_supported=True,
        gradient_tape_supported=True,
    )


def run_tensorflow_xla_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowXLACompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow XLA JIT.

    This route requests ``tf.function(jit_compile=True)`` for the bounded
    classifier loss only. It does not claim general XLA lowering, arbitrary
    simulator tracing, provider execution, or production performance.
    """
    tensorflow_module = _load_tensorflow()
    function = _tensorflow_function(tensorflow_module)
    tape_factory = _tensorflow_gradient_tape(tensorflow_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _tensorflow_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _tensorflow_tensor(tensorflow_module, feature_matrix)
    label_tensor = _tensorflow_tensor(tensorflow_module, label_vector)
    parameter_tensor = _tensorflow_variable(tensorflow_module, parameter_values)

    def loss_fn(candidate_params: object) -> object:
        return _tensorflow_bounded_qnn_loss_tensor(
            tensorflow_module,
            feature_tensor,
            label_tensor,
            candidate_params,
        )

    try:
        xla_loss_fn = function(loss_fn, jit_compile=True)
    except TypeError as exc:
        raise RuntimeError("TensorFlow tf.function does not accept jit_compile") from exc
    with tape_factory() as tape:
        watch = getattr(tape, "watch", None)
        if callable(watch):
            watch(parameter_tensor)
        tensorflow_loss = xla_loss_fn(parameter_tensor)
    gradient_method = getattr(tape, "gradient", None)
    if not callable(gradient_method):
        raise RuntimeError("TensorFlow GradientTape does not expose gradient")
    tensorflow_gradient = gradient_method(tensorflow_loss, parameter_tensor)
    if tensorflow_gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    gradient = _as_parameter_vector(
        "TensorFlow XLA bounded phase-QNN gradient",
        _tensorflow_values_to_numpy(tensorflow_gradient),
        width=parameter_values.size,
    )
    loss = _tensorflow_values_to_float(tensorflow_loss)
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    if abs(loss - reference_loss) > tolerance_value:
        raise RuntimeError(
            "TensorFlow XLA bounded phase-QNN loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_qnn_classifier_gradient(
            feature_matrix,
            label_vector,
            parameter_values,
        ),
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTensorFlowXLACompatibilityResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        tensorflow_loss=tensorflow_loss,
        tensorflow_gradient=tensorflow_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        function_supported=True,
        gradient_tape_supported=True,
        xla_compile_requested=True,
    )


__all__ = [
    "PhaseTensorFlowFunctionCompatibilityResult",
    "PhaseTensorFlowGradientTapeCompatibilityResult",
    "PhaseTensorFlowParameterShiftResult",
    "PhaseTensorFlowQNNGradientResult",
    "PhaseTensorFlowXLACompatibilityResult",
    "is_phase_tensorflow_available",
    "run_tensorflow_function_compatibility_audit",
    "run_tensorflow_gradient_tape_compatibility_audit",
    "run_tensorflow_xla_compatibility_audit",
    "tensorflow_bounded_qnn_value_and_grad",
    "tensorflow_parameter_shift_value_and_grad",
]
