# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow Bounded Gradients
"""Bounded TensorFlow gradient execution and direct validation primitives.

This one-way leaf owns host-boundary parameter-shift and analytic bounded-QNN
gradient execution. The public compatibility facade injects its active optional
TensorFlow loader so its fail-closed and monkeypatch behavior stays stable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

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
from .tensorflow_bridge_contracts import (
    FloatArray,
    PhaseTensorFlowParameterShiftResult,
    PhaseTensorFlowQNNGradientResult,
)

ScalarObjective = Callable[[FloatArray], float]
TensorFlowLoader: TypeAlias = Callable[[], Any]


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


def tensorflow_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike | object,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowParameterShiftResult:
    """Return phase parameter-shift value and gradient as NumPy and TensorFlow tensors.

    This is a host-boundary bridge. The quantum expectation is evaluated through
    SCPN's deterministic parameter-shift rule and the result is converted to
    TensorFlow tensors for framework pipelines. It does not claim native
    TensorFlow autodiff through a quantum simulator.
    """
    tensorflow_module = _tensorflow_loader()
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
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowQNNGradientResult:
    """Return bounded phase-QNN loss and gradient as NumPy plus TensorFlow tensors.

    This route is narrower than arbitrary TensorFlow autodiff through a quantum
    simulator. It evaluates the bounded classifier's analytic tensor-gradient
    formula and verifies it against the canonical SCPN parameter-shift gradient
    before returning TensorFlow tensors.
    """
    tensorflow_module = _tensorflow_loader()
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
