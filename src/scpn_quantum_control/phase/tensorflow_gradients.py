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
from ..differentiable_tensor_admission import (
    AdmissionPrimitives as _AdmissionPrimitives,
)
from ..differentiable_tensor_admission import (
    require_nonnegative_tolerance as _require_nonnegative_tolerance,
)
from ..differentiable_tensor_admission import (
    require_real_parameter_vector as _require_real_parameter_vector,
)
from ..differentiable_tensor_construction import (
    TensorConstructionPrimitives as _TensorConstructionPrimitives,
)
from ..differentiable_tensor_construction import (
    construct_tensorflow_tensor as _construct_tensorflow_tensor,
)
from ..differentiable_tensor_materialization import (
    TensorMaterializationPrimitives as _TensorMaterializationPrimitives,
)
from ..differentiable_tensor_materialization import (
    materialize_tensorflow_host_values as _materialize_tensorflow_host_values,
)
from ..phase_qnn_framework_gradients import QNNGradientOperations as _QNNGradientOperations
from ..phase_qnn_framework_gradients import (
    tensorflow_bounded_qnn_audit as _tensorflow_bounded_qnn_audit,
)
from ..phase_qnn_model_numerics import PhaseModelNumerics as _PhaseModelNumerics
from ..phase_qnn_numpy_model import (
    numpy_phase_classifier_loss_and_gradient as _phase_qnn_analytic_model,
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
    vector = _require_real_parameter_vector(
        name,
        values,
        width=width,
        primitives=_AdmissionPrimitives(
            numpy=lambda: np, scalar=lambda: float, invalid=lambda: ValueError
        ),
    )
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
    return _require_nonnegative_tolerance(
        value,
        error_text="tolerance must be a non-negative finite float",
        negative_first=False,
        primitives=_AdmissionPrimitives(
            numpy=lambda: np, scalar=lambda: float, invalid=lambda: ValueError
        ),
    )


def _tensorflow_values_to_numpy(values: object) -> FloatArray:
    candidate = _materialize_tensorflow_host_values(
        values,
        primitives=_TensorMaterializationPrimitives(
            attribute=lambda: getattr, predicate=lambda: callable
        ),
    )
    return _as_parameter_vector("values", candidate)


def _tensorflow_tensor(tensorflow_module: Any, values: object) -> Any:
    return _construct_tensorflow_tensor(
        tensorflow_module,
        values,
        primitives=_TensorConstructionPrimitives(
            attribute=lambda: getattr, predicate=lambda: callable, invalid=lambda: RuntimeError
        ),
    )


def _tensorflow_qnn_routes() -> _QNNGradientOperations:
    """Return QNN gradient resolvers reading this module's live names."""
    return _QNNGradientOperations(
        feature_matrix=lambda: _as_feature_matrix,
        label_vector=lambda: _as_label_vector,
        parameter_vector=lambda: _as_parameter_vector,
        tolerance=lambda: _as_non_negative_tolerance,
        analytic_model=lambda: (
            lambda features, labels, params: _phase_qnn_analytic_model(
                features,
                labels,
                params,
                numerics=_PhaseModelNumerics(numpy=lambda: np, scalar=lambda: float),
            )
        ),
        reference_loss=lambda: parameter_shift_qnn_classifier_loss,
        reference_gradient=lambda: parameter_shift_qnn_classifier_gradient,
        tensor=lambda: _tensorflow_tensor,
    )


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

    return _tensorflow_bounded_qnn_audit(
        tensorflow_module,
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance_value,
        _tensorflow_qnn_routes(),
    )
