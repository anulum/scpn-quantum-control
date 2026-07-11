# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase TensorFlow Bridge
"""Signature-stable facade for optional TensorFlow phase-gradient routes.

Immutable records live in :mod:`.tensorflow_bridge_contracts`, bounded gradient
implementations live in :mod:`.tensorflow_gradients`, and compatibility,
lowering, Keras, and maturity execution lives in
:mod:`.tensorflow_compatibility`. This module retains optional loading and
injects its active loader through signature-stable public wrappers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from numpy.typing import ArrayLike

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
)
from .tensorflow_bridge_contracts import (
    FloatArray as FloatArray,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowFunctionCompatibilityResult as PhaseTensorFlowFunctionCompatibilityResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowGradientTapeCompatibilityResult as PhaseTensorFlowGradientTapeCompatibilityResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowKerasLayerWrapperAuditResult as PhaseTensorFlowKerasLayerWrapperAuditResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowMaturityAuditResult as PhaseTensorFlowMaturityAuditResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowParameterShiftResult as PhaseTensorFlowParameterShiftResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowPhaseQNodeLoweringMatrixResult as PhaseTensorFlowPhaseQNodeLoweringMatrixResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowPhaseQNodeLoweringRoute as PhaseTensorFlowPhaseQNodeLoweringRoute,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowQNNGradientResult as PhaseTensorFlowQNNGradientResult,
)
from .tensorflow_bridge_contracts import (
    PhaseTensorFlowXLACompatibilityResult as PhaseTensorFlowXLACompatibilityResult,
)
from .tensorflow_bridge_contracts import (
    _result_to_dict as _result_to_dict,
)
from .tensorflow_compatibility import (
    _tensorflow_bounded_qnn_loss_tensor as _tensorflow_bounded_qnn_loss_tensor,
)
from .tensorflow_compatibility import (
    _tensorflow_function as _tensorflow_function,
)
from .tensorflow_compatibility import (
    _tensorflow_gradient_tape as _tensorflow_gradient_tape,
)
from .tensorflow_compatibility import (
    _tensorflow_keras_layer_base_and_constant as _tensorflow_keras_layer_base_and_constant,
)
from .tensorflow_compatibility import (
    _tensorflow_trainable_parameter_count as _tensorflow_trainable_parameter_count,
)
from .tensorflow_compatibility import (
    _tensorflow_values_to_float as _tensorflow_values_to_float,
)
from .tensorflow_compatibility import (
    _tensorflow_variable as _tensorflow_variable,
)
from .tensorflow_compatibility import (
    run_tensorflow_function_compatibility_audit as _run_tensorflow_function_compatibility_audit,
)
from .tensorflow_compatibility import (
    run_tensorflow_gradient_tape_compatibility_audit as _run_tensorflow_gradient_tape_compatibility_audit,
)
from .tensorflow_compatibility import (
    run_tensorflow_keras_layer_wrapper_audit as _run_tensorflow_keras_layer_wrapper_audit,
)
from .tensorflow_compatibility import (
    run_tensorflow_maturity_audit as _run_tensorflow_maturity_audit,
)
from .tensorflow_compatibility import (
    run_tensorflow_phase_qnode_lowering_matrix as _run_tensorflow_phase_qnode_lowering_matrix,
)
from .tensorflow_compatibility import (
    run_tensorflow_xla_compatibility_audit as _run_tensorflow_xla_compatibility_audit,
)
from .tensorflow_compatibility import (
    tensorflow_bounded_qnn_keras_layer as _tensorflow_bounded_qnn_keras_layer,
)
from .tensorflow_gradients import (
    _as_feature_matrix as _as_feature_matrix,
)
from .tensorflow_gradients import (
    _as_label_vector as _as_label_vector,
)
from .tensorflow_gradients import (
    _as_non_negative_tolerance as _as_non_negative_tolerance,
)
from .tensorflow_gradients import (
    _as_parameter_vector as _as_parameter_vector,
)
from .tensorflow_gradients import (
    _tensorflow_tensor as _tensorflow_tensor,
)
from .tensorflow_gradients import (
    _tensorflow_values_to_numpy as _tensorflow_values_to_numpy,
)
from .tensorflow_gradients import (
    tensorflow_bounded_qnn_value_and_grad as _tensorflow_bounded_qnn_value_and_grad,
)
from .tensorflow_gradients import (
    tensorflow_parameter_shift_value_and_grad as _tensorflow_parameter_shift_value_and_grad,
)

ScalarObjective = Callable[[FloatArray], float]


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


def run_tensorflow_phase_qnode_lowering_matrix() -> PhaseTensorFlowPhaseQNodeLoweringMatrixResult:
    """Return the TensorFlow parity matrix for registered Phase-QNode lowering.

    The current TensorFlow surface is implemented for bounded phase-QNN tensor,
    ``GradientTape``, ``tf.function``, XLA, and Keras-layer routes. Arbitrary
    registered Phase-QNode graph lowering remains blocked until native lowering
    rules, graph autodiff parity artefacts, provider safety evidence, hardware
    evidence, and isolated benchmark artefacts exist.
    """
    return _run_tensorflow_phase_qnode_lowering_matrix()


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
    return _tensorflow_parameter_shift_value_and_grad(
        objective,
        values,
        parameters=parameters,
        rule=rule,
        _tensorflow_loader=_load_tensorflow,
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
    return _tensorflow_bounded_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
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
    return _run_tensorflow_gradient_tape_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
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
    return _run_tensorflow_function_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
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
    return _run_tensorflow_xla_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
    )


def tensorflow_bounded_qnn_keras_layer(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
) -> Any:
    """Return a Keras ``Layer`` wrapper for the bounded phase-QNN loss."""
    return _tensorflow_bounded_qnn_keras_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
        _tensorflow_loader=_load_tensorflow,
    )


def run_tensorflow_keras_layer_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowKerasLayerWrapperAuditResult:
    """Audit bounded phase-QNN TensorFlow Keras layer-wrapper gradients."""
    return _run_tensorflow_keras_layer_wrapper_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
    )


def run_tensorflow_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTensorFlowMaturityAuditResult:
    """Aggregate bounded TensorFlow evidence and provider-level blockers."""
    return _run_tensorflow_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=tolerance,
        _tensorflow_loader=_load_tensorflow,
    )


__all__ = [
    "PhaseTensorFlowFunctionCompatibilityResult",
    "PhaseTensorFlowGradientTapeCompatibilityResult",
    "PhaseTensorFlowKerasLayerWrapperAuditResult",
    "PhaseTensorFlowMaturityAuditResult",
    "PhaseTensorFlowParameterShiftResult",
    "PhaseTensorFlowPhaseQNodeLoweringMatrixResult",
    "PhaseTensorFlowPhaseQNodeLoweringRoute",
    "PhaseTensorFlowQNNGradientResult",
    "PhaseTensorFlowXLACompatibilityResult",
    "is_phase_tensorflow_available",
    "run_tensorflow_function_compatibility_audit",
    "run_tensorflow_gradient_tape_compatibility_audit",
    "run_tensorflow_keras_layer_wrapper_audit",
    "run_tensorflow_maturity_audit",
    "run_tensorflow_phase_qnode_lowering_matrix",
    "run_tensorflow_xla_compatibility_audit",
    "tensorflow_bounded_qnn_value_and_grad",
    "tensorflow_bounded_qnn_keras_layer",
    "tensorflow_parameter_shift_value_and_grad",
]
