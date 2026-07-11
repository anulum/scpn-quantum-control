# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow Compatibility and Maturity
"""Bounded TensorFlow compatibility, lowering, and maturity execution.

This one-way leaf owns GradientTape, ``tf.function``, XLA, Keras-layer,
Phase-QNode lowering-matrix, and maturity evidence. The public facade injects
its active optional-TensorFlow loader so fail-closed and monkeypatch behavior
stays stable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)
from .tensorflow_bridge_contracts import (
    FloatArray,
    PhaseTensorFlowFunctionCompatibilityResult,
    PhaseTensorFlowGradientTapeCompatibilityResult,
    PhaseTensorFlowKerasLayerWrapperAuditResult,
    PhaseTensorFlowMaturityAuditResult,
    PhaseTensorFlowPhaseQNodeLoweringMatrixResult,
    PhaseTensorFlowPhaseQNodeLoweringRoute,
    PhaseTensorFlowXLACompatibilityResult,
)
from .tensorflow_gradients import (
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _tensorflow_tensor,
    _tensorflow_values_to_numpy,
    tensorflow_bounded_qnn_value_and_grad,
)

TensorFlowLoader: TypeAlias = Callable[[], Any]


def run_tensorflow_phase_qnode_lowering_matrix() -> PhaseTensorFlowPhaseQNodeLoweringMatrixResult:
    """Return the TensorFlow parity matrix for registered Phase-QNode lowering.

    The current TensorFlow surface is implemented for bounded phase-QNN tensor,
    ``GradientTape``, ``tf.function``, XLA, and Keras-layer routes. Arbitrary
    registered Phase-QNode graph lowering remains blocked until native lowering
    rules, graph autodiff parity artefacts, provider safety evidence, hardware
    evidence, and isolated benchmark artefacts exist.
    """
    routes = (
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="bounded_qnn_analytic_tensor",
            status="passed",
            reason="bounded phase-QNN analytic TensorFlow tensor value-and-gradient is implemented",
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="bounded_qnn_gradient_tape",
            status="passed",
            reason="bounded phase-QNN GradientTape compatibility is implemented",
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="bounded_qnn_tf_function",
            status="passed",
            reason="bounded phase-QNN tf.function compatibility is implemented",
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="bounded_qnn_xla",
            status="passed",
            reason="bounded phase-QNN XLA compatibility is implemented",
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="bounded_qnn_keras_layer_wrapper",
            status="passed",
            reason="bounded TensorFlow Keras layer wrapper is implemented",
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_statevector_lowering",
            status="blocked",
            reason="arbitrary registered Phase-QNode circuits do not yet lower into TensorFlow graphs",
            requires=(
                "native_tensorflow_lowering_rules",
                "gate_observable_coverage_matrix",
                "statevector_gradient_parity_artifact",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_graph_lowering",
            status="blocked",
            reason="full graph autodiff through arbitrary simulators is not implemented",
            requires=(
                "graph_autodiff_contract",
                "simulator_lowering_rules",
                "gradient_tape_parity_artifact",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_finite_shot_lowering",
            status="blocked",
            reason="finite-shot TensorFlow lowering needs sampler, seed, and uncertainty provenance",
            requires=(
                "shot_policy",
                "rng_seed_provenance",
                "uncertainty_artifact",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_provider_lowering",
            status="blocked",
            reason="provider callbacks are not native TensorFlow graph-safe routes",
            requires=(
                "provider_allowlist",
                "callback_transform_safety_audit",
                "provider_execution_artifact",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_hardware_lowering",
            status="blocked",
            reason="live hardware TensorFlow lowering requires ticketed execution evidence",
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_dynamic_circuit_lowering",
            status="blocked",
            reason="mid-circuit measurement and feedback are outside the TensorFlow lowering boundary",
            requires=(
                "dynamic_circuit_semantics",
                "classical_feedback_contract",
                "gradient_policy",
            ),
        ),
        PhaseTensorFlowPhaseQNodeLoweringRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion requires isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseTensorFlowPhaseQNodeLoweringMatrixResult(routes=routes)


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


def _tensorflow_keras_layer_base_and_constant(tensorflow_module: Any) -> tuple[Any, Any]:
    keras = getattr(tensorflow_module, "keras", None)
    layers = getattr(keras, "layers", None)
    layer_base = getattr(layers, "Layer", None)
    initializers = getattr(keras, "initializers", None)
    constant = getattr(initializers, "Constant", None)
    if not isinstance(layer_base, type):
        raise RuntimeError("TensorFlow module does not expose tf.keras.layers.Layer")
    if not callable(constant):
        raise RuntimeError("TensorFlow module does not expose tf.keras.initializers.Constant")
    return layer_base, constant


def _tensorflow_trainable_parameter_count(layer: Any) -> int:
    variables = getattr(layer, "trainable_variables", ())
    count = 0
    for variable in variables:
        try:
            values = _tensorflow_values_to_numpy(variable)
        except ValueError:
            continue
        count += int(values.size)
    return count


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


def run_tensorflow_gradient_tape_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowGradientTapeCompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow ``GradientTape``.

    The audited route is the bounded classifier loss only. It does not expose
    arbitrary TensorFlow autodiff through SCPN simulator kernels or provider
    hardware execution.
    """
    tensorflow_module = _tensorflow_loader()
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
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowFunctionCompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow ``tf.function``.

    The traced route is the bounded classifier loss only. It does not claim XLA,
    Keras integration, arbitrary simulator tracing, or provider execution.
    """
    tensorflow_module = _tensorflow_loader()
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
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowXLACompatibilityResult:
    """Audit bounded phase-QNN compatibility with TensorFlow XLA JIT.

    This route requests ``tf.function(jit_compile=True)`` for the bounded
    classifier loss only. It does not claim general XLA lowering, arbitrary
    simulator tracing, provider execution, or production performance.
    """
    tensorflow_module = _tensorflow_loader()
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


def tensorflow_bounded_qnn_keras_layer(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
    _tensorflow_loader: TensorFlowLoader,
) -> Any:
    """Return a Keras ``Layer`` wrapper for the bounded phase-QNN loss."""
    tensorflow_module = _tensorflow_loader()
    layer_base, constant_initializer = _tensorflow_keras_layer_base_and_constant(
        tensorflow_module,
    )
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _tensorflow_values_to_numpy(initial_params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "initial_params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    feature_tensor = _tensorflow_tensor(tensorflow_module, feature_matrix)
    label_tensor = _tensorflow_tensor(tensorflow_module, label_vector)
    dtype = getattr(tensorflow_module, "float64", None)

    class _BoundedPhaseQNNKerasLayer(layer_base):  # type: ignore[misc, valid-type]  # dynamic optional Keras base
        def __init__(self) -> None:
            super().__init__()
            self.features = feature_tensor
            self.labels = label_tensor
            self.params = self.add_weight(
                name="params",
                shape=parameter_values.shape,
                initializer=constant_initializer(parameter_values),
                trainable=bool(trainable),
                dtype=dtype,
            )
            self.feature_width = int(feature_matrix.shape[1])
            self.host_boundary = False
            self.native_framework_autodiff = True
            self.claim_boundary = "bounded_tensorflow_keras_layer_wrapper"

        def call(self, params: Any | None = None) -> Any:
            parameter_source = self.params if params is None else params
            return _tensorflow_bounded_qnn_loss_tensor(
                tensorflow_module,
                self.features,
                self.labels,
                parameter_source,
            )

        def parameter_shift_gradient(self, params: Any | None = None) -> FloatArray:
            parameter_source = self.params if params is None else params
            raw_params = _as_parameter_vector(
                "TensorFlow bounded phase-QNN Keras parameters",
                _tensorflow_values_to_numpy(parameter_source),
                width=feature_matrix.shape[1],
            )
            reference_gradient = parameter_shift_qnn_classifier_gradient(
                feature_matrix,
                label_vector,
                raw_params,
            )
            return _as_parameter_vector(
                "SCPN bounded phase-QNN parameter-shift gradient",
                reference_gradient,
                width=feature_matrix.shape[1],
            )

    return _BoundedPhaseQNNKerasLayer()


def run_tensorflow_keras_layer_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowKerasLayerWrapperAuditResult:
    """Audit bounded phase-QNN TensorFlow Keras layer-wrapper gradients."""
    tensorflow_module = _tensorflow_loader()
    tape_factory = _tensorflow_gradient_tape(tensorflow_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    layer = tensorflow_bounded_qnn_keras_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=True,
        _tensorflow_loader=_tensorflow_loader,
    )
    tensorflow_params = layer.params
    with tape_factory() as tape:
        watch = getattr(tape, "watch", None)
        if callable(watch):
            watch(tensorflow_params)
        tensorflow_loss = layer()
    gradient_method = getattr(tape, "gradient", None)
    if not callable(gradient_method):
        raise RuntimeError("TensorFlow GradientTape does not expose gradient")
    tensorflow_gradient = gradient_method(tensorflow_loss, tensorflow_params)
    if tensorflow_gradient is None:
        raise RuntimeError("TensorFlow GradientTape returned no gradient")
    gradient = _as_parameter_vector(
        "TensorFlow Keras bounded phase-QNN gradient",
        _tensorflow_values_to_numpy(tensorflow_gradient),
        width=layer.feature_width,
    )
    parameter_shift_gradient = layer.parameter_shift_gradient(tensorflow_params)
    delta = gradient - parameter_shift_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTensorFlowKerasLayerWrapperAuditResult(
        loss=_tensorflow_values_to_float(tensorflow_loss),
        gradient=gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        tensorflow_layer=layer,
        tensorflow_loss=tensorflow_loss,
        tensorflow_gradient=tensorflow_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        keras_layer_supported=True,
        gradient_tape_supported=True,
        trainable_parameters=_tensorflow_trainable_parameter_count(layer),
    )


def run_tensorflow_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
    _tensorflow_loader: TensorFlowLoader,
) -> PhaseTensorFlowMaturityAuditResult:
    """Aggregate bounded TensorFlow evidence and provider-level blockers."""
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        _tensorflow_values_to_numpy(params),
        width=feature_matrix.shape[1],
    )

    analytic_tensor = tensorflow_bounded_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
        _tensorflow_loader=_tensorflow_loader,
    )
    gradient_tape = run_tensorflow_gradient_tape_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        _tensorflow_loader=_tensorflow_loader,
    )
    tf_function = run_tensorflow_function_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        _tensorflow_loader=_tensorflow_loader,
    )
    xla = run_tensorflow_xla_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        _tensorflow_loader=_tensorflow_loader,
    )
    keras_layer = run_tensorflow_keras_layer_wrapper_audit(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        tolerance=tolerance_value,
        _tensorflow_loader=_tensorflow_loader,
    )
    phase_qnode_lowering_matrix = run_tensorflow_phase_qnode_lowering_matrix()

    evidence: dict[str, object] = {
        "analytic_tensor": analytic_tensor,
        "gradient_tape": gradient_tape,
        "tf_function": tf_function,
        "xla": xla,
        "keras_layer": keras_layer,
        "phase_qnode_lowering_matrix": phase_qnode_lowering_matrix,
    }
    bounded_model_ready = all(
        result.passed
        for result in (
            analytic_tensor,
            gradient_tape,
            tf_function,
            xla,
            keras_layer,
        )
    )
    required_capabilities = {
        "analytic_tensor": "passed" if analytic_tensor.passed else "failed",
        "gradient_tape": "passed" if gradient_tape.passed else "failed",
        "tf_function": "passed" if tf_function.passed else "failed",
        "xla": "passed" if xla.passed else "failed",
        "keras_layer": "passed" if keras_layer.passed else "failed",
        "bounded_phase_qnode_lowering_matrix": (
            "passed" if phase_qnode_lowering_matrix.bounded_qnn_routes_ready else "failed"
        ),
        "arbitrary_phase_qnode_tensorflow_lowering": "blocked",
        "full_graph_autodiff_through_simulator": "blocked",
        "provider_callbacks": "blocked",
        "hardware_gradient_execution": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PhaseTensorFlowMaturityAuditResult(
        bounded_model_ready=bounded_model_ready,
        ready_for_provider_exceedance=bounded_model_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        open_gaps=open_gaps,
    )
