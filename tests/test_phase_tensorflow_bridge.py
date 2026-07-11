# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# SCPN Quantum Control — TensorFlow Phase Bridge Tests

"""Contract tests for TensorFlow phase-gradient and QNode bridge integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.tensorflow_bridge as tensorflow_bridge
from scpn_quantum_control.phase import (
    PhaseTensorFlowMaturityAuditResult,
    PhaseTensorFlowPhaseQNodeLoweringMatrixResult,
    PhaseTensorFlowQNNGradientResult,
    is_phase_tensorflow_available,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    run_tensorflow_keras_layer_wrapper_audit,
    run_tensorflow_maturity_audit,
    run_tensorflow_phase_qnode_lowering_matrix,
    tensorflow_bounded_qnn_keras_layer,
    tensorflow_bounded_qnn_value_and_grad,
    tensorflow_parameter_shift_value_and_grad,
)

FloatArray = NDArray[np.float64]


class _FakeTensorFlowTensor:
    """NumPy-backed tensor carrying explicit derivative rows."""

    def __init__(
        self,
        values: object,
        *,
        derivative: FloatArray | None = None,
    ) -> None:
        """Initialize deterministic state for the fake framework object."""
        if isinstance(values, _FakeTensorFlowTensor):
            if derivative is None:
                derivative = values.derivative()
            values = values.numpy()
        self._values = np.asarray(values, dtype=float)
        self._derivative = None if derivative is None else np.asarray(derivative, dtype=float)

    def numpy(self) -> FloatArray:
        """Return the tensor payload as a NumPy array."""
        return self._values.copy()

    def derivative(self) -> FloatArray | None:
        """Return a copy of the tensor's derivative rows."""
        if self._derivative is None:
            return None
        return self._derivative.copy()

    @staticmethod
    def _with_derivative(
        values: FloatArray, derivative: FloatArray | None
    ) -> _FakeTensorFlowTensor:
        """Create a fake tensor with explicit derivative rows."""
        return _FakeTensorFlowTensor(values, derivative=derivative)

    @staticmethod
    def _broadcast_derivative(
        derivative: FloatArray | None,
        *,
        target_shape: tuple[int, ...],
        width: int,
    ) -> FloatArray:
        """Broadcast derivative rows to the requested output shape."""
        if derivative is None:
            return np.zeros((*target_shape, width), dtype=float)
        return np.broadcast_to(derivative, (*target_shape, width)).copy()

    @staticmethod
    def _derivative_width(
        left: FloatArray | None,
        right: FloatArray | None,
    ) -> int | None:
        """Return the active derivative-vector width."""
        if left is not None:
            return int(left.shape[-1])
        if right is not None:
            return int(right.shape[-1])
        return None

    def __add__(self, other: object) -> _FakeTensorFlowTensor:
        """Apply elementwise addition with NumPy broadcasting."""
        rhs = _FakeTensorFlowTensor(other)
        values = self._values + rhs.numpy()
        width = self._derivative_width(self._derivative, rhs._derivative)
        derivative = None
        if width is not None:
            derivative = self._broadcast_derivative(
                self._derivative,
                target_shape=values.shape,
                width=width,
            ) + self._broadcast_derivative(
                rhs._derivative,
                target_shape=values.shape,
                width=width,
            )
        return self._with_derivative(values, derivative)

    __radd__ = __add__

    def __sub__(self, other: object) -> _FakeTensorFlowTensor:
        """Apply elementwise subtraction with NumPy broadcasting."""
        rhs = _FakeTensorFlowTensor(other)
        values = self._values - rhs.numpy()
        width = self._derivative_width(self._derivative, rhs._derivative)
        derivative = None
        if width is not None:
            derivative = self._broadcast_derivative(
                self._derivative,
                target_shape=values.shape,
                width=width,
            ) - self._broadcast_derivative(
                rhs._derivative,
                target_shape=values.shape,
                width=width,
            )
        return self._with_derivative(values, derivative)

    def __rsub__(self, other: object) -> _FakeTensorFlowTensor:
        """Apply reflected elementwise subtraction."""
        lhs = _FakeTensorFlowTensor(other)
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> _FakeTensorFlowTensor:
        """Apply elementwise multiplication with NumPy broadcasting."""
        rhs = _FakeTensorFlowTensor(other)
        lhs_values = self._values
        rhs_values = rhs.numpy()
        values = lhs_values * rhs_values
        width = self._derivative_width(self._derivative, rhs._derivative)
        derivative = None
        if width is not None:
            lhs_derivative = self._broadcast_derivative(
                self._derivative,
                target_shape=values.shape,
                width=width,
            )
            rhs_derivative = self._broadcast_derivative(
                rhs._derivative,
                target_shape=values.shape,
                width=width,
            )
            derivative = (
                lhs_derivative * np.broadcast_to(rhs_values, values.shape)[..., np.newaxis]
                + rhs_derivative * np.broadcast_to(lhs_values, values.shape)[..., np.newaxis]
            )
        return self._with_derivative(values, derivative)

    __rmul__ = __mul__


class _FakeTensorFlowVariable(_FakeTensorFlowTensor):
    """Mutable fake TensorFlow tensor used as a trainable variable."""

    def __init__(self, values: object) -> None:
        """Initialize deterministic state for the fake framework object."""
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError("fake TensorFlow Variable only supports vectors")
        derivative = np.eye(array.size, dtype=float)
        super().__init__(array, derivative=derivative)


class _FakeTensorFlowGradientTape:
    """Gradient-tape double that reads stored derivative rows."""

    def __init__(self, module: _FakeTensorFlow) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.module = module
        self.watched: list[_FakeTensorFlowTensor] = []
        self.entered = False

    def __enter__(self) -> _FakeTensorFlowGradientTape:
        """Enter the fake gradient-tape context."""
        self.entered = True
        self.module.gradient_tape_entries += 1
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the fake gradient-tape context without suppressing errors."""
        del exc_type, exc, tb

    def watch(self, tensor: _FakeTensorFlowTensor) -> None:
        """Record the variable watched by this gradient tape."""
        self.watched.append(tensor)

    def gradient(
        self,
        loss: _FakeTensorFlowTensor,
        params: _FakeTensorFlowTensor,
    ) -> _FakeTensorFlowTensor | None:
        """Evaluate the recorded gradient transform for input values."""
        del params
        self.module.gradient_calls += 1
        gradient = loss.derivative()
        if gradient is None:
            return None
        return _FakeTensorFlowTensor(gradient)


class _FakeTensorFlowKerasLayer:
    """Minimal Keras layer base with deterministic weight creation."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.trainable_variables: list[_FakeTensorFlowVariable] = []
        self.non_trainable_variables: list[_FakeTensorFlowVariable] = []

    def add_weight(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        initializer: object,
        trainable: bool = True,
        dtype: object | None = None,
    ) -> _FakeTensorFlowVariable:
        """Create one deterministic trainable layer weight."""
        del name, dtype
        values = (
            initializer(shape=shape) if callable(initializer) else np.zeros(shape, dtype=float)
        )
        variable = _FakeTensorFlowVariable(values)
        if trainable:
            self.trainable_variables.append(variable)
        else:
            self.non_trainable_variables.append(variable)
        return variable

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the fake callable with NumPy-backed inputs."""
        return cast(Any, self).call(*args, **kwargs)


class _FakeTensorFlowConstantInitializer:
    """Constant initializer returning copied NumPy values."""

    def __init__(self, values: object) -> None:
        """Initialize deterministic state for the fake framework object."""
        self._values = _FakeTensorFlowTensor(values).numpy()

    def __call__(self, *, shape: tuple[int, ...]) -> FloatArray:
        """Execute the fake callable with NumPy-backed inputs."""
        values = np.asarray(self._values, dtype=float)
        if values.shape != shape:
            return np.broadcast_to(values, shape).copy()
        return values.copy()


class _FakeTensorFlowKerasInitializers:
    """Namespace exposing the fake constant initializer."""

    Constant = _FakeTensorFlowConstantInitializer


class _FakeTensorFlowKerasLayers:
    """Namespace exposing the fake Keras layer base."""

    Layer = _FakeTensorFlowKerasLayer


class _FakeTensorFlowKeras:
    """Keras namespace composed from the fake layers and initializers."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.layers = _FakeTensorFlowKerasLayers()
        self.initializers = _FakeTensorFlowKerasInitializers()


class _FakeTensorFlow:
    """Bounded TensorFlow facade backed by NumPy test doubles."""

    float64 = np.float64

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.convert_calls: list[FloatArray] = []
        self.gradient_tape_entries = 0
        self.gradient_calls = 0
        self.function_traces = 0
        self.function_calls = 0
        self.function_jit_flags: list[bool | None] = []
        self.keras = _FakeTensorFlowKeras()

    def convert_to_tensor(
        self,
        values: object,
        *,
        dtype: object | None = None,
    ) -> _FakeTensorFlowTensor:
        """Convert values to a NumPy-backed fake TensorFlow tensor."""
        del dtype
        array = np.asarray(values, dtype=float)
        self.convert_calls.append(array.copy())
        return _FakeTensorFlowTensor(array)

    def Variable(self, values: object, *, dtype: object | None = None) -> _FakeTensorFlowVariable:
        """Create a fake trainable TensorFlow variable."""
        del dtype
        return _FakeTensorFlowVariable(values)

    def GradientTape(self) -> _FakeTensorFlowGradientTape:
        """Create a fake TensorFlow gradient-tape context."""
        return _FakeTensorFlowGradientTape(self)

    def function(self, fn: object, *, jit_compile: bool | None = None) -> object:
        """Decorate a callable while recording XLA compilation requests."""
        self.function_traces += 1
        self.function_jit_flags.append(jit_compile)

        def wrapped(*args: object, **kwargs: object) -> object:
            """Forward calls through the fake TensorFlow function wrapper."""
            self.function_calls += 1
            return cast(Callable[..., object], fn)(*args, **kwargs)

        return wrapped

    def cos(self, values: object) -> _FakeTensorFlowTensor:
        """Apply elementwise cosine while preserving fake tensor wrapping."""
        tensor = _FakeTensorFlowTensor(values)
        array = tensor.numpy()
        derivative = tensor.derivative()
        if derivative is not None:
            derivative = -np.sin(array)[..., np.newaxis] * derivative
        return _FakeTensorFlowTensor(np.cos(array), derivative=derivative)

    def reduce_mean(
        self,
        values: object,
        *,
        axis: int | None = None,
    ) -> _FakeTensorFlowTensor:
        """Return the scalar mean while propagating derivative rows."""
        tensor = _FakeTensorFlowTensor(values)
        derivative = tensor.derivative()
        if derivative is not None:
            value_ndim = tensor.numpy().ndim
            if axis is None:
                derivative_axis: int | tuple[int, ...] = tuple(range(value_ndim))
            else:
                derivative_axis = axis
            derivative = np.mean(derivative, axis=derivative_axis)
        return _FakeTensorFlowTensor(
            np.mean(tensor.numpy(), axis=axis),
            derivative=derivative,
        )


def _objective(values: FloatArray) -> float:
    """Evaluate the shared two-parameter cosine objective in radians."""
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_tensorflow_gradient_tape_compatibility_audit_checks_native_gradient_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow gradient tape compatibility audit checks native gradient
    tape.
    """
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTensorFlowTensor(np.array([0.45], dtype=float))

    result = tensorflow_bridge.run_tensorflow_gradient_tape_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(
        result,
        tensorflow_bridge.PhaseTensorFlowGradientTapeCompatibilityResult,
    )
    assert result.passed
    assert result.gradient_tape_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_tensorflow_gradient_tape_compatibility"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_tf.gradient_tape_entries == 1
    assert fake_tf.gradient_calls == 1
    assert result.to_dict()["gradient_tape_supported"] is True


def test_tensorflow_gradient_tape_compatibility_fails_closed_without_gradient_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow gradient tape compatibility fails closed without gradient
    tape.
    """
    fake_tf = _FakeTensorFlow()
    fake_tf.GradientTape = None  # type: ignore[assignment]
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(RuntimeError, match="GradientTape"):
        tensorflow_bridge.run_tensorflow_gradient_tape_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_tensorflow_function_compatibility_audit_checks_traced_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow function compatibility audit checks traced gradient."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTensorFlowTensor(np.array([0.45], dtype=float))

    result = tensorflow_bridge.run_tensorflow_function_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(
        result,
        tensorflow_bridge.PhaseTensorFlowFunctionCompatibilityResult,
    )
    assert result.passed
    assert result.function_supported
    assert result.gradient_tape_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_tensorflow_function_compatibility"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_tf.function_traces == 1
    assert fake_tf.function_calls == 1
    assert fake_tf.gradient_tape_entries == 1
    assert fake_tf.gradient_calls == 1
    assert result.to_dict()["function_supported"] is True


def test_tensorflow_function_compatibility_fails_closed_without_tf_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow function compatibility fails closed without tf function."""
    fake_tf = _FakeTensorFlow()
    fake_tf.function = None  # type: ignore[assignment]
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(RuntimeError, match="tf.function"):
        tensorflow_bridge.run_tensorflow_function_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_tensorflow_xla_compatibility_audit_requests_jit_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow XLA compatibility audit requests JIT compile."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTensorFlowTensor(np.array([0.45], dtype=float))

    result = tensorflow_bridge.run_tensorflow_xla_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(
        result,
        tensorflow_bridge.PhaseTensorFlowXLACompatibilityResult,
    )
    assert result.passed
    assert result.xla_compile_requested
    assert result.function_supported
    assert result.gradient_tape_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_tensorflow_xla_compatibility"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_tf.function_jit_flags == [True]
    assert fake_tf.function_calls == 1
    assert fake_tf.gradient_calls == 1
    assert result.to_dict()["xla_compile_requested"] is True


def test_tensorflow_xla_compatibility_fails_closed_without_jit_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow XLA compatibility fails closed without JIT compile."""
    fake_tf = _FakeTensorFlow()

    def function_without_jit(fn: object) -> object:
        """Reject unsupported JIT compilation requests in the fake decorator."""
        return fn

    fake_tf.function = function_without_jit  # type: ignore[assignment]
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(RuntimeError, match="jit_compile"):
        tensorflow_bridge.run_tensorflow_xla_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_tensorflow_keras_layer_wraps_bounded_loss_and_reference_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow Keras layer wraps bounded loss and reference gradient."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    layer = tensorflow_bounded_qnn_keras_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
    )
    frozen_layer = tensorflow_bounded_qnn_keras_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=False,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(features, labels, initial_params)
    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert layer.claim_boundary == "bounded_tensorflow_keras_layer_wrapper"
    assert layer.feature_width == 1
    assert layer.host_boundary is False
    assert layer.native_framework_autodiff is True
    assert len(layer.trainable_variables) == 1
    assert len(frozen_layer.trainable_variables) == 0
    np.testing.assert_allclose(layer().numpy(), expected_loss, atol=1e-12)
    np.testing.assert_allclose(layer.parameter_shift_gradient(), expected_gradient, atol=1e-12)
    np.testing.assert_allclose(frozen_layer().numpy(), expected_loss, atol=1e-12)


def test_tensorflow_keras_layer_wrapper_audit_checks_gradient_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow Keras layer wrapper audit checks gradient tape."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    result = run_tensorflow_keras_layer_wrapper_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert isinstance(
        result,
        tensorflow_bridge.PhaseTensorFlowKerasLayerWrapperAuditResult,
    )
    assert result.passed
    assert result.keras_layer_supported
    assert result.gradient_tape_supported
    assert result.trainable_parameters == 1
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_tensorflow_keras_layer_wrapper"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_tf.gradient_tape_entries == 1
    assert fake_tf.gradient_calls == 1
    assert result.to_dict()["keras_layer_supported"] is True


def test_tensorflow_maturity_audit_records_bounded_passes_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow maturity audit records bounded passes and provider gaps."""
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)

    result = run_tensorflow_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        tolerance=1e-12,
    )

    assert isinstance(result, PhaseTensorFlowMaturityAuditResult)
    assert result.bounded_model_ready
    assert not result.ready_for_provider_exceedance
    analytic_tensor = cast(
        PhaseTensorFlowQNNGradientResult,
        result.evidence["analytic_tensor"],
    )
    gradient_tape = cast(
        tensorflow_bridge.PhaseTensorFlowGradientTapeCompatibilityResult,
        result.evidence["gradient_tape"],
    )
    tf_function = cast(
        tensorflow_bridge.PhaseTensorFlowFunctionCompatibilityResult,
        result.evidence["tf_function"],
    )
    xla = cast(
        tensorflow_bridge.PhaseTensorFlowXLACompatibilityResult,
        result.evidence["xla"],
    )
    keras_layer = cast(
        tensorflow_bridge.PhaseTensorFlowKerasLayerWrapperAuditResult,
        result.evidence["keras_layer"],
    )
    assert analytic_tensor.passed
    assert gradient_tape.passed
    assert tf_function.passed
    assert xla.passed
    assert keras_layer.passed
    assert "arbitrary_phase_qnode_tensorflow_lowering" in result.open_gaps
    assert "hardware_gradient_execution" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    required_capabilities = cast(dict[str, str], payload["required_capabilities"])
    assert required_capabilities["xla"] == "passed"
    assert required_capabilities["provider_callbacks"] == "blocked"
    assert payload["claim_boundary"] == "bounded_tensorflow_provider_maturity_audit"


def test_tensorflow_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes() -> None:
    """Verify that TensorFlow phase QNode lowering matrix fails closed for arbitrary
    qnodes.
    """
    result = run_tensorflow_phase_qnode_lowering_matrix()

    assert isinstance(result, PhaseTensorFlowPhaseQNodeLoweringMatrixResult)
    assert result.bounded_qnn_routes_ready
    assert not result.arbitrary_phase_qnode_lowering_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("bounded_qnn_gradient_tape") == "passed"
    assert result.route_status("bounded_qnn_tf_function") == "passed"
    assert result.route_status("bounded_qnn_xla") == "passed"
    assert result.route_status("registered_phase_qnode_statevector_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_graph_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_provider_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_hardware_lowering") == "blocked"
    assert "registered_phase_qnode_statevector_lowering" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_tensorflow_phase_qnode_lowering_matrix"

    payload = cast(dict[str, Any], result.to_dict())
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["bounded_qnn_keras_layer_wrapper"]["status"] == "passed"
    assert routes["registered_phase_qnode_hardware_lowering"]["requires"] == [
        "live_ticket",
        "provider_allowlist",
        "shot_budget",
        "hardware_evidence_id",
    ]


def test_tensorflow_keras_layer_fails_closed_without_keras_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that TensorFlow Keras layer fails closed without Keras layer."""
    fake_tf = _FakeTensorFlow()
    fake_tf.keras = object()  # type: ignore[assignment]
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(RuntimeError, match="tf.keras.layers.Layer"):
        tensorflow_bounded_qnn_keras_layer(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
        )


def test_tensorflow_bridge_fails_closed_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when the optional TensorFlow dependency cannot be imported."""

    def missing_tensorflow() -> object:
        """Raise the deterministic optional-dependency failure for TensorFlow."""
        raise ImportError("tensorflow blocked")

    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", missing_tensorflow)
    assert not is_phase_tensorflow_available()
    with pytest.raises(ImportError, match="tensorflow blocked"):
        tensorflow_parameter_shift_value_and_grad(
            _objective,
            np.array([0.2, -0.4], dtype=float),
        )
    with pytest.raises(ImportError, match="tensorflow blocked"):
        tensorflow_bounded_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="tensorflow blocked"):
        tensorflow_bounded_qnn_keras_layer(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="tensorflow blocked"):
        run_tensorflow_keras_layer_wrapper_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
