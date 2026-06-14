# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Framework Bridges
"""Tests for optional PyTorch and TensorFlow phase-gradient bridges."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control.phase.tensorflow_bridge as tensorflow_bridge
import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTensorFlowParameterShiftResult,
    PhaseTensorFlowQNNGradientResult,
    PhaseTorchAutogradQNNGradientResult,
    PhaseTorchCompileCompatibilityResult,
    PhaseTorchFuncCompatibilityResult,
    PhaseTorchModuleWrapperAuditResult,
    PhaseTorchParameterShiftResult,
    PhaseTorchQNNGradientResult,
    is_phase_tensorflow_available,
    is_phase_torch_available,
    multi_frequency_parameter_shift_rule,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    run_torch_compile_compatibility_audit,
    run_torch_func_compatibility_audit,
    run_torch_module_wrapper_audit,
    tensorflow_bounded_qnn_value_and_grad,
    tensorflow_parameter_shift_value_and_grad,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
    torch_bounded_qnn_value_and_grad,
    torch_parameter_shift_value_and_grad,
)


class _FakeTorchTensor:
    def __init__(self, values: object) -> None:
        if isinstance(values, _FakeTorchTensor):
            values = values.numpy()
        self._values = np.asarray(values, dtype=float)
        self.grad: _FakeTorchTensor | None = None

    def detach(self) -> _FakeTorchTensor:
        return self

    def clone(self) -> _FakeTorchTensor:
        return _FakeTorchTensor(self._values.copy())

    def requires_grad_(self, requires_grad: bool = True) -> _FakeTorchTensor:
        del requires_grad
        return self

    def cpu(self) -> _FakeTorchTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values.copy()

    def __mul__(self, other: object) -> _FakeTorchTensor:
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values * other._values)
        return _FakeTorchTensor(self._values * np.asarray(other, dtype=float))

    __rmul__ = __mul__

    def __add__(self, other: object) -> _FakeTorchTensor:
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values + other._values)
        return _FakeTorchTensor(self._values + np.asarray(other, dtype=float))

    __radd__ = __add__

    def __sub__(self, other: object) -> _FakeTorchTensor:
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values - other._values)
        return _FakeTorchTensor(self._values - np.asarray(other, dtype=float))

    def __rsub__(self, other: object) -> _FakeTorchTensor:
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(other._values - self._values)
        return _FakeTorchTensor(np.asarray(other, dtype=float) - self._values)

    def unsqueeze(self, axis: int) -> _FakeTorchTensor:
        return _FakeTorchTensor(np.expand_dims(self._values, axis=axis))


class _FakeTorchAutogradFunction:
    @classmethod
    def apply(cls, *args: object) -> _FakeTorchTensor:
        ctx = type("_FakeAutogradContext", (), {})()
        result = cls.forward(ctx, *args)
        result._ctx = ctx  # type: ignore[attr-defined]
        result._function_cls = cls  # type: ignore[attr-defined]
        return result


class _FakeTorchAutograd:
    Function = _FakeTorchAutogradFunction

    def grad(
        self,
        outputs: _FakeTorchTensor,
        inputs: _FakeTorchTensor,
        *,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> tuple[_FakeTorchTensor]:
        del inputs, retain_graph, create_graph
        backward = outputs._function_cls.backward  # type: ignore[attr-defined]
        result = backward(outputs._ctx, _FakeTorchTensor(np.asarray(1.0, dtype=float)))  # type: ignore[attr-defined]
        if isinstance(result, tuple):
            return result
        return (result,)


class _FakeTorchModule:
    def __init__(self) -> None:
        self._buffers: dict[str, _FakeTorchTensor] = {}

    def register_buffer(self, name: str, tensor: _FakeTorchTensor) -> None:
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def parameters(self) -> tuple[_FakeTorchTensor, ...]:
        params = getattr(self, "params", None)
        if params is None:
            return ()
        return (params,)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)


class _FakeTorchNN:
    Module = _FakeTorchModule

    @staticmethod
    def Parameter(values: object, *, requires_grad: bool = True) -> _FakeTorchTensor:
        return _FakeTorchTensor(values).requires_grad_(requires_grad)


class _FakeTorchFunc:
    def __init__(self) -> None:
        self.grad_calls = 0
        self.vmap_calls = 0
        self.jacrev_calls = 0

    def grad(self, loss_fn: object) -> object:
        del loss_fn
        self.grad_calls += 1

        def gradient(params: object) -> _FakeTorchTensor:
            return _FakeTorchTensor(self._gradient(params))

        return gradient

    def vmap(self, gradient_fn: object) -> object:
        self.vmap_calls += 1

        def mapped(params_batch: object) -> _FakeTorchTensor:
            batch = np.asarray(_FakeTorchTensor(params_batch).numpy(), dtype=float)
            return _FakeTorchTensor(np.vstack([self._gradient(row) for row in batch]))

        return mapped

    def jacrev(self, loss_fn: object) -> object:
        del loss_fn
        self.jacrev_calls += 1

        def jacobian(params: object) -> _FakeTorchTensor:
            return _FakeTorchTensor(self._gradient(params))

        return jacobian

    @staticmethod
    def _gradient(params: object) -> np.ndarray:
        features = np.array([[0.0], [np.pi]], dtype=float)
        labels = np.array([0.0, 1.0], dtype=float)
        return parameter_shift_qnn_classifier_gradient(
            features,
            labels,
            _FakeTorchTensor(params).numpy(),
        )


class _FakeTorch:
    float64 = np.float64

    def __init__(self) -> None:
        self.autograd = _FakeTorchAutograd()
        self.func = _FakeTorchFunc()
        self.nn = _FakeTorchNN()
        self.as_tensor_calls: list[np.ndarray] = []
        self.compile_calls: list[dict[str, object]] = []

    def as_tensor(self, values: object, *, dtype: object | None = None) -> _FakeTorchTensor:
        del dtype
        array = np.asarray(values, dtype=float)
        self.as_tensor_calls.append(array.copy())
        return _FakeTorchTensor(array)

    def compile(
        self,
        fn: object,
        *,
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> object:
        self.compile_calls.append({"fullgraph": fullgraph, "dynamic": dynamic})
        return fn

    def cos(self, values: object) -> _FakeTorchTensor:
        return _FakeTorchTensor(np.cos(_FakeTorchTensor(values).numpy()))

    def mean(self, values: object, *, dim: int | None = None) -> _FakeTorchTensor:
        return _FakeTorchTensor(np.mean(_FakeTorchTensor(values).numpy(), axis=dim))


class _FakeTorchWithoutAutogradFunction(_FakeTorch):
    def __init__(self) -> None:
        super().__init__()
        self.autograd = object()


class _FakeTorchWithoutFunc(_FakeTorch):
    def __init__(self) -> None:
        super().__init__()
        self.func = object()


class _FakeTorchWithoutCompile(_FakeTorch):
    compile = None


class _FakeTorchWithoutNN(_FakeTorch):
    def __init__(self) -> None:
        super().__init__()
        self.nn = object()


class _FakeTensorFlowTensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=float)

    def numpy(self) -> np.ndarray:
        return self._values.copy()


class _FakeTensorFlow:
    float64 = np.float64

    def __init__(self) -> None:
        self.convert_calls: list[np.ndarray] = []

    def convert_to_tensor(
        self,
        values: object,
        *,
        dtype: object | None = None,
    ) -> _FakeTensorFlowTensor:
        del dtype
        array = np.asarray(values, dtype=float)
        self.convert_calls.append(array.copy())
        return _FakeTensorFlowTensor(array)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_torch_bridge_returns_tensor_and_numpy_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = torch_parameter_shift_value_and_grad(
        _objective,
        _FakeTorchTensor(np.array([0.2, -0.4], dtype=float)),
    )

    assert isinstance(result, PhaseTorchParameterShiftResult)
    assert is_phase_torch_available()
    assert result.method == "parameter_shift"
    assert result.host_boundary
    assert result.evaluations == 5
    assert isinstance(result.torch_value, _FakeTorchTensor)
    assert isinstance(result.torch_gradient, _FakeTorchTensor)
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.torch_gradient.numpy(), result.gradient, atol=1e-12)


def test_torch_bridge_reports_multi_frequency_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    result = torch_parameter_shift_value_and_grad(
        objective,
        _FakeTorchTensor(np.array([0.4], dtype=float)),
        rule=rule,
    )

    expected = np.array([np.cos(0.4) - 0.2 * np.sin(0.8)], dtype=float)
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    assert result.to_dict()["shift_terms"] == len(rule.terms)


def test_torch_bounded_qnn_gradient_matches_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = torch_bounded_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(
        features,
        labels,
        params.numpy(),
    )
    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTorchQNNGradientResult)
    assert result.passed
    assert result.analytic_framework_gradient
    assert not result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["passed"] is True


def test_torch_autograd_qnn_gradient_uses_custom_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = torch_autograd_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTorchAutogradQNNGradientResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert result.custom_autograd_function
    assert not result.host_boundary
    assert result.method == "torch_bounded_phase_qnn_custom_autograd_function"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["custom_autograd_function"] is True


def test_torch_autograd_qnn_gradient_fails_closed_without_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorchWithoutAutogradFunction()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.autograd.Function"):
        torch_autograd_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_torch_func_compatibility_audit_checks_grad_vmap_and_jacrev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)

    result = run_torch_func_compatibility_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    expected_batch = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch],
    )
    assert isinstance(result, PhaseTorchFuncCompatibilityResult)
    assert result.passed
    assert result.func_grad_supported
    assert result.func_vmap_supported
    assert result.func_jacrev_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_func_compatibility"
    np.testing.assert_allclose(result.grad_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.jacrev_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.vmap_gradients, expected_batch, atol=1e-12)
    assert fake_torch.func.grad_calls == 1
    assert fake_torch.func.vmap_calls == 1
    assert fake_torch.func.jacrev_calls == 1
    assert result.to_dict()["func_vmap_supported"] is True


def test_torch_func_compatibility_audit_fails_closed_without_torch_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorchWithoutFunc()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.func"):
        run_torch_func_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.45]], dtype=float),
        )


def test_torch_compile_compatibility_audit_checks_compiled_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTorchTensor(np.array([0.45], dtype=float))

    result = run_torch_compile_compatibility_audit(
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
    assert isinstance(result, PhaseTorchCompileCompatibilityResult)
    assert result.passed
    assert result.torch_compile_supported
    assert result.compiled_loss_supported
    assert result.compiled_gradient_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_compile_compatibility"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert fake_torch.func.grad_calls == 1
    assert fake_torch.compile_calls == [{"fullgraph": True, "dynamic": False}]
    assert result.to_dict()["compiled_gradient_supported"] is True


def test_torch_compile_compatibility_audit_fails_closed_without_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorchWithoutCompile()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.compile"):
        run_torch_compile_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_torch_bounded_qnn_module_and_layer_wrap_bounded_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
    )
    layer = torch_bounded_qnn_layer(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=False,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(features, labels, initial_params)
    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert module.claim_boundary == "bounded_torch_module_layer_wrapper"
    assert module.feature_width == 1
    assert module.host_boundary is False
    np.testing.assert_allclose(module().numpy(), expected_loss, atol=1e-12)
    np.testing.assert_allclose(module.parameter_shift_gradient(), expected_gradient, atol=1e-12)
    np.testing.assert_allclose(layer().numpy(), expected_loss, atol=1e-12)


def test_torch_module_wrapper_audit_checks_module_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    result = run_torch_module_wrapper_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        tolerance=1e-12,
    )

    expected_gradient = parameter_shift_qnn_classifier_gradient(features, labels, initial_params)
    assert isinstance(result, PhaseTorchModuleWrapperAuditResult)
    assert result.passed
    assert result.module_wrapper_supported
    assert result.layer_wrapper_supported
    assert result.trainable_parameters == 1
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "bounded_torch_module_layer_wrapper"
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.torch_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["module_wrapper_supported"] is True
    assert fake_torch.func.grad_calls == 1


def test_torch_module_wrapper_fails_closed_without_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorchWithoutNN()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(RuntimeError, match="torch.nn.Module"):
        torch_bounded_qnn_module(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
        )


def test_torch_bounded_qnn_gradient_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="params width must match feature width"):
        torch_bounded_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_tensorflow_bridge_returns_tensor_and_numpy_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    result = tensorflow_parameter_shift_value_and_grad(
        _objective,
        _FakeTensorFlowTensor(np.array([0.2, -0.4], dtype=float)),
    )

    assert isinstance(result, PhaseTensorFlowParameterShiftResult)
    assert is_phase_tensorflow_available()
    assert result.method == "parameter_shift"
    assert result.host_boundary
    assert result.evaluations == 5
    assert isinstance(result.tensorflow_value, _FakeTensorFlowTensor)
    assert isinstance(result.tensorflow_gradient, _FakeTensorFlowTensor)
    np.testing.assert_allclose(
        result.gradient,
        np.array([-np.sin(0.2), 0.25 * np.cos(-0.4)], dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.tensorflow_gradient.numpy(),
        result.gradient,
        atol=1e-12,
    )


def test_tensorflow_bridge_reports_multi_frequency_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    result = tensorflow_parameter_shift_value_and_grad(
        objective,
        _FakeTensorFlowTensor(np.array([0.4], dtype=float)),
        rule=rule,
    )

    expected = np.array([np.cos(0.4) - 0.2 * np.sin(0.8)], dtype=float)
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.gradient, expected, atol=1e-12)
    assert result.to_dict()["shift_terms"] == len(rule.terms)


def test_tensorflow_bounded_qnn_gradient_matches_parameter_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = _FakeTensorFlowTensor(np.array([0.45], dtype=float))

    result = tensorflow_bounded_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=1e-12,
    )

    expected_loss = parameter_shift_qnn_classifier_loss(
        features,
        labels,
        params.numpy(),
    )
    expected_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        params.numpy(),
    )
    assert isinstance(result, PhaseTensorFlowQNNGradientResult)
    assert result.passed
    assert result.analytic_framework_gradient
    assert not result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.parameter_shift_gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(result.tensorflow_gradient.numpy(), expected_gradient, atol=1e-12)
    assert result.to_dict()["passed"] is True


def test_tensorflow_bounded_qnn_gradient_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tf = _FakeTensorFlow()
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", lambda: fake_tf)

    with pytest.raises(ValueError, match="params width must match feature width"):
        tensorflow_bounded_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_framework_bridges_fail_closed_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torch() -> object:
        raise ImportError("torch blocked")

    def missing_tensorflow() -> object:
        raise ImportError("tensorflow blocked")

    monkeypatch.setattr(torch_bridge, "_load_torch", missing_torch)
    monkeypatch.setattr(tensorflow_bridge, "_load_tensorflow", missing_tensorflow)

    assert not is_phase_torch_available()
    assert not is_phase_tensorflow_available()
    with pytest.raises(ImportError, match="torch blocked"):
        torch_parameter_shift_value_and_grad(_objective, np.array([0.2, -0.4], dtype=float))
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_autograd_qnn_value_and_grad(
            np.array([[0.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_func_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.2], dtype=float),
            params_batch=np.array([[0.2]], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_compile_compatibility_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_module(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        torch_bounded_qnn_layer(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
    with pytest.raises(ImportError, match="torch blocked"):
        run_torch_module_wrapper_audit(
            features=np.array([[0.0]], dtype=float),
            labels=np.array([0.0], dtype=float),
            initial_params=np.array([0.2], dtype=float),
        )
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
