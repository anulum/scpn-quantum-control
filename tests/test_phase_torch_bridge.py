# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# SCPN Quantum Control — PyTorch Phase Bridge Tests

"""Contract tests for PyTorch phase-gradient and QNode bridge integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseTorchAutogradQNNGradientResult,
    PhaseTorchCloudValidationRunSpec,
    PhaseTorchCompileCompatibilityResult,
    PhaseTorchEcosystemMaturityAuditResult,
    PhaseTorchFuncCompatibilityResult,
    PhaseTorchMaturityAuditResult,
    PhaseTorchModuleWrapperAuditResult,
    PhaseTorchParameterShiftResult,
    PhaseTorchPhaseQNodeCompileResult,
    PhaseTorchPhaseQNodeLoweringMatrixResult,
    PhaseTorchPhaseQNodeStatevectorResult,
    PhaseTorchPhaseQNodeTransformResult,
    PhaseTorchQNNGradientResult,
    PhaseTorchTrainingLoopAuditResult,
    is_phase_torch_available,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
    plan_torch_cloud_validation_batch,
    run_torch_compile_compatibility_audit,
    run_torch_ecosystem_maturity_audit,
    run_torch_func_compatibility_audit,
    run_torch_maturity_audit,
    run_torch_module_wrapper_audit,
    run_torch_phase_qnode_lowering_matrix,
    run_torch_training_loop_audit,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_layer,
    torch_bounded_qnn_module,
    torch_bounded_qnn_value_and_grad,
    torch_parameter_shift_value_and_grad,
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
)

FloatArray = NDArray[np.float64]


class _FakeTorchTensor:
    """NumPy-backed tensor implementing the PyTorch operations under test."""

    def __init__(self, values: object) -> None:
        """Initialize deterministic state for the fake framework object."""
        if isinstance(values, _FakeTorchTensor):
            values = values.numpy()
        self._values = np.asarray(values, dtype=float)
        self.grad: _FakeTorchTensor | None = None

    def detach(self) -> _FakeTorchTensor:
        """Return this tensor without changing its NumPy-backed state."""
        return self

    def clone(self) -> _FakeTorchTensor:
        """Return an independent copy of this fake tensor."""
        return _FakeTorchTensor(self._values.copy())

    def requires_grad_(self, requires_grad: bool = True) -> _FakeTorchTensor:
        """Record whether callers requested gradient tracking."""
        del requires_grad
        return self

    def cpu(self) -> _FakeTorchTensor:
        """Return this CPU-only fake tensor unchanged."""
        return self

    def numpy(self) -> FloatArray:
        """Return the tensor payload as a NumPy array."""
        return self._values.copy()

    def __mul__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise multiplication with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values * other._values)
        return _FakeTorchTensor(self._values * np.asarray(other, dtype=float))

    __rmul__ = __mul__

    def __add__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise addition with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values + other._values)
        return _FakeTorchTensor(self._values + np.asarray(other, dtype=float))

    __radd__ = __add__

    def __sub__(self, other: object) -> _FakeTorchTensor:
        """Apply elementwise subtraction with NumPy broadcasting."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(self._values - other._values)
        return _FakeTorchTensor(self._values - np.asarray(other, dtype=float))

    def __rsub__(self, other: object) -> _FakeTorchTensor:
        """Apply reflected elementwise subtraction."""
        if isinstance(other, _FakeTorchTensor):
            return _FakeTorchTensor(other._values - self._values)
        return _FakeTorchTensor(np.asarray(other, dtype=float) - self._values)

    def unsqueeze(self, axis: int) -> _FakeTorchTensor:
        """Insert one singleton dimension at the requested axis."""
        return _FakeTorchTensor(np.expand_dims(self._values, axis=axis))


class _FakeTorchAutogradFunction:
    """Minimal custom-autograd function dispatcher for bridge tests."""

    @classmethod
    def apply(cls, *args: object) -> _FakeTorchTensor:
        """Execute the supplied fake custom-autograd function."""
        ctx = type("_FakeAutogradContext", (), {})()
        result = cast(Any, cls).forward(ctx, *args)
        result._ctx = ctx
        result._function_cls = cls
        return cast(_FakeTorchTensor, result)


class _FakeTorchAutograd:
    """Finite-difference autograd facade for deterministic bridge tests."""

    Function = _FakeTorchAutogradFunction

    def grad(
        self,
        outputs: _FakeTorchTensor,
        inputs: _FakeTorchTensor,
        *,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> tuple[_FakeTorchTensor]:
        """Build or evaluate the deterministic fake gradient operation."""
        del inputs, retain_graph, create_graph
        backward = outputs._function_cls.backward  # type: ignore[attr-defined]
        result = backward(outputs._ctx, _FakeTorchTensor(np.asarray(1.0, dtype=float)))  # type: ignore[attr-defined]
        if isinstance(result, tuple):
            return result
        return (result,)


class _FakeTorchModule:
    """Small module base exposing buffers, parameters, and calls."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self._buffers: dict[str, _FakeTorchTensor] = {}

    def register_buffer(self, name: str, tensor: _FakeTorchTensor) -> None:
        """Attach a named non-trainable buffer to the fake module."""
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def parameters(self) -> tuple[_FakeTorchTensor, ...]:
        """Return the fake module's trainable parameter tuple."""
        params = getattr(self, "params", None)
        if params is None:
            return ()
        return (params,)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the fake callable with NumPy-backed inputs."""
        return cast(Any, self).forward(*args, **kwargs)


class _FakeTorchNN:
    """Namespace containing the fake PyTorch module and parameter types."""

    Module = _FakeTorchModule

    @staticmethod
    def Parameter(values: object, *, requires_grad: bool = True) -> _FakeTorchTensor:
        """Wrap values as a trainable fake PyTorch tensor."""
        return _FakeTorchTensor(values).requires_grad_(requires_grad)


class _FakeTorchFunc:
    """Deterministic torch.func facade recording transform usage."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.grad_calls = 0
        self.vmap_calls = 0
        self.jacrev_calls = 0

    def grad(self, loss_fn: object) -> object:
        """Build or evaluate the deterministic fake gradient operation."""
        del loss_fn
        self.grad_calls += 1

        def gradient(params: object) -> _FakeTorchTensor:
            """Evaluate the recorded gradient transform for input values."""
            return _FakeTorchTensor(self._gradient(params))

        return gradient

    def vmap(self, gradient_fn: object) -> object:
        """Build a deterministic vectorizing transform wrapper."""
        self.vmap_calls += 1

        def mapped(params_batch: object) -> _FakeTorchTensor:
            """Evaluate the wrapped function across the leading batch axis."""
            batch = np.asarray(_FakeTorchTensor(params_batch).numpy(), dtype=float)
            return _FakeTorchTensor(np.vstack([self._gradient(row) for row in batch]))

        return mapped

    def jacrev(self, loss_fn: object) -> object:
        """Build a deterministic reverse-Jacobian transform wrapper."""
        del loss_fn
        self.jacrev_calls += 1

        def jacobian(params: object) -> _FakeTorchTensor:
            """Evaluate the finite-difference Jacobian for one input vector."""
            return _FakeTorchTensor(self._gradient(params))

        return jacobian

    @staticmethod
    def _gradient(params: object) -> FloatArray:
        """Evaluate a central finite-difference gradient in float64."""
        features = np.array([[0.0], [np.pi]], dtype=float)
        labels = np.array([0.0, 1.0], dtype=float)
        return parameter_shift_qnn_classifier_gradient(
            features,
            labels,
            _FakeTorchTensor(params).numpy(),
        )


class _FakeTorch:
    """Bounded PyTorch facade backed by NumPy test doubles."""

    float64 = np.float64
    __version__ = "fake-torch"

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        self.autograd = _FakeTorchAutograd()
        self.func = _FakeTorchFunc()
        self.nn = _FakeTorchNN()
        self.as_tensor_calls: list[FloatArray] = []
        self.compile_calls: list[dict[str, object]] = []

    def as_tensor(self, values: object, *, dtype: object | None = None) -> _FakeTorchTensor:
        """Convert values to a NumPy-backed fake PyTorch tensor."""
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
        """Record compilation and return the callable unchanged."""
        self.compile_calls.append({"fullgraph": fullgraph, "dynamic": dynamic})
        return fn

    def cos(self, values: object) -> _FakeTorchTensor:
        """Apply elementwise cosine while preserving fake tensor wrapping."""
        return _FakeTorchTensor(np.cos(_FakeTorchTensor(values).numpy()))

    def mean(self, values: object, *, dim: int | None = None) -> _FakeTorchTensor:
        """Return the scalar mean as a fake tensor."""
        return _FakeTorchTensor(np.mean(_FakeTorchTensor(values).numpy(), axis=dim))


class _FakeTorchWithoutAutogradFunction(_FakeTorch):
    """PyTorch facade lacking custom autograd support."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).autograd = object()


class _FakeTorchWithoutFunc(_FakeTorch):
    """PyTorch facade lacking the torch.func transform namespace."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).func = object()


class _FakeTorchWithoutCompile(_FakeTorch):
    """PyTorch facade lacking the torch.compile entry point."""

    compile: Any = None


class _FakeTorchWithoutNN(_FakeTorch):
    """PyTorch facade lacking the torch.nn module namespace."""

    def __init__(self) -> None:
        """Initialize deterministic state for the fake framework object."""
        super().__init__()
        cast(Any, self).nn = object()


def _objective(values: FloatArray) -> float:
    """Evaluate the shared two-parameter cosine objective in radians."""
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def test_torch_bridge_returns_tensor_and_numpy_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that PyTorch bridge returns tensor and NumPy gradients."""
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
    """Verify that PyTorch bridge reports multi frequency parameter shift."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: FloatArray) -> float:
        """Evaluate the local cosine objective used to inspect rule metadata."""
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
    """Verify that PyTorch bounded QNN gradient matches parameter shift."""
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
    """Verify that PyTorch autograd QNN gradient uses custom function."""
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
    """Verify that PyTorch autograd QNN gradient fails closed without function."""
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
    """Verify that PyTorch func compatibility audit checks grad vmap and jacrev."""
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
    """Verify that PyTorch func compatibility audit fails closed without PyTorch func."""
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
    """Verify that PyTorch compile compatibility audit checks compiled grad."""
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
    """Verify that PyTorch compile compatibility audit fails closed without compile."""
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
    """Verify that PyTorch bounded QNN module and layer wrap bounded loss."""
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
    """Verify that PyTorch module wrapper audit checks module grad."""
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


def test_torch_training_loop_audit_updates_module_with_compile_and_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch training loop audit updates module with compile and func."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    initial_params = np.array([0.45], dtype=float)

    result = run_torch_training_loop_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        learning_rate=0.2,
        steps=4,
        tolerance=1e-12,
        fullgraph=True,
    )

    initial_loss = parameter_shift_qnn_classifier_loss(features, labels, initial_params)
    final_reference_gradient = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        result.final_params,
    )
    assert isinstance(result, PhaseTorchTrainingLoopAuditResult)
    assert result.passed
    assert result.steps == 4
    assert result.learning_rate == 0.2
    assert result.initial_loss == pytest.approx(initial_loss)
    assert result.final_loss < result.initial_loss
    assert result.loss_history.shape == (5,)
    assert np.all(np.diff(result.loss_history) <= 1e-12)
    assert result.gradient_history.shape == (4, 1)
    assert result.module_wrapper_supported
    assert result.func_grad_supported
    assert result.torch_compile_supported
    assert result.compiled_loss_supported
    assert result.parameter_update_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    np.testing.assert_allclose(result.final_gradient, final_reference_gradient, atol=1e-12)
    payload = result.to_dict()
    assert payload["claim_boundary"] == "bounded_torch_training_loop_parity"
    assert payload["compiled_loss_supported"] is True
    json.dumps(payload)
    assert fake_torch.func.grad_calls >= 1
    assert len(fake_torch.compile_calls) >= 1


@pytest.mark.parametrize(
    ("learning_rate", "steps", "tolerance", "match"),
    [
        (0.0, 4, 1e-12, "learning_rate"),
        (-0.1, 4, 1e-12, "learning_rate"),
        (0.2, 0, 1e-12, "steps"),
        (0.2, True, 1e-12, "steps"),
        (0.2, 4, -1e-12, "tolerance"),
    ],
)
def test_torch_training_loop_audit_fails_closed_on_invalid_controls(
    monkeypatch: pytest.MonkeyPatch,
    learning_rate: float,
    steps: int,
    tolerance: float,
    match: str,
) -> None:
    """Verify that PyTorch training loop audit fails closed on invalid controls."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match=match):
        run_torch_training_loop_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
            learning_rate=learning_rate,
            steps=steps,
            tolerance=tolerance,
        )


def test_torch_training_loop_audit_fails_closed_on_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch training loop audit fails closed on shape mismatch."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match=r"initial_params must have shape \(2,\)"):
        run_torch_training_loop_audit(
            features=np.array([[0.0, np.pi]], dtype=float),
            labels=np.array([1.0], dtype=float),
            initial_params=np.array([0.45], dtype=float),
        )


def test_torch_maturity_audit_records_bounded_passes_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that PyTorch maturity audit records bounded passes and provider gaps."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)
    live_overlay_artifact = tmp_path / "diff-qnode-external-comparison.json"
    live_overlay_artifact.write_text(
        json.dumps(
            {
                "artifact_id": "diff-qnode-external-comparison-local",
                "classification": "functional_non_isolated",
                "promotion_ready": False,
                "rows": [
                    {
                        "backend": "pytorch",
                        "status": "success",
                        "value_error": 0.0,
                        "gradient_error": 0.0,
                        "runtime_seconds": 0.1,
                        "memory_peak_bytes": 4096,
                        "batching_support": "torch.func.vmap",
                        "transform_support": "torch.func.grad/jacrev",
                        "dependency_versions": {"torch": "2.11.0+cpu"},
                        "claim_boundary": "bounded CPU comparison only",
                    }
                ],
            },
        ),
        encoding="utf-8",
    )

    result = run_torch_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=1e-12,
        live_overlay_artifact_path=live_overlay_artifact,
    )

    assert isinstance(result, PhaseTorchMaturityAuditResult)
    assert result.bounded_model_ready
    assert not result.ready_for_provider_exceedance
    evidence = cast(dict[str, Any], result.evidence)
    assert evidence["analytic_tensor"].passed
    assert evidence["custom_autograd"].passed
    assert evidence["torch_func"].passed
    assert evidence["torch_compile"].passed
    assert evidence["module_layer_wrapper"].passed
    assert evidence["training_loop"].passed
    assert evidence["ecosystem_maturity"].route_status("torch_compile_callable") == "passed"
    assert evidence["cloud_validation_batch"].ready_for_cloud_dispatch
    assert evidence["live_overlay"].passed
    assert evidence["live_overlay"].artifact_id == "diff-qnode-external-comparison-local"
    assert "finite_shot_provider_hardware_torch_phase_qnode_lowering" in result.open_gaps
    assert "torch_ecosystem_maturity" in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps
    assert "live_overlay_execution" not in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    json.dumps(payload)
    required_capabilities = cast(dict[str, str], payload["required_capabilities"])
    assert required_capabilities["torch_compile"] == "passed"
    assert required_capabilities["training_loop"] == "passed"
    assert required_capabilities["torch_ecosystem_maturity"] == "blocked"
    assert required_capabilities["cloud_validation_batch"] == "scheduled"
    assert required_capabilities["live_overlay_execution"] == "passed"
    assert (
        cast(dict[str, Any], payload["evidence"])["live_overlay"]["torch_version"] == "2.11.0+cpu"
    )
    assert payload["claim_boundary"] == "bounded_torch_provider_maturity_audit"


def test_torch_maturity_audit_rejects_invalid_lowering_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch maturity aggregation should fail closed on invalid lowering evidence."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(
        torch_bridge,
        "plan_torch_cloud_validation_batch",
        lambda: PhaseTorchCloudValidationRunSpec(
            runner="local",
            local_execution_status="skipped",
            local_skip_reason="unit test",
            torch_version="fake",
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_names=(),
            blocked_local_routes=(),
            required_artifacts=(),
            required_environment={},
            commands=(),
            ready_for_cloud_dispatch=False,
        ),
    )
    monkeypatch.setattr(torch_bridge, "run_torch_phase_qnode_lowering_matrix", lambda: object())

    with pytest.raises(RuntimeError, match="phase-QNode lowering matrix"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
        )


def test_torch_ecosystem_maturity_audit_records_broad_module_func_compile_device_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch ecosystem maturity audit records broad module func compile
    device gaps.
    """
    fake_torch = _FakeTorch()
    fake_torch.__version__ = "2.11.0+cpu"
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = run_torch_ecosystem_maturity_audit()

    assert isinstance(result, PhaseTorchEcosystemMaturityAuditResult)
    assert not result.ready_for_provider_exceedance
    assert result.route_status("nn_module_parameter_surface") == "passed"
    assert result.route_status("torch_func_grad_vmap_jacrev") == "passed"
    assert result.route_status("torch_func_jacfwd_hessian") == "blocked"
    assert result.route_status("torch_compile_callable") == "passed"
    assert result.route_status("registered_phase_qnode_torch_compile_lowering") == "passed"
    assert (
        result.route_status("registered_phase_qnode_torch_compile_fullgraph_lowering") == "blocked"
    )
    assert result.route_status("cuda_accelerator_device") == "blocked"
    assert result.torch_version == "2.11.0+cpu"
    assert not result.cuda_available
    assert "cuda_accelerator_device" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    assert (
        cast(dict[str, Any], payload["routes"])["cuda_accelerator_device"]["status"] == "blocked"
    )
    assert payload["claim_boundary"] == "torch_ecosystem_device_maturity_audit"


def test_torch_maturity_audit_rejects_incomplete_live_overlay_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that PyTorch maturity audit rejects incomplete live overlay artifact."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    bad_artifact = tmp_path / "diff-qnode-external-comparison.json"
    bad_artifact.write_text(
        json.dumps(
            {
                "artifact_id": "diff-qnode-external-comparison-local",
                "classification": "functional_non_isolated",
                "rows": [{"backend": "pytorch", "status": "hard_gap"}],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="successful PyTorch row"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
            live_overlay_artifact_path=bad_artifact,
        )


def test_torch_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes() -> None:
    """Verify that PyTorch phase QNode lowering matrix fails closed for arbitrary
    qnodes.
    """
    result = run_torch_phase_qnode_lowering_matrix()

    assert isinstance(result, PhaseTorchPhaseQNodeLoweringMatrixResult)
    assert result.bounded_qnn_routes_ready
    assert not result.arbitrary_phase_qnode_lowering_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("bounded_qnn_custom_autograd") == "passed"
    assert result.route_status("registered_phase_qnode_statevector_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_torch_func_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_torch_compile_lowering") == "passed"
    assert (
        result.route_status("registered_phase_qnode_torch_compile_fullgraph_lowering") == "blocked"
    )
    assert result.route_status("registered_phase_qnode_cuda_device_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_provider_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_hardware_lowering") == "blocked"
    assert "registered_phase_qnode_statevector_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_func_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_compile_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_torch_phase_qnode_lowering_matrix"

    payload = cast(dict[str, Any], result.to_dict())
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["bounded_qnn_torch_compile"]["status"] == "passed"
    assert routes["registered_phase_qnode_hardware_lowering"]["status"] == "blocked"
    assert routes["registered_phase_qnode_hardware_lowering"]["requires"] == [
        "live_ticket",
        "provider_allowlist",
        "shot_budget",
        "hardware_evidence_id",
    ]


def test_torch_cloud_validation_batch_schedules_incompatible_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch cloud validation batch schedules incompatible local device."""
    fake_torch = _FakeTorch()
    fake_torch.__version__ = "2.11.0+cpu"
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = plan_torch_cloud_validation_batch(runner="jarvislabs")

    assert isinstance(result, PhaseTorchCloudValidationRunSpec)
    assert result.runner == "jarvislabs"
    assert result.ready_for_cloud_dispatch
    assert result.local_execution_status == "skipped_incompatible_local_hardware"
    assert "CUDA" in result.local_skip_reason
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in result.blocked_local_routes
    assert "registered_phase_qnode_cuda_device_lowering" in result.blocked_local_routes
    assert "registered_phase_qnode_fullgraph_compile_artifact" in result.required_artifacts
    assert "cuda_device_phase_qnode_gradient_artifact" in result.required_artifacts
    assert "isolated_benchmark_artifact" in result.required_artifacts
    assert any("test_phase_framework_bridges.py" in command for command in result.commands)
    assert result.required_environment["accelerator_backend"] == "cuda"
    assert result.claim_boundary == "torch_cloud_validation_batch_plan"
    payload = result.to_dict()
    assert payload["local_execution_status"] == "skipped_incompatible_local_hardware"
    json.dumps(payload)


def test_torch_phase_qnode_compile_audit_lowers_registered_statevector() -> None:
    """Verify that PyTorch phase QNode compile audit lowers registered statevector."""
    pytest.importorskip("torch", reason="native Torch Phase-QNode compile requires PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = torch_phase_qnode_compile_audit(
        circuit,
        params,
        tolerance=1e-8,
        fullgraph=False,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseTorchPhaseQNodeCompileResult)
    assert result.passed
    assert result.torch_compile_supported
    assert result.compiled_value_supported
    assert result.compiled_gradient_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert not result.fullgraph
    assert result.claim_boundary == "registered_phase_qnode_torch_compile_lowering"
    np.testing.assert_allclose(result.value, reference.value, atol=1e-8)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    payload = result.to_dict()
    assert payload["compiled_gradient_supported"] is True
    json.dumps(payload)


def test_torch_phase_qnode_value_and_grad_lowers_registered_statevector() -> None:
    """Verify that PyTorch phase QNode value and grad lowers registered statevector."""
    pytest.importorskip("torch", reason="native Torch phase-QNode lowering requires PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = torch_phase_qnode_value_and_grad(circuit, params, tolerance=1e-8)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseTorchPhaseQNodeStatevectorResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "registered_phase_qnode_torch_statevector_lowering"
    np.testing.assert_allclose(result.value, reference.value, atol=1e-8)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    assert result.max_abs_error <= 1e-8
    assert result.state.shape == (4,)
    payload = result.to_dict()
    assert payload["method"] == "torch_native_registered_phase_qnode_statevector_value_and_grad"
    assert payload["host_boundary"] is False
    json.dumps(payload)


def test_torch_phase_qnode_transform_audit_checks_grad_jacrev_and_vmap() -> None:
    """Verify that PyTorch phase QNode transform audit checks grad jacrev and vmap."""
    pytest.importorskip("torch", reason="native Torch Phase-QNode transforms require PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)
    params_batch = np.array([[0.37, -0.21], [0.4, -0.25]], dtype=float)

    result = torch_phase_qnode_transform_audit(
        circuit,
        params,
        params_batch=params_batch,
        tolerance=1e-8,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)
    reference_batch = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in params_batch]
    )

    assert isinstance(result, PhaseTorchPhaseQNodeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.func_grad_supported
    assert result.func_vmap_supported
    assert result.func_jacrev_supported
    assert result.claim_boundary == "registered_phase_qnode_torch_func_transform_lowering"
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=1e-8)
    np.testing.assert_allclose(result.vmap_gradients, reference_batch, atol=1e-8)
    payload = result.to_dict()
    assert payload["host_boundary"] is False
    assert payload["func_vmap_supported"] is True
    json.dumps(payload)


def test_torch_phase_qnode_transform_audit_fails_closed_without_torch_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch phase QNode transform audit fails closed without PyTorch
    func.
    """
    fake_torch = _FakeTorchWithoutFunc()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="torch.func"):
        torch_phase_qnode_transform_audit(
            circuit,
            np.array([0.2], dtype=float),
            params_batch=np.array([[0.2]], dtype=float),
        )


def test_torch_maturity_audit_fails_closed_on_bad_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch maturity audit fails closed on bad batch shape."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="params_batch"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([0.25], dtype=float),
        )


def test_torch_module_wrapper_fails_closed_without_nn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch module wrapper fails closed without nn."""
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
    """Verify that PyTorch bounded QNN gradient fails closed on shape mismatch."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="params width must match feature width"):
        torch_bounded_qnn_value_and_grad(
            np.array([[0.0, 1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.45], dtype=float),
        )


def test_torch_bridge_fails_closed_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when the optional PyTorch dependency cannot be imported."""

    def missing_torch() -> object:
        """Raise the deterministic optional-dependency failure for PyTorch."""
        raise ImportError("torch blocked")

    monkeypatch.setattr(torch_bridge, "_load_torch", missing_torch)
    assert not is_phase_torch_available()
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
