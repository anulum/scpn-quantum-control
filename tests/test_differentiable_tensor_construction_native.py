# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real native tensor construction
"""Exercise real CPU SDK allocation, storage sharing and gradient lifetimes."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_tensor_construction import (
    TensorConstructionPrimitives,
    construct_tensorflow_tensor,
    construct_torch_tensor,
    prepare_torch_trainable_tensor,
)
from scpn_quantum_control.phase import tensorflow_gradients, torch_gradients

pytestmark = pytest.mark.framework_imports


def _primitives() -> TensorConstructionPrimitives:
    return TensorConstructionPrimitives(
        attribute=lambda: getattr, predicate=lambda: callable, invalid=lambda: RuntimeError
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_real_torch_factory_precedence_and_dtype(
    monkeypatch: pytest.MonkeyPatch, fallback: bool, explicit_dtype: bool
) -> None:
    """Execute both real factories and both original dtype branches.

    Parameters
    ----------
    monkeypatch
        Restore actual SDK attributes after capability-absence checks.
    fallback
        Remove only the preferred factory, exercising the real fallback.
    explicit_dtype
        Retain or remove the SDK's advertised float64 attribute.

    """
    torch = importlib.import_module("torch")
    source = np.array([0.2, -0.4], dtype=np.float32)
    original = source.copy()
    with monkeypatch.context() as scoped:
        if fallback:
            scoped.delattr(torch, "as_tensor")
        if not explicit_dtype:
            scoped.delattr(torch, "float64")
        tensor = construct_torch_tensor(torch, source, primitives=_primitives())
    actual = np.asarray(tensor)
    expected_dtype = np.float64 if explicit_dtype else np.float32
    assert actual.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(actual, original.astype(expected_dtype))
    assert np.shares_memory(actual, source) == (not fallback and not explicit_dtype)
    np.testing.assert_array_equal(source, original)


def test_real_torch_trainable_tensor_is_fresh_leaf() -> None:
    """Detach a real graph, clone its storage and retain native backward."""
    torch = importlib.import_module("torch")
    source = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    nonleaf = source * 2.0
    actual = prepare_torch_trainable_tensor(nonleaf, primitives=_primitives())
    assert torch.is_tensor(actual)
    tensor = torch.as_tensor(actual)
    assert tensor.is_leaf and tensor.requires_grad
    assert tensor.data_ptr() != nonleaf.data_ptr()
    loss = torch.sum(tensor.square())
    loss.backward()
    np.testing.assert_allclose(np.asarray(tensor.grad, dtype=np.float64), [0.8, -1.6])
    assert source.grad is None
    np.testing.assert_array_equal(source.detach().numpy(), [0.2, -0.4])
    assert tensor is actual


@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_real_tensorflow_factory_dtype(
    monkeypatch: pytest.MonkeyPatch, explicit_dtype: bool
) -> None:
    """Execute TensorFlow conversion with and without explicit dtype support.

    Parameters
    ----------
    monkeypatch
        Restore the actual SDK dtype attribute after its absence check.
    explicit_dtype
        Whether the conversion explicitly requests native float64.

    """
    tensorflow = importlib.import_module("tensorflow")
    source = np.array([0.2, -0.4], dtype=np.float32)
    with monkeypatch.context() as scoped:
        if not explicit_dtype:
            scoped.delattr(tensorflow, "float64")
        tensor = construct_tensorflow_tensor(tensorflow, source, primitives=_primitives())
    expected_dtype = np.float64 if explicit_dtype else np.float32
    assert np.asarray(tensor).dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(np.asarray(tensor), source.astype(expected_dtype))


def test_original_public_torch_parameter_shift_uses_live_attribute_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real SDK route observes rebinding between its dtype and factory reads.

    Parameters
    ----------
    monkeypatch
        Restore the original module namespace after its live-read probe.

    """
    reads: list[str] = []

    def second(value: object, name: str, default: object) -> object:
        reads.append("second:" + name)
        return getattr(value, name, default)

    def first(value: object, name: str, default: object) -> object:
        reads.append("first:" + name)
        if name == "float64":
            monkeypatch.setattr(torch_gradients, "getattr", second, raising=False)
        return getattr(value, name, default)

    monkeypatch.setattr(torch_gradients, "getattr", first, raising=False)
    result = torch_gradients.torch_parameter_shift_value_and_grad(
        lambda values: float(np.cos(values[0])), [0.2]
    )
    np.testing.assert_allclose(result.gradient, [-np.sin(0.2)])
    np.testing.assert_array_equal(np.asarray(result.torch_gradient), result.gradient)
    assert "first:float64" in reads and "second:as_tensor" in reads


def test_original_public_tensorflow_route_uses_actual_tensor_conversion() -> None:
    """The original public host-gradient adapter still returns native tensors."""
    tensorflow = importlib.import_module("tensorflow")
    result = tensorflow_gradients.tensorflow_parameter_shift_value_and_grad(
        lambda values: float(np.cos(values[0])),
        [0.2],
        _tensorflow_loader=lambda: tensorflow,
    )
    np.testing.assert_allclose(result.gradient, [-np.sin(0.2)])
    np.testing.assert_array_equal(np.asarray(result.tensorflow_gradient), result.gradient)
    assert result.tensorflow_gradient.dtype == tensorflow.float64


def test_tensorflow_conversion_retains_actual_tape_derivative() -> None:
    """Converting a real watched variable retains its gradient-tape connection."""
    tensorflow = importlib.import_module("tensorflow")
    source = tensorflow.Variable([0.2, -0.4], dtype=tensorflow.float64)
    with tensorflow.GradientTape() as tape:
        tensor = construct_tensorflow_tensor(tensorflow, source, primitives=_primitives())
        loss = tensorflow.reduce_sum(tensorflow.square(tensor))
    np.testing.assert_allclose(np.asarray(tape.gradient(loss, source)), [0.4, -0.8])
    np.testing.assert_array_equal(np.asarray(source), [0.2, -0.4])


def test_original_public_qnn_autograd_route_uses_fresh_trainable_parameters() -> None:
    """The original custom-backward route exercises the retained trainable helper."""
    source = np.array([0.4], dtype=np.float64)
    result = torch_gradients.torch_autograd_qnn_value_and_grad(
        [[0.0], [np.pi]], [0.0, 1.0], source
    )
    assert result.passed
    np.testing.assert_array_equal(source, [0.4])
    np.testing.assert_array_equal(np.asarray(result.torch_gradient), result.gradient)


@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
def test_original_public_route_retains_caller_error_constructor(
    monkeypatch: pytest.MonkeyPatch, framework: str
) -> None:
    """Missing actual SDK capabilities retain the old caller's exception class.

    Parameters
    ----------
    monkeypatch
        Restore native capabilities and the caller's error binding.
    framework
        Actual CPU SDK whose advertised factory is temporarily absent.

    """
    runtime = importlib.import_module(framework)

    class ConversionError(RuntimeError):
        """The caller's original module selects this conversion refusal."""

    def objective(values: NDArray[np.float64]) -> float:
        module = torch_gradients if framework == "torch" else tensorflow_gradients
        monkeypatch.setattr(module, "RuntimeError", ConversionError, raising=False)
        return float(np.cos(values[0]))

    if framework == "torch":
        monkeypatch.delattr(runtime, "as_tensor")
        monkeypatch.delattr(runtime, "tensor")
        with pytest.raises(ConversionError) as caught:
            torch_gradients.torch_parameter_shift_value_and_grad(objective, [0.2])
        assert str(caught.value) == "PyTorch module does not expose as_tensor or tensor"
    else:
        monkeypatch.delattr(runtime, "convert_to_tensor")
        with pytest.raises(ConversionError) as caught:
            tensorflow_gradients.tensorflow_parameter_shift_value_and_grad(
                objective, [0.2], _tensorflow_loader=lambda: runtime
            )
        assert str(caught.value) == "TensorFlow module does not expose convert_to_tensor"


def test_original_qnn_autograd_route_retains_trainable_capability_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing real tensor method refuses before the custom backward executes.

    Parameters
    ----------
    monkeypatch
        Restore the actual native Tensor method and caller error binding.

    """
    torch = importlib.import_module("torch")

    class TrainableError(RuntimeError):
        """Original caller-owned refusal for a missing trainable capability."""

    monkeypatch.setattr(torch.Tensor, "requires_grad_", None)
    monkeypatch.setattr(torch_gradients, "RuntimeError", TrainableError, raising=False)
    with pytest.raises(TrainableError) as caught:
        torch_gradients.torch_autograd_qnn_value_and_grad([[0.0], [np.pi]], [0.0, 1.0], [0.4])
    assert str(caught.value) == "PyTorch tensor does not expose requires_grad_"
