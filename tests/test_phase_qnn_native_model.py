# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native phase classifier model tests
"""Compare native phase-model derivatives with independent scalar arithmetic."""

from __future__ import annotations

import importlib
import math
import pickle
from types import FunctionType
from typing import cast

import numpy as np
import pytest
from _pickle_identity import IdentityUnpickler
from numpy.typing import NDArray

from scpn_quantum_control.phase_qnn_native_contracts import (
    JAXPhaseNamespace,
    PhaseTensor,
    TorchPhaseNamespace,
    TorchPhaseTensor,
)
from scpn_quantum_control.phase_qnn_native_model import (
    jax_phase_classifier_gradient,
    jax_phase_classifier_loss,
    tensorflow_phase_classifier_loss,
    torch_phase_classifier_loss,
)

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize(
    "owner,name,consumers",
    [
        ("jax_gradients", "_jax_bounded_qnn_loss", ("jax_compatibility",)),
        ("jax_gradients", "_jax_bounded_qnn_gradient", ("jax_compatibility",)),
        (
            "torch_compatibility",
            "_torch_bounded_qnn_loss_tensor",
            ("torch_bridge", "torch_aot_autograd_export", "torch_dynamic_shape_export"),
        ),
        (
            "tensorflow_compatibility",
            "_tensorflow_bounded_qnn_loss_tensor",
            ("tensorflow_bridge",),
        ),
        ("torch_gradients", "_bounded_qnn_loss_gradient_reference", ("torch_bridge",)),
        ("torch_autograd_function", "_bounded_analytic_qnn_gradient", ()),
        ("tensorflow_gradients", "tensorflow_bounded_qnn_value_and_grad", ()),
    ],
)
def test_original_model_namespace_aliases_and_serialization(
    owner: str, name: str, consumers: tuple[str, ...]
) -> None:
    """Resolve the original defining object through imports and a real pickle.

    Parameters
    ----------
    owner
        Original natural defining module below the phase package.
    name
        Existing model helper whose global dictionary must remain unchanged.
    consumers
        Actual modules retaining the original imported binding.

    """
    module = importlib.import_module("scpn_quantum_control.phase." + owner)
    function = getattr(module, name)
    assert isinstance(function, FunctionType)
    assert function.__module__ == module.__name__
    assert function.__qualname__ == name
    assert function.__globals__ is vars(module)
    expected = {(module.__name__, name): function}
    assert IdentityUnpickler(pickle.dumps(function), expected).load() is function
    for consumer in consumers:
        imported = importlib.import_module("scpn_quantum_control.phase." + consumer)
        assert getattr(imported, name) is function


def _reference(
    features: NDArray[np.float64],
    labels: NDArray[np.float64],
    parameters: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    samples, width = features.shape
    residuals = [
        sum(
            (1.0 - math.cos(float(features[row, column] + parameters[column]))) / 2.0
            for column in range(width)
        )
        / width
        - float(labels[row])
        for row in range(samples)
    ]
    gradient = np.array(
        [
            sum(
                residuals[row] * math.sin(float(features[row, column] + parameters[column]))
                for row in range(samples)
            )
            / (samples * width)
            for column in range(width)
        ]
    )
    hessian = np.zeros((width, width), dtype=np.float64)
    for first in range(width):
        for second in range(width):
            hessian[first, second] = (
                sum(
                    math.sin(float(features[row, first] + parameters[first]))
                    * math.sin(float(features[row, second] + parameters[second]))
                    / (2.0 * width * width)
                    + (
                        residuals[row]
                        * math.cos(float(features[row, first] + parameters[first]))
                        / width
                        if first == second
                        else 0.0
                    )
                    for row in range(samples)
                )
                / samples
            )
    return sum(value * value for value in residuals) / samples, gradient, hessian


@pytest.mark.parametrize("samples,width", [(1, 1), (3, 2), (5, 4)])
def test_jax_model_preserves_jit_gradient_curvature_and_public_bridge(
    samples: int, width: int
) -> None:
    """Retain feature coupling through native JIT and both legacy QNN entries.

    Parameters
    ----------
    samples
        Independent labelled sample count, including a singleton batch.
    width
        Phase parameter count and feature-reduction axis length.

    """
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    bridge = importlib.import_module("scpn_quantum_control.phase.jax_bridge")
    features = np.linspace(-1.3, 1.7, samples * width).reshape(samples, width)
    labels = np.linspace(0.1, 0.9, samples)
    parameters = np.linspace(-0.4, 0.3, width)
    loss, gradient, hessian = _reference(features, labels, parameters)
    with jax.enable_x64():
        native_features = jnp.asarray(features)
        native_labels = jnp.asarray(labels)
        native_parameters = jnp.asarray(parameters)

        def objective(values: PhaseTensor) -> PhaseTensor:
            return jax_phase_classifier_loss(
                cast(JAXPhaseNamespace, jnp), native_features, native_labels, values
            )

        value, actual_gradient = jax.jit(jax.value_and_grad(objective))(native_parameters)
        actual_hessian = jax.jit(jax.hessian(objective))(native_parameters)
        analytic_gradient = jax.jit(
            lambda values: jax_phase_classifier_gradient(
                cast(JAXPhaseNamespace, jnp), native_features, native_labels, values
            )
        )(native_parameters)
        np.testing.assert_allclose(value, loss, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(actual_gradient, gradient, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(analytic_gradient, gradient, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(actual_hessian, hessian, rtol=0.0, atol=1e-14)
        for entry in (
            bridge.jax_native_qnn_value_and_grad,
            bridge.jax_custom_vjp_qnn_value_and_grad,
        ):
            result = entry(features, labels, parameters, jit=True, tolerance=1e-9)
            assert result.passed
            assert result.loss == pytest.approx(loss, abs=1e-14)
            np.testing.assert_allclose(result.gradient, gradient, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(parameters, np.linspace(-0.4, 0.3, width))


@pytest.mark.parametrize("samples,width", [(1, 1), (3, 2), (5, 4)])
def test_torch_model_preserves_autograd_curvature_and_public_module(
    samples: int, width: int
) -> None:
    """Differentiate real native tensors and execute the original module factory.

    Parameters
    ----------
    samples
        Independent sample count for the full-batch loss.
    width
        Number of feature phases, including coupled multi-feature predictions.

    """
    torch = importlib.import_module("torch")
    bridge = importlib.import_module("scpn_quantum_control.phase.torch_bridge")
    features = np.linspace(-1.3, 1.7, samples * width).reshape(samples, width)
    labels = np.linspace(0.1, 0.9, samples)
    parameters = np.linspace(-0.4, 0.3, width)
    loss, gradient, hessian = _reference(features, labels, parameters)
    native_features = torch.tensor(features, dtype=torch.float64)
    native_labels = torch.tensor(labels, dtype=torch.float64)
    native_parameters = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)

    def objective(values: TorchPhaseTensor) -> PhaseTensor:
        return torch_phase_classifier_loss(
            cast(TorchPhaseNamespace, torch), native_features, native_labels, values
        )

    value = objective(native_parameters)
    actual_gradient = torch.autograd.grad(value, native_parameters)[0]
    actual_hessian = torch.autograd.functional.hessian(objective, native_parameters)
    np.testing.assert_allclose(torch.Tensor.detach(value), loss, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=0.0, atol=1e-14)
    module = bridge.torch_bounded_qnn_module(
        features=features, labels=labels, initial_params=parameters
    )
    module_value = module()
    module_gradient = torch.autograd.grad(module_value, module.params)[0]
    np.testing.assert_allclose(module_value.detach(), loss, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(module_gradient, gradient, rtol=0.0, atol=1e-14)
    assert type(module).__module__ == "scpn_quantum_control.phase.torch_compatibility"
    host_result = bridge.torch_bounded_qnn_value_and_grad(features, labels, parameters)
    assert host_result.passed
    assert host_result.loss == pytest.approx(loss, abs=1e-14)
    np.testing.assert_allclose(host_result.gradient, gradient, rtol=0.0, atol=1e-14)
    custom = importlib.import_module("scpn_quantum_control.phase.torch_autograd_function")
    custom_parameters = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
    custom_loss = custom.torch_autograd_function_qnn_loss(features, labels, custom_parameters)
    custom_gradient = torch.autograd.grad(custom_loss, custom_parameters)[0]
    np.testing.assert_allclose(custom_gradient, gradient, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(parameters, np.linspace(-0.4, 0.3, width))


@pytest.mark.parametrize("samples,width", [(1, 1), (3, 2), (5, 4)])
def test_tensorflow_model_preserves_nested_tapes_and_public_keras_layer(
    samples: int, width: int
) -> None:
    """Retain trainable-variable gradients and cross-feature second derivatives.

    Parameters
    ----------
    samples
        Independent sample count for full-batch averaging.
    width
        Number of feature phases watched by both native tapes.

    """
    tf = importlib.import_module("tensorflow")
    bridge = importlib.import_module("scpn_quantum_control.phase.tensorflow_bridge")
    features = np.linspace(-1.3, 1.7, samples * width).reshape(samples, width)
    labels = np.linspace(0.1, 0.9, samples)
    parameters = np.linspace(-0.4, 0.3, width)
    loss, gradient, hessian = _reference(features, labels, parameters)
    native_features = tf.constant(features, dtype=tf.float64)
    native_labels = tf.constant(labels, dtype=tf.float64)
    native_parameters = tf.Variable(parameters, dtype=tf.float64)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            value = tensorflow_phase_classifier_loss(
                tf.cos, tf.reduce_mean, native_features, native_labels, native_parameters
            )
        actual_gradient = inner.gradient(value, native_parameters)
    actual_hessian = outer.jacobian(actual_gradient, native_parameters)
    np.testing.assert_allclose(tf.convert_to_tensor(value).numpy(), loss, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=0.0, atol=1e-14)
    layer = bridge.tensorflow_bounded_qnn_keras_layer(
        features=features, labels=labels, initial_params=parameters
    )
    with tf.GradientTape() as tape:
        layer_value = layer()
    layer_gradient = tape.gradient(layer_value, layer.params)
    np.testing.assert_allclose(layer_value, loss, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(layer_gradient, gradient, rtol=0.0, atol=1e-14)
    assert type(layer).__module__ == "scpn_quantum_control.phase.tensorflow_compatibility"
    host_result = bridge.tensorflow_bounded_qnn_value_and_grad(features, labels, parameters)
    assert host_result.passed
    assert host_result.loss == pytest.approx(loss, abs=1e-14)
    np.testing.assert_allclose(host_result.gradient, gradient, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(parameters, np.linspace(-0.4, 0.3, width))
