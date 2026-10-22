# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native forward and reverse Jacobian products
"""Qualify rectangular native Jacobian products and original producer resources."""

from __future__ import annotations

import gc
import importlib
import inspect
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_native_linearization import (
    jax_jvp,
    jax_vjp,
    tensorflow_jvp,
    tensorflow_vjp,
    torch_jvp,
    torch_vjp,
)
from scpn_quantum_control.differentiable_transform_algebra import TransformAlgebraCase

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_native_products_match_rectangular_nonsymmetric_jacobian(
    framework: str, width: int
) -> None:
    """Distinguish forward and transpose products using independent analytic derivatives.

    Parameters
    ----------
    framework
        Required actual CPU runtime, with no optional skip or diagnostic substitute.
    width
        Input dimension; the output has one additional component.

    """
    parameters = np.linspace(-0.3, 0.4, width, dtype=np.float64)
    original = parameters.copy()
    matrix = (np.arange((width + 1) * width, dtype=np.float64) + 1.0).reshape(width + 1, width)
    matrix /= (width + 1) * width + 1.0
    coefficients = np.linspace(0.15, 0.35, width + 1, dtype=np.float64)
    tangent = np.linspace(-0.4, 0.7, width, dtype=np.float64)
    cotangent = np.linspace(0.6, -0.2, width + 1, dtype=np.float64)
    expected_value = matrix @ np.sin(parameters) + coefficients * np.sum(parameters**2)
    jacobian = matrix * np.cos(parameters)[None, :] + np.outer(coefficients, 2.0 * parameters)
    array: Callable[[Any], NDArray[np.float64]]
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            native_matrix = jnp.asarray(matrix, dtype=jnp.float64)
            native_coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
            native_parameters = jnp.asarray(parameters, dtype=jnp.float64)

            def jax_function(values: Any) -> Any:
                """Evaluate the actual vector function with native JAX operations."""
                return native_matrix @ jnp.sin(values) + native_coefficients * jnp.sum(values**2)

            forward_value, pushed = jax_jvp(
                jax, jax_function, native_parameters, jnp.asarray(tangent, dtype=jnp.float64)
            )
            reverse_value, pullback = jax_vjp(jax, jax_function, native_parameters)
            reverse = pullback(jnp.asarray(cotangent, dtype=jnp.float64))
            reverse_twice = pullback(jnp.asarray(2.0 * cotangent, dtype=jnp.float64))
            assert len(reverse) == len(reverse_twice) == 1
            pulled, pulled_twice = reverse[0], reverse_twice[0]
            assert pushed.dtype == pulled.dtype == np.float64
        array = np.asarray
    elif framework == "pytorch":
        import torch

        native_matrix = torch.tensor(matrix, dtype=torch.float64, device="cpu")
        native_coefficients = torch.tensor(coefficients, dtype=torch.float64, device="cpu")
        native_parameters = torch.tensor(parameters, dtype=torch.float64, device="cpu")

        def torch_function(values: Any) -> Any:
            """Evaluate the actual vector function with native torch operations."""
            return native_matrix @ torch.sin(values) + native_coefficients * torch.sum(values**2)

        forward_value, pushed = torch_jvp(
            torch,
            torch_function,
            native_parameters,
            torch.tensor(tangent, dtype=torch.float64, device="cpu"),
        )
        reverse_value, pullback = torch_vjp(torch, torch_function, native_parameters)
        reverse = pullback(torch.tensor(cotangent, dtype=torch.float64, device="cpu"))
        reverse_twice = pullback(torch.tensor(2.0 * cotangent, dtype=torch.float64, device="cpu"))
        assert len(reverse) == len(reverse_twice) == 1
        pulled, pulled_twice = reverse[0], reverse_twice[0]
        assert pushed.dtype == pulled.dtype == torch.float64

        def torch_array(value: Any) -> NDArray[np.float64]:
            """Read the actual CPU float64 product without retaining its graph."""
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

        array = torch_array
    else:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            native_matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
            native_coefficients = tf.convert_to_tensor(coefficients, dtype=tf.float64)
            native_parameters = tf.convert_to_tensor(parameters, dtype=tf.float64)
            native_tangent = tf.convert_to_tensor(tangent, dtype=tf.float64)
            native_cotangent = tf.convert_to_tensor(cotangent, dtype=tf.float64)

            def tensorflow_function(values: Any) -> Any:
                """Evaluate the actual vector function with native TensorFlow operations."""
                return tf.linalg.matvec(native_matrix, tf.sin(values)) + (
                    native_coefficients * tf.reduce_sum(values**2)
                )

            with tf.autodiff.ForwardAccumulator(native_parameters, native_tangent) as accumulator:
                forward_value = tensorflow_function(native_parameters)
            pushed = tensorflow_jvp(accumulator, forward_value)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(native_parameters)
                reverse_value = tensorflow_function(native_parameters)
            pulled = tensorflow_vjp(tape, reverse_value, native_parameters, native_cotangent)
            pulled_twice = tensorflow_vjp(
                tape, reverse_value, native_parameters, 2.0 * native_cotangent
            )
            assert pushed.dtype == pulled.dtype == tf.float64
        array = np.asarray
    np.testing.assert_allclose(array(forward_value), expected_value, atol=1e-12, rtol=0)
    np.testing.assert_allclose(array(reverse_value), expected_value, atol=1e-12, rtol=0)
    np.testing.assert_allclose(array(pushed), jacobian @ tangent, atol=1e-12, rtol=0)
    np.testing.assert_allclose(array(pulled), jacobian.T @ cotangent, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        array(pulled_twice), 2.0 * (jacobian.T @ cotangent), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        np.dot(cotangent, array(pushed)), np.dot(array(pulled), tangent), atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(parameters, original)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_public_producers_retain_original_pullback_and_accumulator_scopes(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep actual native resources in their original caller frames through reporting.

    Parameters
    ----------
    framework
        Required actual native producer; all original case and refusal rows execute.
    monkeypatch
        Source observers that delegate every operation to its real implementation.

    """
    native = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_" + native + "_transform_algebra"
    )
    original_forward: Callable[..., Any] = getattr(module, native + "_jvp")
    original_reverse: Callable[..., Any] = getattr(module, native + "_vjp")
    original_case: Callable[..., TransformAlgebraCase] = module._framework_transform_case
    resources: list[weakref.ReferenceType[Any]] = []
    forwards: list[str] = []
    reports: list[str] = []

    def forward(*arguments: Any) -> Any:
        """Require native products to execute under the original numerical caller."""
        frame = inspect.currentframe()
        try:
            assert frame is not None
            frame = frame.f_back
            assert frame is not None and frame.f_globals is vars(module)
            if framework == "tensorflow":
                assert frame.f_code.co_qualname == "_execute_cases.<locals>.forward_gradient"
                assert frame.f_code.co_freevars == ("derivative", "tf", "x")
                assert frame.f_locals["accumulator"] is arguments[0]
                assert frame.f_locals["gradient"] is arguments[1]
            else:
                assert frame.f_code is module._execute_cases.__code__
                assert frame.f_locals["derivative"] is arguments[1]
                assert frame.f_locals["x"] is arguments[2]
                assert frame.f_locals["tangent"] is arguments[3]
        finally:
            del frame
        forwards.append("forward")
        return original_forward(*arguments)

    def reverse(*arguments: Any) -> Any:
        """Execute the real reverse operation and retain only an observing weakref."""
        frame = inspect.currentframe()
        try:
            assert frame is not None
            frame = frame.f_back
            assert frame is not None and frame.f_code is module._execute_cases.__code__
            assert frame.f_globals is vars(module)
            if framework == "tensorflow":
                assert frame.f_locals["pullback"] is arguments[0]
                assert frame.f_locals["gradient"] is arguments[1]
            else:
                assert frame.f_locals["derivative"] is arguments[1]
        finally:
            del frame
        result = original_reverse(*arguments)
        resources.append(weakref.ref(arguments[0] if framework == "tensorflow" else result[1]))
        return result

    def report(*arguments: Any) -> TransformAlgebraCase:
        """Require the original actual pullback binding throughout all real reporting."""
        frame = inspect.currentframe()
        try:
            while frame is not None and frame.f_code is not module._execute_cases.__code__:
                frame = frame.f_back
            assert frame is not None and frame.f_globals is vars(module)
            assert len(resources) == 1
            assert resources[0]() is not None
            assert frame.f_locals["pullback"] is resources[0]()
        finally:
            del frame
        reports.append("case")
        return original_case(*arguments)

    monkeypatch.setattr(module, native + "_jvp", forward)
    monkeypatch.setattr(module, native + "_vjp", reverse)
    monkeypatch.setattr(module, "_framework_transform_case", report)
    producer = getattr(module, "run_" + native + "_transform_algebra_cases")
    records = producer(np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64))
    assert len(forwards) >= 2 if framework == "tensorflow" else len(forwards) == 1
    assert len(reports) == 10
    assert len(records) == 11
    assert all(case.status == "passed" for case in records[:-1])
    assert records[-1].status == "blocked" and records[-1].blocked_reasons
    gc.collect()
    assert resources[0]() is None
