# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native derivative matrix contracts
"""Qualify actual derivative matrices, callback resources and host-conversion order."""

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

from scpn_quantum_control.differentiable_native_curvature import (
    jax_hessian,
    jax_jacfwd,
    jax_jacrev,
    jax_reverse_gradient_jacobian,
    tensorflow_forward_jacobian,
    tensorflow_jacobian,
    torch_forward_gradient_jacobian,
    torch_hessian,
    torch_jacfwd,
    torch_jacrev,
    torch_reverse_gradient_jacobian,
)
from scpn_quantum_control.differentiable_transform_algebra import TransformAlgebraCase

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_native_matrices_match_rectangular_jacobian_and_coupled_curvature(
    framework: str, width: int
) -> None:
    """Check actual native matrices against independent non-diagonal derivatives.

    Parameters
    ----------
    framework
        Required actual CPU framework without an optional skip or substitute.
    width
        Scalar-objective input width; the vector output has one extra component.

    """
    parameters = np.linspace(-0.3, 0.4, width, dtype=np.float64)
    original = parameters.copy()
    scale = np.arange(1, width + 1, dtype=np.float64) / width
    quadratic = 0.2 * np.outer(scale, scale) + np.diag(np.linspace(0.3, 0.6, width))
    matrix = (np.arange((width + 1) * width, dtype=np.float64) + 1).reshape(width + 1, width)
    matrix /= (width + 1) * width + 1
    coefficients = np.linspace(0.15, 0.35, width + 1, dtype=np.float64)
    expected_jacobian = matrix * np.cos(parameters)[None, :] + np.outer(
        coefficients, 2.0 * parameters
    )
    expected_hessian = (
        -np.sin(np.sum(parameters)) * np.ones((width, width))
        + quadratic
        + np.diag(3.0 * parameters**2)
    )
    array: Callable[[Any], NDArray[np.float64]]
    native_dtype: Any
    jacobians: list[Any]
    curvatures: list[Any]
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            native_parameters = jnp.asarray(parameters, dtype=jnp.float64)
            native_quadratic = jnp.asarray(quadratic, dtype=jnp.float64)
            native_matrix = jnp.asarray(matrix, dtype=jnp.float64)
            native_coefficients = jnp.asarray(coefficients, dtype=jnp.float64)

            def scalar(values: Any) -> Any:
                """Evaluate actual scalar coupling with native JAX operations."""
                return (
                    jnp.sin(jnp.sum(values))
                    + 0.5 * (values @ native_quadratic @ values)
                    + 0.25 * jnp.sum(values**4)
                )

            def vector(values: Any) -> Any:
                """Evaluate the native vector function with rectangular derivatives."""
                return native_matrix @ jnp.sin(values) + native_coefficients * jnp.sum(values**2)

            jacobians = [
                jax_jacfwd(jax, vector, native_parameters),
                jax_jacrev(jax, vector, native_parameters),
            ]
            curvatures = [
                jax_hessian(jax, scalar, native_parameters),
                jax_reverse_gradient_jacobian(jax, scalar, native_parameters),
            ]
        array, native_dtype = np.asarray, np.float64
    elif framework == "pytorch":
        import torch

        native_parameters = torch.tensor(parameters, dtype=torch.float64, device="cpu")
        native_quadratic = torch.tensor(quadratic, dtype=torch.float64, device="cpu")
        native_matrix = torch.tensor(matrix, dtype=torch.float64, device="cpu")
        native_coefficients = torch.tensor(coefficients, dtype=torch.float64, device="cpu")

        def scalar(values: Any) -> Any:
            """Evaluate actual scalar coupling with native torch operations."""
            return (
                torch.sin(torch.sum(values))
                + 0.5 * (values @ native_quadratic @ values)
                + 0.25 * torch.sum(values**4)
            )

        def vector(values: Any) -> Any:
            """Evaluate the native vector function with rectangular derivatives."""
            return native_matrix @ torch.sin(values) + native_coefficients * torch.sum(values**2)

        jacobians = [
            torch_jacfwd(torch, vector, native_parameters),
            torch_jacrev(torch, vector, native_parameters),
        ]
        curvatures = [
            torch_hessian(torch, scalar, native_parameters),
            torch_forward_gradient_jacobian(torch, scalar, native_parameters),
            torch_reverse_gradient_jacobian(torch, scalar, native_parameters),
        ]

        def torch_array(value: Any) -> NDArray[np.float64]:
            """Observe actual CPU float64 matrices without retaining their graphs."""
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

        array, native_dtype = torch_array, torch.float64
    else:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            native_parameters = tf.convert_to_tensor(parameters, dtype=tf.float64)
            native_quadratic = tf.convert_to_tensor(quadratic, dtype=tf.float64)
            native_matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
            native_coefficients = tf.convert_to_tensor(coefficients, dtype=tf.float64)
            with tf.GradientTape(persistent=True) as jacobian_tape:
                jacobian_tape.watch(native_parameters)
                vector_value = tf.linalg.matvec(native_matrix, tf.sin(native_parameters)) + (
                    native_coefficients * tf.reduce_sum(native_parameters**2)
                )

            def forward_product(direction: Any) -> Any:
                """Record the actual vector JVP with a caller-owned native accumulator."""
                with tf.autodiff.ForwardAccumulator(native_parameters, direction) as accumulator:
                    value = tf.linalg.matvec(native_matrix, tf.sin(native_parameters)) + (
                        native_coefficients * tf.reduce_sum(native_parameters**2)
                    )
                return accumulator.jvp(value)

            jacobians = [
                tensorflow_jacobian(jacobian_tape, vector_value, native_parameters),
                tensorflow_forward_jacobian(tf, forward_product, parameters[None, :]),
            ]
            with tf.GradientTape(persistent=True) as outer:
                outer.watch(native_parameters)
                with tf.GradientTape() as inner:
                    inner.watch(native_parameters)
                    scalar_value = (
                        tf.sin(tf.reduce_sum(native_parameters))
                        + 0.5
                        * tf.reduce_sum(
                            native_parameters
                            * tf.linalg.matvec(native_quadratic, native_parameters)
                        )
                        + 0.25 * tf.reduce_sum(native_parameters**4)
                    )
                gradient = inner.gradient(scalar_value, native_parameters)
            curvatures = [tensorflow_jacobian(outer, gradient, native_parameters)]
        array, native_dtype = np.asarray, tf.float64
    for actual in jacobians:
        assert actual.dtype == native_dtype
        np.testing.assert_allclose(array(actual), expected_jacobian, atol=1e-12, rtol=0)
    for actual in curvatures:
        assert actual.dtype == native_dtype
        np.testing.assert_allclose(array(actual), expected_hessian, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(parameters, original)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_public_producer_keeps_native_host_order_and_original_tapes(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every real producer while observing original expression/resource order.

    Parameters
    ----------
    framework
        Actual native public producer, including its registered custom rules.
    monkeypatch
        Scoped source observers that delegate all numerical and host operations.

    """
    native = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_" + native + "_transform_algebra"
    )
    names = {
        "jax": ["jax_hessian", "jax_jacfwd", "jax_jacrev", "jax_reverse_gradient_jacobian"],
        "pytorch": [
            "torch_hessian",
            "torch_jacfwd",
            "torch_jacrev",
            "torch_forward_gradient_jacobian",
            "torch_reverse_gradient_jacobian",
        ],
        "tensorflow": ["tensorflow_forward_jacobian", "tensorflow_jacobian"],
    }
    operations: list[tuple[str, str]] = []
    events: list[str] = []
    custom_tapes: list[weakref.ReferenceType[Any]] = []

    def observer(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Observe the actual numerical input/caller before executing its native kernel."""

        def execute(*arguments: Any) -> Any:
            frame = inspect.currentframe()
            try:
                assert frame is not None
                frame = frame.f_back
                assert frame is not None and frame.f_globals is vars(module)
                if framework == "tensorflow":
                    if name == "tensorflow_forward_jacobian":
                        objective = frame.f_locals["forward_gradient"]
                        assert objective is arguments[1]
                        assert objective.__globals__ is vars(module)
                        assert objective.__qualname__ == "_execute_cases.<locals>.forward_gradient"
                        assert objective.__code__.co_freevars == ("derivative", "tf", "x")
                        assert frame.f_locals["matrix"] is arguments[2]
                        label = "forward_gradient"
                    elif frame.f_code is module._execute_cases.__code__:
                        assert frame.f_locals["custom_outer"] is arguments[0]
                        custom_tapes.append(weakref.ref(arguments[0]))
                        label = "custom_outer"
                    else:
                        assert frame.f_code.co_qualname == "_execute_cases.<locals>.hessian"
                        assert frame.f_code.co_freevars == ("derivative", "tf")
                        assert frame.f_locals["tape"] is arguments[0]
                        assert frame.f_locals["gradient"] is arguments[1]
                        label = "hessian"
                elif name.endswith(("jacfwd", "jacrev")):
                    assert frame.f_locals["derivative"] is arguments[1]
                    label = "derivative"
                else:
                    objective = arguments[1]
                    assert inspect.isfunction(objective)
                    assert objective.__globals__ is vars(module)
                    assert (
                        objective.__qualname__ == "_execute_cases.<locals>." + objective.__name__
                    )
                    label = objective.__name__
            finally:
                del frame
            operations.append((name, label))
            events.append("native")
            return original(*arguments)

        return execute

    for name in names[framework]:
        monkeypatch.setattr(module, name, observer(name, getattr(module, name)))
    host_name = "asarray" if framework == "pytorch" else "ravel"
    original_host: Callable[..., Any] = getattr(np, host_name)

    def host(*arguments: Any, **keywords: Any) -> Any:
        """Observe actual producer host conversion without changing its result."""
        frame = inspect.currentframe()
        try:
            assert frame is not None
            frame = frame.f_back
            if frame is not None and frame.f_globals is vars(module):
                events.append("host")
        finally:
            del frame
        return original_host(*arguments, **keywords)

    monkeypatch.setattr(np, host_name, host)
    original_case: Callable[..., TransformAlgebraCase] = module._framework_transform_case

    def report(*arguments: Any) -> TransformAlgebraCase:
        """Retain the original custom outer tape throughout real reporting."""
        if framework == "tensorflow":
            assert len(custom_tapes) == 1
            assert custom_tapes[0]() is not None
        return original_case(*arguments)

    monkeypatch.setattr(module, "_framework_transform_case", report)
    producer = getattr(module, "run_" + native + "_transform_algebra_cases")
    records = producer(np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64))
    positions = [i for i, event in enumerate(events) if event == "native"]
    if framework == "tensorflow":
        assert (
            operations
            == [
                ("tensorflow_forward_jacobian", "forward_gradient"),
                ("tensorflow_jacobian", "custom_outer"),
            ]
            + [("tensorflow_jacobian", "hessian")] * 2
        )
        assert events[positions[-1] - 1] == events[positions[-1] + 1] == "host"
        gc.collect()
        assert custom_tapes[0]() is None
    else:
        expected = (
            [
                ("jax_hessian", "forward_composed"),
                ("jax_reverse_gradient_jacobian", "reverse_composed"),
            ]
            if framework == "jax"
            else [
                ("torch_forward_gradient_jacobian", "custom_composed"),
                ("torch_reverse_gradient_jacobian", "custom_composed"),
            ]
        )
        assert operations == expected + [
            (native + "_hessian", "loss"),
            (native + "_jacfwd", "derivative"),
            (native + "_jacrev", "derivative"),
        ]
        for position in [0, 1, 3, 4] if framework == "jax" else range(5):
            assert events[positions[position] + 1] == "host"
    assert len(records) == 11
    assert all(case.status == "passed" for case in records[:-1])
    assert records[-1].status == "blocked" and records[-1].blocked_reasons
