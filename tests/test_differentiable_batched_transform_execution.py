# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native batched derivative contracts
"""Qualify real native shared-input batch derivatives and producer callback scopes."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_batched_transform_execution import (
    jax_batched_gradient,
    tensorflow_vectorized_derivatives,
    torch_batched_gradient,
)

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("batch_axis", [0, 1])
def test_native_shared_bias_derivatives_remain_per_sample(
    framework: str, width: int, batch_axis: int
) -> None:
    """Compare genuine vectorised derivatives with independent coupled derivatives.

    Parameters
    ----------
    framework
        Required actual CPU runtime; missing imports fail qualification.
    width
        Number of independent weights in each of three samples.
    batch_axis
        Original source axis containing samples, with the bias shared.

    """
    rows = np.linspace(-0.4, 0.7, 3 * width, dtype=np.float64).reshape(3, width)
    original = rows.copy()
    bias = -0.2
    oriented = rows if batch_axis == 0 else rows.T
    expected_weights = np.cos(np.sum(rows, axis=1))[:, None] + 2.0 * bias * rows
    expected_bias = np.sum(rows**2, axis=1) + 3.0 * bias**2
    axes = ({"weights": batch_axis, "bias": None},)
    array: Callable[[Any], NDArray[np.float64]]
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        def jax_objective(parameters: dict[str, Any]) -> Any:
            """Evaluate the coupled scalar with native JAX primitives."""
            w, b = parameters["weights"], parameters["bias"]
            return jnp.sin(jnp.sum(w)) + b * jnp.sum(w**2) + b**3

        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            parameters = {
                "weights": jnp.asarray(oriented, dtype=jnp.float64),
                "bias": jnp.asarray(bias, dtype=jnp.float64),
            }
            gradient = jax_batched_gradient(jax, jax_objective, parameters, axes)
            assert gradient["weights"].dtype == np.float64
            assert gradient["bias"].dtype == np.float64
        array = np.asarray
    elif framework == "pytorch":
        import torch

        def torch_objective(parameters: dict[str, Any]) -> Any:
            """Evaluate the coupled scalar with native torch primitives."""
            w, b = parameters["weights"], parameters["bias"]
            return torch.sin(torch.sum(w)) + b * torch.sum(w**2) + b**3

        parameters = {
            "weights": torch.tensor(oriented, dtype=torch.float64, device="cpu"),
            "bias": torch.tensor(bias, dtype=torch.float64, device="cpu"),
        }
        gradient = torch_batched_gradient(torch, torch_objective, parameters, axes)
        assert gradient["weights"].dtype == torch.float64
        assert gradient["bias"].dtype == torch.float64

        def torch_array(value: Any) -> NDArray[np.float64]:
            """Observe the native CPU float64 derivative without retaining its graph."""
            return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)

        array = torch_array
    else:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            shared_bias = tf.convert_to_tensor(bias, dtype=tf.float64)
            raw = tf.convert_to_tensor(oriented, dtype=tf.float64)
            batch = raw if batch_axis == 0 else tf.transpose(raw)

            def tensorflow_derivative(weights: Any) -> dict[str, Any]:
                """Differentiate each row with the caller-owned shared bias and tape."""
                with tf.GradientTape() as tape:
                    tape.watch((weights, shared_bias))
                    value = (
                        tf.sin(tf.reduce_sum(weights))
                        + shared_bias * tf.reduce_sum(weights**2)
                        + shared_bias**3
                    )
                weight_gradient, bias_gradient = tape.gradient(value, (weights, shared_bias))
                return {"weights": weight_gradient, "bias": bias_gradient}

            gradient = tensorflow_vectorized_derivatives(tf, tensorflow_derivative, batch)
            assert gradient["weights"].dtype == tf.float64
            assert gradient["bias"].dtype == tf.float64
        array = np.asarray
    np.testing.assert_allclose(array(gradient["weights"]), expected_weights, atol=1e-12, rtol=0)
    np.testing.assert_allclose(array(gradient["bias"]), expected_bias, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(rows, original)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_native_producers_keep_original_structured_and_custom_objective_scopes(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe original callbacks while every delegated operation executes real AD.

    Parameters
    ----------
    framework
        Real native public producer whose mixed source boundary is exercised.
    monkeypatch
        Scoped observation of the actual numerical callback and its defining module.

    """
    native_name = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_" + native_name + "_transform_algebra"
    )
    operation_name = (
        "tensorflow_vectorized_derivatives"
        if framework == "tensorflow"
        else native_name + "_batched_gradient"
    )
    original: Callable[..., Any] = getattr(module, operation_name)
    calls: list[str] = []

    def observe(backend: Any, objective: Any, *arguments: Any) -> Any:
        """Read the genuine callback's source scope and execute the real operation."""
        assert inspect.isfunction(objective)
        assert objective.__globals__ is vars(module)
        assert objective.__module__ == module.__name__
        assert objective.__qualname__ == "_execute_cases.<locals>." + objective.__name__
        calls.append(objective.__name__)
        return original(backend, objective, *arguments)

    monkeypatch.setattr(module, operation_name, observe)
    producer = getattr(module, "run_" + native_name + "_transform_algebra_cases")
    records = producer(np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64))
    expected_callbacks = {
        "jax": ["structured_loss", "forward_composed", "reverse_composed"],
        "pytorch": ["structured_loss", "custom_composed"],
        "tensorflow": ["structured_batch_gradient", "custom_batch_gradient"],
    }
    assert calls == expected_callbacks[framework]
    assert len(records) == 11
    assert all(record.status == "passed" for record in records[:-1])
    assert records[-1].status == "blocked"
    assert records[-1].blocked_reasons
