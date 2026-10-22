# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native structured derivative contracts
"""Qualify mixed structured derivatives and original native callback lifetimes."""

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

from scpn_quantum_control.differentiable_structured_transform_execution import (
    jax_structured_derivatives,
    tensorflow_structured_gradient,
    tensorflow_structured_hessian,
    torch_structured_derivatives,
)
from scpn_quantum_control.differentiable_transform_algebra import TransformAlgebraCase

pytestmark = pytest.mark.framework_imports
FloatArray = NDArray[np.float64]


def _references(weights: FloatArray, bias: float) -> tuple[FloatArray, float, FloatArray]:
    """Return independent derivatives of sin(sum(weights)) plus bias-weight coupling."""
    total = float(np.sum(weights))
    gradient = np.cos(total) + 2.0 * bias * weights
    bias_gradient = float(weights @ weights + 3.0 * bias**2)
    hessian = -np.sin(total) * np.ones((weights.size, weights.size), dtype=np.float64)
    hessian += 2.0 * bias * np.eye(weights.size, dtype=np.float64)
    return gradient, bias_gradient, hessian


def _assert_native_blocks(
    gradient: dict[str, Any],
    hessian: dict[str, dict[str, Any]],
    weights: FloatArray,
    bias: float,
    array: Callable[[Any], FloatArray],
) -> None:
    """Compare every native gradient and mixed Hessian block with analytic derivatives."""
    expected_gradient, expected_bias, expected_hessian = _references(weights, bias)
    np.testing.assert_allclose(array(gradient["weights"]), expected_gradient, atol=1e-12, rtol=0)
    np.testing.assert_allclose(array(gradient["bias"]), expected_bias, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        array(hessian["weights"]["weights"]), expected_hessian, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        array(hessian["weights"]["bias"]), 2.0 * weights, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        array(hessian["bias"]["weights"]), 2.0 * weights, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(array(hessian["bias"]["bias"]), 6.0 * bias, atol=1e-12, rtol=0)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_structured_native_operators_match_independent_coupled_derivatives(
    framework: str, width: int
) -> None:
    """Exercise actual native operators on a coupled non-quartic numerical objective.

    Parameters
    ----------
    framework
        Required real CPU runtime, with no diagnostic fallback or optional skip.
    width
        Number of independent weights coupled to a shared scalar bias.

    """
    weights = np.linspace(-0.3, 0.4, width, dtype=np.float64)
    original = weights.copy()
    bias = -0.2
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        def jax_objective(parameters: dict[str, Any]) -> Any:
            """Evaluate the actual coupled scalar using native JAX primitives."""
            w, b = parameters["weights"], parameters["bias"]
            return jnp.sin(jnp.sum(w)) + b * jnp.sum(w**2) + b**3

        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            parameters = {
                "weights": jnp.asarray(weights, dtype=jnp.float64),
                "bias": jnp.asarray(bias, dtype=jnp.float64),
            }
            gradient, hessian = jax_structured_derivatives(jax, jax_objective, parameters)
            _assert_native_blocks(gradient, hessian, weights, bias, np.asarray)
    elif framework == "pytorch":
        import torch

        def torch_objective(parameters: dict[str, Any]) -> Any:
            """Evaluate the actual coupled scalar with native torch operations."""
            w, b = parameters["weights"], parameters["bias"]
            return torch.sin(torch.sum(w)) + b * torch.sum(w**2) + b**3

        parameters = {
            "weights": torch.tensor(weights, dtype=torch.float64, device="cpu"),
            "bias": torch.tensor(bias, dtype=torch.float64, device="cpu"),
        }
        gradient, hessian = torch_structured_derivatives(torch, torch_objective, parameters)
        _assert_native_blocks(
            gradient, hessian, weights, bias, lambda value: value.detach().cpu().numpy()
        )
    else:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            parameters = {
                "weights": tf.convert_to_tensor(weights, dtype=tf.float64),
                "bias": tf.convert_to_tensor(bias, dtype=tf.float64),
            }
            with tf.GradientTape(persistent=True) as outer:
                outer.watch(parameters)
                with tf.GradientTape() as inner:
                    inner.watch(parameters)
                    w, b = parameters["weights"], parameters["bias"]
                    value = tf.sin(tf.reduce_sum(w)) + b * tf.reduce_sum(w**2) + b**3
                gradient, flat = tensorflow_structured_gradient(tf, inner, value, parameters)
            weights_hessian, bias_hessian = tensorflow_structured_hessian(outer, flat, parameters)
        expected_gradient, expected_bias, expected_hessian = _references(weights, bias)
        np.testing.assert_allclose(gradient["weights"], expected_gradient, atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient["bias"], expected_bias, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            flat, np.concatenate([expected_gradient, [expected_bias]]), atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            weights_hessian,
            np.vstack([expected_hessian, 2.0 * weights]),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            bias_hessian, np.concatenate([2.0 * weights, [6.0 * bias]]), atol=1e-12, rtol=0
        )
    np.testing.assert_array_equal(weights, original)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_original_native_producer_preserves_fixture_scope_and_tape_lifetime(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe genuine callbacks/resources while every operator executes its real kernel.

    Parameters
    ----------
    framework
        Real native producer whose original source contract is exercised.
    monkeypatch
        Scoped source observers that delegate to the actual numerical operators.

    """
    native_name = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_" + native_name + "_transform_algebra"
    )
    operation_name = (
        "tensorflow_structured_gradient"
        if framework == "tensorflow"
        else native_name + "_structured_derivatives"
    )
    original: Callable[..., tuple[Any, Any]] = getattr(module, operation_name)
    calls: list[str] = []
    tapes: list[weakref.ReferenceType[Any]] = []

    def observe(backend: Any, objective_or_tape: Any, *arguments: Any) -> tuple[Any, Any]:
        """Inspect the actual defining scope or caller-owned tape and execute real AD."""
        if framework == "tensorflow":
            tapes.append(weakref.ref(objective_or_tape))
        else:
            assert inspect.isfunction(objective_or_tape)
            assert objective_or_tape.__globals__ is vars(module)
            assert objective_or_tape.__module__ == module.__name__
            assert objective_or_tape.__qualname__ == "_execute_cases.<locals>.structured_loss"
            assert objective_or_tape.__code__.co_freevars == (
                "jnp" if framework == "jax" else "torch",
            )
        calls.append(framework)
        return original(backend, objective_or_tape, *arguments)

    monkeypatch.setattr(module, operation_name, observe)
    record_calls: list[str] = []
    if framework == "tensorflow":
        original_case: Callable[..., TransformAlgebraCase] = module._framework_transform_case

        def consume(*arguments: Any) -> TransformAlgebraCase:
            """Require the original structured tape to stay alive through real reporting."""
            assert len(tapes) == 1
            assert tapes[0]() is not None
            record_calls.append("case")
            return original_case(*arguments)

        monkeypatch.setattr(module, "_framework_transform_case", consume)
    producer = getattr(module, "run_" + native_name + "_transform_algebra_cases")
    records = producer(np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64))
    assert calls == [framework]
    assert len(records) == 11
    assert all(record.status == "passed" for record in records[:-1])
    assert records[-1].status == "blocked"
    assert records[-1].blocked_reasons
    if framework == "tensorflow":
        assert len(record_calls) == 10
        gc.collect()
        assert tapes[0]() is None
