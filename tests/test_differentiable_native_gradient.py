# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native gradient contracts
"""Qualify actual gradient reuse and original producer callback and tape ownership."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_native_gradient import (
    jax_gradient,
    tensorflow_gradient,
    torch_gradient,
)
from scpn_quantum_control.differentiable_transform_algebra import TransformAlgebraCase

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_native_gradients_reuse_coupled_objective_and_watched_inputs(
    framework: str, width: int
) -> None:
    """Compare reused native gradients with independent coupled analytic values.

    Parameters
    ----------
    framework
        Required actual CPU runtime; missing imports fail the test.
    width
        Number of coupled scalar-objective inputs.

    """
    values = np.linspace(-0.3, 0.4, width, dtype=np.float64)
    original = values.copy()
    arrays: list[NDArray[np.float64]]
    if framework == "jax":
        import jax
        import jax.numpy as jnp

        def objective(parameters: Any) -> Any:
            """Evaluate the native JAX coupled scalar objective."""
            return (
                jnp.sin(jnp.sum(parameters))
                + 0.25 * jnp.sum(parameters**4)
                + 0.2 * jnp.sum(parameters**2)
            )

        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            parameters = jnp.asarray(values, dtype=jnp.float64)
            derivative = jax_gradient(jax, objective)
            arrays = [np.asarray(derivative(parameters)), np.asarray(derivative(2.0 * parameters))]
    elif framework == "pytorch":
        import torch

        def objective(parameters: Any) -> Any:
            """Evaluate the native PyTorch coupled scalar objective."""
            return (
                torch.sin(torch.sum(parameters))
                + 0.25 * torch.sum(parameters**4)
                + 0.2 * torch.sum(parameters**2)
            )

        parameters = torch.tensor(values, dtype=torch.float64, device="cpu")
        derivative = torch_gradient(torch, objective)
        arrays = [
            np.asarray(derivative(primal).detach().cpu().numpy(), dtype=np.float64)
            for primal in [parameters, 2.0 * parameters]
        ]
    else:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            parameters = tf.convert_to_tensor(values, dtype=tf.float64)
            other = 2.0 * parameters
            with tf.GradientTape(persistent=True) as tape:
                tape.watch((parameters, other))
                value = (
                    tf.sin(tf.reduce_sum(parameters))
                    + 0.25 * tf.reduce_sum(parameters**4)
                    + 0.2 * tf.reduce_sum(parameters**2)
                )
                other_value = (
                    tf.sin(tf.reduce_sum(other))
                    + 0.25 * tf.reduce_sum(other**4)
                    + 0.2 * tf.reduce_sum(other**2)
                )
            arrays = [
                np.asarray(tensorflow_gradient(tape, value, parameters)),
                np.asarray(tensorflow_gradient(tape, other_value, other)),
            ]
    for scale, actual in zip([1.0, 2.0], arrays, strict=True):
        primal = scale * values
        expected = np.cos(np.sum(primal)) + primal**3 + 0.4 * primal
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_public_producers_retain_native_gradient_binding_and_original_tapes(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe real producer gradients while delegating every native operation.

    Parameters
    ----------
    framework
        Actual public native producer, including its registered custom rules.
    monkeypatch
        Scoped observers that retain all real numerical results.

    """
    native = "torch" if framework == "pytorch" else framework
    module = importlib.import_module(
        "scpn_quantum_control.differentiable_" + native + "_transform_algebra"
    )
    operation = native + "_gradient"
    original: Callable[..., Any] = getattr(module, operation)
    results: list[Any] = []
    callbacks: list[str] = []
    tape_scopes: list[str] = []

    def observe(*arguments: Any) -> Any:
        """Read original caller identities before executing the actual native gradient."""
        frame = inspect.currentframe()
        try:
            assert frame is not None
            frame = frame.f_back
            assert frame is not None and frame.f_globals is vars(module)
            if framework == "tensorflow":
                assert any(
                    frame.f_locals.get(name) is arguments[0]
                    for name in ["tape", "aggregate_tape", "custom_inner"]
                )
                assert any(
                    frame.f_locals.get(name) is arguments[1]
                    for name in ["value", "aggregate_loss", "custom_value"]
                )
                tape_scopes.append(frame.f_code.co_qualname)
            else:
                objective = arguments[1]
                assert objective.__globals__ is vars(module)
                assert objective.__qualname__ == "_execute_cases.<locals>." + objective.__name__
                callbacks.append(objective.__name__)
        finally:
            del frame
        result = original(*arguments)
        results.append(result)
        return result

    monkeypatch.setattr(module, operation, observe)
    original_case: Callable[..., TransformAlgebraCase] = module._framework_transform_case
    reports: list[str] = []

    def report(*arguments: Any) -> TransformAlgebraCase:
        """Verify the original producer keeps the exact native derivative binding."""
        if framework != "tensorflow":
            frame = inspect.currentframe()
            try:
                assert frame is not None
                frame = frame.f_back
                while frame is not None and frame.f_code is not module._execute_cases.__code__:
                    frame = frame.f_back
                assert frame is not None
                assert frame.f_locals["derivative"] is results[0]
            finally:
                del frame
        reports.append("case")
        return original_case(*arguments)

    monkeypatch.setattr(module, "_framework_transform_case", report)
    records = getattr(module, "run_" + native + "_transform_algebra_cases")(
        np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
    )
    if framework == "tensorflow":
        assert set(tape_scopes) == {
            "_execute_cases",
            "_execute_cases.<locals>.derivative",
            "_execute_cases.<locals>.structured_batch_gradient",
            "_execute_cases.<locals>.custom_batch_gradient",
        }
    else:
        assert callbacks == (
            ["loss", "aggregate", "forward_composed", "reverse_composed"]
            if framework == "jax"
            else ["loss", "aggregate", "custom_composed"]
        )
    assert len(reports) == 10
    assert [case.status for case in records] == ["passed"] * 10 + ["blocked"]
    assert records[-1].blocked_reasons
