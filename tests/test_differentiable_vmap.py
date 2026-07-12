# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable vmap tests
# scpn-quantum-control -- canonical vmap transform tests
"""Tests for the canonical differentiable-programming vectorization transform."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import grad, vmap
from scpn_quantum_control.differentiable_vmap import (
    _normalise_vmap_in_axes,
    _stack_vmap_outputs,
    _trace_array_ndim,
    _trace_array_shape,
    _trace_value_context,
)
from scpn_quantum_control.differentiable_vmap import (
    vmap as module_vmap,
)
from scpn_quantum_control.program_ad_registry import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)
from scpn_quantum_control.whole_program_trace_runtime import _WholeProgramTraceContext

FloatArray = NDArray[np.float64]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed vmap result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_vmap_vectorizes_single_argument_scalar_objective() -> None:
    """Canonical vmap should map scalar objectives over a selected input axis."""

    batched = vmap(lambda row: row[0] ** 2 + 3.0 * row[1])
    values = np.array([[2.0, -1.0], [0.5, 4.0], [-2.0, 3.0]], dtype=np.float64)

    _assert_allclose(batched(values), [1.0, 12.25, 13.0], atol=1.0e-12)


def test_vmap_facade_delegates_to_extracted_module() -> None:
    """The differentiable facade should expose the module-owned vmap transform."""

    assert vmap is module_vmap
    assert vars(differentiable)["vmap"] is module_vmap
    assert _normalise_vmap_in_axes((0, None), 2) == (0, None)


def test_vmap_supports_broadcast_arguments_out_axes_and_nested_outputs() -> None:
    """vmap should preserve nested output structure and broadcast static inputs."""

    def affine(row: FloatArray, weights: FloatArray, bias: float) -> dict[str, object]:
        projection = row * weights + bias
        return {
            "projection": projection,
            "summary": (np.array([float(np.sum(projection))], dtype=np.float64), [projection[:1]]),
        }

    batched = vmap(affine, in_axes=(0, None, None), out_axes=1)
    values = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    weights = np.array([2.0, -0.5], dtype=np.float64)

    result = batched(values, weights, 0.25)
    result_payload = cast(dict[str, Any], result)

    _assert_allclose(result_payload["projection"], [[2.25, 6.25], [-0.75, 0.75]])
    _assert_allclose(result_payload["summary"][0], [[1.5, 7.0]])
    _assert_allclose(result_payload["summary"][1][0], [[2.25, 6.25]])


def test_vmap_dispatches_registered_primitive_batching_rule() -> None:
    """Primitive identities should route vmap through registered batching rules."""

    identity = PrimitiveIdentity("scpn.test.vmap", "batch")
    registry = CustomDerivativeRegistry()

    def batching_rule(
        function: object,
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> dict[str, object]:
        return {"function": function, "args": args, "axes": axes, "out_axes": out_axes}

    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=CustomDerivativeRule(
                name="test_vmap_batching_rule",
                value_fn=lambda values: values,
                jvp_rule=lambda _values, tangent: tangent,
            ),
            batching_rule=batching_rule,
        )
    )

    def function(value: object) -> object:
        return value

    result = vmap(
        function,
        primitive_identity=identity,
        registry=registry,
        in_axes=(0,),
        out_axes=-1,
    )(np.array([1.0, 2.0], dtype=np.float64))
    payload = cast(dict[str, object], result)

    assert payload["function"] is function
    assert payload["axes"] == (0,)
    assert payload["out_axes"] == -1


def test_vmap_preserves_whole_program_trace_values() -> None:
    """vmap should slice and stack facade-owned whole-program trace arrays lazily."""

    context = _WholeProgramTraceContext(2, scalar_factory=differentiable.TraceADScalar)
    first = context.make("parameter", ("x0",), 2.0, np.array([1.0, 0.0], dtype=np.float64))
    second = context.make("parameter", ("x1",), 3.0, np.array([0.0, 1.0], dtype=np.float64))
    trace_array = differentiable.TraceADArray((first, second), (2,), context, (0, 1))

    result = vmap(lambda value: value * 2.0)(trace_array)

    assert isinstance(result, differentiable.TraceADArray)
    assert result.shape == (2,)
    result_items = cast(
        tuple[differentiable.TraceADScalar, ...], tuple(cast(Iterable[object], result))
    )
    _assert_allclose([item.primal for item in result_items], [4.0, 6.0])
    _assert_allclose(
        [item.tangent for item in result_items],
        [[2.0, 0.0], [0.0, 2.0]],
    )


def test_vmap_composes_with_grad_transform() -> None:
    """vmap should compose with canonical gradient transforms over batches."""

    batched_grad = vmap(
        lambda row: grad(
            lambda values: values[0] ** 2 + np.sin(values[1]), row, method="finite_difference"
        )
    )
    values = np.array([[2.0, 0.0], [-1.5, 0.25]], dtype=np.float64)

    _assert_allclose(
        batched_grad(values),
        [[4.0, 1.0], [-3.0, math.cos(0.25)]],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_vmap_rejects_invalid_axes_and_ragged_outputs() -> None:
    """vmap should fail closed on ambiguous vectorization contracts."""

    with pytest.raises(ValueError, match="must be callable"):
        vmap(cast(Any, 1.0))
    with pytest.raises(ValueError, match="out_axes must be an integer"):
        vmap(lambda value: value, out_axes=cast(Any, 0.5))
    with pytest.raises(ValueError, match="requires at least one argument"):
        vmap(lambda: 1.0)()
    with pytest.raises(ValueError, match="in_axes length"):
        vmap(lambda lhs, rhs: lhs + rhs, in_axes=(0,))(1.0, 2.0)
    with pytest.raises(ValueError, match="integers or None"):
        vmap(lambda value: value, in_axes=(cast(Any, "axis"),))(np.array([1.0]))
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        vmap(lambda value: value)(1.0)
    with pytest.raises(ValueError, match="out of bounds"):
        vmap(lambda value: value, in_axes=2)(np.array([1.0]))
    with pytest.raises(ValueError, match="non-empty"):
        vmap(lambda value: value)(np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="same length"):
        vmap(lambda lhs, rhs: lhs + rhs)(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    _assert_allclose(
        vmap(lambda lhs, rhs: lhs + rhs)(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        ),
        [4.0, 6.0],
    )
    with pytest.raises(ValueError, match="at least one"):
        vmap(lambda value: value, in_axes=None)(1.0)
    with pytest.raises(ValueError, match="consistent shapes"):
        vmap(lambda value: np.ones(int(value), dtype=np.float64))(
            np.array([1.0, 2.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="out_axes is out of bounds"):
        vmap(lambda value: value, out_axes=2)(np.array([1.0, 2.0], dtype=np.float64))
    _assert_allclose(
        _stack_vmap_outputs(
            [np.array([1.0, 2.0], dtype=np.float64), np.array([3.0, 4.0], dtype=np.float64)],
            -1,
        ),
        [[1.0, 3.0], [2.0, 4.0]],
    )
    with pytest.raises(ValueError, match="consistent structure"):
        vmap(lambda value: (value,) if value > 0 else (value, value))(
            np.array([1.0, -1.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="consistent structure"):
        vmap(lambda value: [value] if value > 0 else [value, value])(
            np.array([1.0, -1.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="consistent keys"):
        vmap(lambda value: {"a": value} if value > 0 else {"b": value})(
            np.array([1.0, -1.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="must be numeric"):
        _stack_vmap_outputs([True, False], 0)
    with pytest.raises(ValueError, match="must be non-empty"):
        _stack_vmap_outputs([], 0)
    with pytest.raises(ValueError, match="numeric arrays"):
        _stack_vmap_outputs([object(), object()], 0)
    with pytest.raises(ValueError, match="static shape"):
        _trace_array_shape(object())
    with pytest.raises(ValueError, match="trace context"):
        _trace_value_context(object())

    context = _WholeProgramTraceContext(2, scalar_factory=differentiable.TraceADScalar)
    scalar = context.make("parameter", ("x",), 1.0, np.array([1.0, 0.0], dtype=np.float64))
    first_trace = differentiable.TraceADArray((scalar,), (1,), context, (0,))
    second_trace = differentiable.TraceADArray((scalar, scalar), (2,), context, (0, 0))
    with pytest.raises(ValueError, match="consistent shapes"):
        _stack_vmap_outputs([first_trace, second_trace], 0)

    class TraceADArray:
        """Structural trace-array stand-in without an ``ndim`` attribute."""

        context: object
        shape: tuple[int, ...]

    fake_trace = TraceADArray()
    fake_trace.context = object()
    fake_trace.shape = (2,)
    assert _trace_array_ndim(fake_trace) == 1


def test_vmap_is_exported_from_package_root() -> None:
    """The vectorization transform should be stable as a root-level API."""

    import scpn_quantum_control as scpn

    assert scpn.vmap is vmap
