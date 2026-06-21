# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD broadcast assembly tests
"""Tests for Program AD broadcast assembly primitive semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import program_ad_broadcast_assembly as broadcast_assembly
from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_assembly_broadcast_arrays_derivative_rule,
    program_ad_assembly_broadcast_to_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_broadcast_assembly import (
    program_ad_assembly_broadcast_arrays_derivative_rule as extracted_broadcast_arrays_rule,
)
from scpn_quantum_control.program_ad_broadcast_assembly import (
    program_ad_assembly_broadcast_to_derivative_rule as extracted_broadcast_to_rule,
)

FloatArray = NDArray[np.float64]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_broadcast_direct_rules_are_exposed_from_extracted_module() -> None:
    """Facade and extracted broadcast assembly factories should share identity."""

    assert extracted_broadcast_to_rule is program_ad_assembly_broadcast_to_derivative_rule
    assert extracted_broadcast_arrays_rule is program_ad_assembly_broadcast_arrays_derivative_rule


def test_program_ad_broadcast_direct_rules_fail_closed_on_static_shapes() -> None:
    """Extracted broadcast direct rules should reject malformed static shapes."""

    assert broadcast_assembly._normalise_program_ad_broadcast_shape(3) == (3,)
    with pytest.raises(ValueError, match="requires an integer shape"):
        broadcast_assembly._normalise_program_ad_broadcast_shape("2")
    with pytest.raises(ValueError, match="non-negative"):
        broadcast_assembly._normalise_program_ad_broadcast_shape((2, -1))
    with pytest.raises(ValueError, match="compatible with source broadcasting"):
        extracted_broadcast_to_rule((2, 3), (2, 2))
    with pytest.raises(ValueError, match="requires operands"):
        extracted_broadcast_arrays_rule(())
    with pytest.raises(ValueError, match="broadcast-compatible operands"):
        extracted_broadcast_arrays_rule(((2,), (3,)))


def test_program_ad_broadcast_direct_adjoint_rejects_impossible_cotangents() -> None:
    """Extracted broadcast adjoint reductions should reject impossible cotangent layouts."""

    _assert_allclose(
        broadcast_assembly._program_ad_assembly_broadcast_adjoint(
            np.ones((2, 3), dtype=np.float64),
            source_shape=(),
        ),
        np.array(6.0, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="rank mismatch"):
        broadcast_assembly._program_ad_assembly_broadcast_adjoint(
            np.ones((2,), dtype=np.float64),
            source_shape=(2, 2),
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        broadcast_assembly._program_ad_assembly_broadcast_adjoint(
            np.ones((2, 3), dtype=np.float64),
            source_shape=(2, 2),
        )


def test_program_ad_assembly_broadcast_to_contract_and_direct_rule() -> None:
    """np.broadcast_to should expose exact static broadcast adjoint contracts."""

    values = np.array([1.0, -2.0], dtype=np.float64)
    tangent = np.array([0.25, -0.5], dtype=np.float64)
    cotangent = np.array([[0.2, -0.4], [0.6, -0.8], [1.0, -1.2]], dtype=np.float64)

    contract = primitive_contract_for("scpn.program_ad.assembly:broadcast_to")
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "broadcast_to", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.broadcast_to"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_broadcast_to_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "source_shape:ranked_tensor_shape;output_shape"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((values, (3, 2))) == (3, 2)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((values, (3, 2))) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((values, (3, 2))) == (values.shape, (3, 2))
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_broadcast_to_derivative_rule(values.shape, (3, 2))
    assert rule.name == "program_ad_assembly_broadcast_to_2_to_3x2_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    _assert_allclose(rule.value_fn(values), np.broadcast_to(values, (3, 2)).reshape(-1))
    _assert_allclose(
        jvp_rule(values, tangent),
        np.broadcast_to(tangent, (3, 2)).reshape(-1),
    )
    _assert_allclose(
        vjp_rule(values, cotangent.reshape(-1)),
        np.sum(cotangent, axis=0),
    )


def test_program_ad_assembly_broadcast_arrays_contract_and_direct_rule() -> None:
    """np.broadcast_arrays should expose exact per-operand scatter adjoints."""

    column = np.array([[1.0], [-2.0]], dtype=np.float64)
    row = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    scalar = np.array(1.5, dtype=np.float64)
    tangent_column = np.array([[0.1], [-0.2]], dtype=np.float64)
    tangent_row = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    tangent_scalar = np.array(-0.6, dtype=np.float64)
    values = np.concatenate([column.reshape(-1), row.reshape(-1), scalar.reshape(-1)])
    tangent = np.concatenate(
        [tangent_column.reshape(-1), tangent_row.reshape(-1), tangent_scalar.reshape(-1)]
    )
    cotangents = (
        np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float64),
        np.array([[0.7, -0.8, 0.9], [-1.0, 1.1, -1.2]], dtype=np.float64),
        np.array([[1.3, -1.4, 1.5], [-1.6, 1.7, -1.8]], dtype=np.float64),
    )
    cotangent = np.concatenate([item.reshape(-1) for item in cotangents])

    contract = primitive_contract_for("scpn.program_ad.assembly:broadcast_arrays")
    assert contract.identity == PrimitiveIdentity(
        "scpn.program_ad.assembly", "broadcast_arrays", "1"
    )
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.broadcast_arrays"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_broadcast_arrays_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "operand_shapes:ranked_tensor_shapes;output_shape"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((column, row, scalar)) == (18,)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((column, row, scalar)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((column, row, scalar)) == (
        (column.shape, row.shape, scalar.shape),
        (2, 3),
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_broadcast_arrays_derivative_rule(
        (column.shape, row.shape, scalar.shape)
    )
    assert rule.name == "program_ad_assembly_broadcast_arrays_3_operands_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    expected_values = np.concatenate(
        [item.reshape(-1) for item in np.broadcast_arrays(column, row, scalar)]
    )
    expected_tangent = np.concatenate(
        [
            item.reshape(-1)
            for item in np.broadcast_arrays(tangent_column, tangent_row, tangent_scalar)
        ]
    )
    expected_adjoint = np.concatenate(
        [
            np.sum(cotangents[0], axis=1, keepdims=True).reshape(-1),
            np.sum(cotangents[1], axis=0).reshape(-1),
            np.array([np.sum(cotangents[2])], dtype=np.float64),
        ]
    )
    _assert_allclose(rule.value_fn(values), expected_values)
    _assert_allclose(jvp_rule(values, tangent), expected_tangent)
    _assert_allclose(vjp_rule(values, cotangent), expected_adjoint)


def test_program_ad_assembly_broadcast_batching_rules_map_outer_axes() -> None:
    """Broadcast batching should map data axes and keep shape metadata static."""

    broadcast_to_contract = primitive_contract_for("scpn.program_ad.assembly:broadcast_to")
    broadcast_arrays_contract = primitive_contract_for("scpn.program_ad.assembly:broadcast_arrays")
    assert broadcast_to_contract.batching_rule is not None
    assert broadcast_arrays_contract.batching_rule is not None
    broadcast_to_batching_rule = broadcast_to_contract.batching_rule
    broadcast_arrays_batching_rule = broadcast_arrays_contract.batching_rule

    def broadcast_to_fn(source: FloatArray, shape: tuple[int, ...]) -> FloatArray:
        return cast(FloatArray, np.broadcast_to(source, shape))

    source_batch = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    expected_to = np.stack([np.broadcast_to(source_batch[index], (3, 2)) for index in range(2)])
    _assert_allclose(
        broadcast_to_batching_rule(broadcast_to_fn, (source_batch, (3, 2)), (0, None), 0),
        expected_to,
    )
    _assert_allclose(
        broadcast_to_batching_rule(broadcast_to_fn, (source_batch, (3, 2)), (0, None), 1),
        np.moveaxis(expected_to, 0, 1),
    )
    with pytest.raises(ValueError, match="keeps output shape static"):
        broadcast_to_batching_rule(broadcast_to_fn, (source_batch, (3, 2)), (0, 0), 0)

    def broadcast_arrays_fn(*operands: FloatArray) -> list[FloatArray]:
        return [cast(FloatArray, item) for item in np.broadcast_arrays(*operands)]

    column_batch = np.array([[[1.0], [-2.0]], [[0.5], [3.0]]], dtype=np.float64)
    row = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    scalar_batch = np.array([1.5, -2.5], dtype=np.float64)
    outputs = [
        np.broadcast_arrays(column_batch[index], row, scalar_batch[index]) for index in range(2)
    ]
    expected_arrays = tuple(
        np.stack([outputs[index][operand_index] for index in range(2)], axis=0)
        for operand_index in range(3)
    )
    result = broadcast_arrays_batching_rule(
        broadcast_arrays_fn,
        (column_batch, row, scalar_batch),
        (0, None, 0),
        0,
    )
    assert isinstance(result, tuple)
    for actual, expected in zip(result, expected_arrays, strict=True):
        _assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="same batch size"):
        broadcast_arrays_batching_rule(
            broadcast_arrays_fn,
            (column_batch, row, np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            (0, None, 0),
            0,
        )


def test_program_ad_broadcast_to_accumulates_repeated_adjoint_paths() -> None:
    """Program AD broadcast_to should accumulate derivatives from repeated uses."""

    weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    def objective(values: Any) -> object:
        broadcast = np.broadcast_to(values, (3, 2))
        return np.sum(broadcast * weights)

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0], dtype=np.float64),
        parameters=(Parameter("x0"), Parameter("x1")),
    )

    assert result.value == pytest.approx(33.0)
    _assert_allclose(result.gradient, [9.0, 12.0], atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_broadcast_to_rejects_subclass_propagation() -> None:
    """Program AD broadcast_to should fail closed on subclass propagation."""

    with pytest.raises(ValueError, match="subok"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.broadcast_to)(values, (2, 2), subok=True)),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_broadcast_arrays_accumulates_operand_adjoint_paths() -> None:
    """Program AD broadcast_arrays should gather each operand's repeated adjoints."""

    column_weights = np.array([[1.0, -2.0, 3.0], [0.5, -1.0, 2.0]], dtype=np.float64)
    row_weights = np.array([[-0.25, 0.75, 1.25], [2.0, -1.5, 0.5]], dtype=np.float64)
    scalar_weights = np.array([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=np.float64)

    def objective(values: Any) -> object:
        column = np.reshape(values[:2], (2, 1))
        row = values[2:5]
        scalar = values[5]
        broadcast_column, broadcast_row, broadcast_scalar = np.broadcast_arrays(
            column, row, scalar
        )
        return (
            np.sum(broadcast_column * column_weights)
            + np.sum(broadcast_row * row_weights)
            + np.sum(broadcast_scalar * scalar_weights)
        )

    result = whole_program_value_and_grad(
        objective,
        np.arange(1.0, 7.0, dtype=np.float64),
        parameters=tuple(Parameter(f"x{index}") for index in range(6)),
    )

    expected_gradient = np.array([2.0, 1.5, 1.75, -0.75, 1.75, -0.6], dtype=np.float64)
    expected_value = float(cast(Any, objective(np.arange(1.0, 7.0, dtype=np.float64))))
    assert result.value == pytest.approx(expected_value)
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_broadcast_arrays_rejects_invalid_contracts() -> None:
    """Program AD broadcast_arrays should fail closed on unsupported broadcast contracts."""

    with pytest.raises(ValueError, match="broadcasting rules"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.broadcast_arrays(values[:2], values[2:5])[0]),
            np.arange(1.0, 7.0, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="subok"):
        whole_program_value_and_grad(
            lambda values: np.sum(
                cast(Any, np.broadcast_arrays)(values[:2], values[:2], subok=True)[0]
            ),
            np.arange(1.0, 7.0, dtype=np.float64),
        )
