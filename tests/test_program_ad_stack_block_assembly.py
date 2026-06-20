# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD stack and block assembly tests
"""Tests for Program AD concatenate, stack, append, and block assembly semantics."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_assembly_append_derivative_rule,
    program_ad_assembly_block_derivative_rule,
    program_ad_assembly_column_stack_derivative_rule,
    program_ad_assembly_concatenate_derivative_rule,
    program_ad_assembly_dstack_derivative_rule,
    program_ad_assembly_hstack_derivative_rule,
    program_ad_assembly_stack_derivative_rule,
    program_ad_assembly_vstack_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AssemblyFactory = Callable[..., CustomDerivativeRule]
AssemblyNumpyFn = Callable[..., FloatArray]


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_concatenate_and_stack_general_static_axes() -> None:
    """Program AD should preserve exact adjoints for static concatenate and stack axes."""

    concat_weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    stack_weights = np.array(
        [
            [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]],
            [[-0.5, 1.25], [0.75, -1.5], [2.5, -0.25]],
        ],
        dtype=np.float64,
    )
    flat_weights = np.array([-0.25, 0.5, -1.5, 2.0, 0.75, -0.5], dtype=np.float64)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        column_assembled = np.concatenate((matrix[:, 2:], matrix[:, :1], matrix[:, 1:2]), axis=1)
        depth_stacked = np.stack((matrix, matrix[:, ::-1]), axis=2)
        flat_assembled = np.concatenate((matrix[:, :1], matrix[:, 1:]), axis=None)
        return (
            np.sum(column_assembled * concat_weights)
            + np.sum(depth_stacked * stack_weights)
            + np.sum(flat_assembled * flat_weights)
        )

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=tuple(Parameter(f"x{index}") for index in range(6)),
    )

    assert result.value == pytest.approx(107.0)
    _assert_allclose(result.gradient, [4.25, 3.25, 1.25, 4.75, 6.0, 7.25])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_stack_conveniences_preserve_assembly_adjoint() -> None:
    """Program AD stack conveniences should lower to exact assembly adjoints."""

    def objective(values: Any) -> object:
        left = values[:2]
        right = values[2:]
        row_left = np.reshape(left, (1, 2))
        row_right = np.reshape(right, (1, 2))
        hstacked = np.hstack((left, right))
        vstacked = np.vstack((left, right))
        columned = np.column_stack((left, right))
        dstacked = np.dstack((row_left, row_right))
        return (
            np.sum(hstacked * np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float64))
            + np.sum(vstacked * np.array([[1.5, -2.0], [0.25, 3.0]], dtype=np.float64))
            + np.sum(columned * np.array([[-0.75, 1.25], [2.5, -1.5]], dtype=np.float64))
            + np.sum(dstacked * np.array([[[0.2, -0.4], [0.6, -0.8]]], dtype=np.float64))
        )

    values = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_value = objective(values)
    expected_gradient = np.array([1.45, 0.1, 3.1, 0.45], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, expected_value)))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_stack_conveniences_reject_incompatible_shapes() -> None:
    """Program AD stack conveniences should fail closed on incompatible shapes."""

    with pytest.raises(ValueError, match="shape-compatible"):
        whole_program_value_and_grad(
            lambda values: np.hstack((np.reshape(values[:4], (2, 2)), values[4:])),
            np.arange(1.0, 7.0, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="shape-compatible"):
        whole_program_value_and_grad(
            lambda values: np.column_stack((values[:2], values[2:])),
            np.arange(1.0, 6.0, dtype=np.float64),
        )


def test_program_ad_block_preserves_nested_assembly_adjoint() -> None:
    """Program AD np.block should preserve nested assembly adjoints exactly."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:4], (2, 2))
        side = np.reshape(values[4:], (2, 1))
        bottom_left = np.reshape(values[:2], (1, 2))
        bottom_right = np.reshape(values[2:3], (1, 1))
        assembled = np.block([[matrix, side], [bottom_left, bottom_right]])
        weights = np.array(
            [[0.5, -1.0, 2.0], [1.5, -0.25, 0.75], [-0.5, 1.25, -2.0]],
            dtype=np.float64,
        )
        return np.sum(assembled * weights)

    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([0.0, 0.25, -0.5, -0.25, 2.0, 0.75], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_block_rejects_incompatible_nested_layouts() -> None:
    """Program AD np.block should fail closed on invalid nested layouts."""

    with pytest.raises(ValueError, match="shape-compatible"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.block([[np.reshape(values[:4], (2, 2)), values[4:5]]])),
            np.arange(1.0, 6.0, dtype=np.float64),
        )


def test_program_ad_assembly_concatenate_contract_and_direct_rule() -> None:
    """np.concatenate should expose a fail-closed assembly primitive direct rule."""

    left = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    right = np.array([[0.25], [-0.75]], dtype=np.float64)
    tangent_left = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    tangent_right = np.array([[-0.5], [0.25]], dtype=np.float64)
    cotangent = np.array([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=np.float64)
    values = np.concatenate([left.reshape(-1), right.reshape(-1)])
    tangent = np.concatenate([tangent_left.reshape(-1), tangent_right.reshape(-1)])

    contract = primitive_contract_for("scpn.program_ad.assembly:concatenate")
    contract = cast(Any, contract)
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "concatenate", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.concatenate"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_concatenate_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "operand_shapes:ranked_tensor_shapes;axis"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule(((left, right), 1)) == (2, 3)
    assert contract.shape_rule(((left, right), None)) == (6,)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule(((left, right), 1)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule(((left, right), 1)) == (((2, 2), (2, 1)), 1)
    assert contract.static_argument_rule(((left, right), None)) == (((2, 2), (2, 1)), None)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_concatenate_derivative_rule((left.shape, right.shape), axis=1)
    rule = cast(Any, rule)
    assert rule.name == "program_ad_assembly_concatenate_2_operands_axis1_direct_rule"
    _assert_allclose(
        rule.value_fn(values),
        np.concatenate([left, right], axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.jvp_rule(values, tangent),
        np.concatenate([tangent_left, tangent_right], axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate([cotangent[:, :2].reshape(-1), cotangent[:, 2:].reshape(-1)]),
    )


def test_program_ad_assembly_concatenate_batching_rule_maps_operand_batches() -> None:
    """Concatenate batching should map operand batches and keep axis static."""

    contract = primitive_contract_for("scpn.program_ad.assembly:concatenate")
    contract = cast(Any, contract)
    assert contract.batching_rule is not None

    def concatenate_fn(operands: tuple[FloatArray, ...], axis: int | None) -> FloatArray:
        return cast(FloatArray, np.concatenate(operands, axis=axis))

    left_batch = np.array(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-1.5, 0.25], [2.0, -0.75]],
        ],
        dtype=np.float64,
    )
    right_batch = np.array(
        [
            [[0.25], [-0.75]],
            [[1.5], [-2.5]],
        ],
        dtype=np.float64,
    )
    expected = np.stack(
        [
            np.concatenate([left_batch[index], right_batch[index]], axis=1)
            for index in range(left_batch.shape[0])
        ],
        axis=0,
    )

    _assert_allclose(
        contract.batching_rule(
            concatenate_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), None),
            0,
        ),
        expected,
    )
    _assert_allclose(
        contract.batching_rule(
            concatenate_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), None),
            1,
        ),
        np.moveaxis(expected, 0, 1),
    )
    with pytest.raises(ValueError, match="keeps axis static"):
        contract.batching_rule(
            concatenate_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), 0),
            0,
        )
    with pytest.raises(ValueError, match="cannot map the concatenate axis"):
        contract.batching_rule(
            concatenate_fn,
            ((left_batch, right_batch), 0),
            ((0, 0), None),
            0,
        )


def test_program_ad_assembly_stack_contract_and_direct_rule() -> None:
    """np.stack should expose a fail-closed assembly primitive direct rule."""

    left = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    right = np.array([[0.25, -0.75], [1.5, -2.5]], dtype=np.float64)
    tangent_left = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    tangent_right = np.array([[-0.5, 0.25], [0.75, -0.125]], dtype=np.float64)
    cotangent = np.array(
        [[[0.2, -0.4], [0.6, -0.8]], [[1.0, -1.2], [1.4, -1.6]]],
        dtype=np.float64,
    )
    values = np.concatenate([left.reshape(-1), right.reshape(-1)])
    tangent = np.concatenate([tangent_left.reshape(-1), tangent_right.reshape(-1)])

    contract = primitive_contract_for("scpn.program_ad.assembly:stack")
    contract = cast(Any, contract)
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "stack", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.stack"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_stack_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "operand_shapes:ranked_tensor_shapes;axis"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule(((left, right), 1)) == (2, 2, 2)
    assert contract.shape_rule(((left, right), -1)) == (2, 2, 2)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule(((left, right), 1)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule(((left, right), 1)) == (((2, 2), (2, 2)), 1)
    assert contract.static_argument_rule(((left, right), -1)) == (((2, 2), (2, 2)), 2)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_stack_derivative_rule((left.shape, right.shape), axis=1)
    rule = cast(Any, rule)
    assert rule.name == "program_ad_assembly_stack_2_operands_axis1_direct_rule"
    _assert_allclose(
        rule.value_fn(values),
        np.stack([left, right], axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.jvp_rule(values, tangent),
        np.stack([tangent_left, tangent_right], axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate([cotangent[:, 0, :].reshape(-1), cotangent[:, 1, :].reshape(-1)]),
    )


def test_program_ad_assembly_stack_batching_rule_maps_operand_batches() -> None:
    """Stack batching should map operand batches and keep the inserted axis static."""

    contract = primitive_contract_for("scpn.program_ad.assembly:stack")
    contract = cast(Any, contract)
    assert contract.batching_rule is not None

    def stack_fn(operands: tuple[FloatArray, ...], axis: int) -> FloatArray:
        return cast(FloatArray, np.stack(operands, axis=axis))

    left_batch = np.array(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-1.5, 0.25], [2.0, -0.75]],
        ],
        dtype=np.float64,
    )
    right_batch = np.array(
        [
            [[0.25, -0.75], [1.5, -2.5]],
            [[0.5, -1.25], [2.5, -3.0]],
        ],
        dtype=np.float64,
    )
    expected = np.stack(
        [
            np.stack([left_batch[index], right_batch[index]], axis=1)
            for index in range(left_batch.shape[0])
        ],
        axis=0,
    )

    _assert_allclose(
        contract.batching_rule(
            stack_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), None),
            0,
        ),
        expected,
    )
    _assert_allclose(
        contract.batching_rule(
            stack_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), None),
            1,
        ),
        np.moveaxis(expected, 0, 1),
    )
    with pytest.raises(ValueError, match="keeps axis static"):
        contract.batching_rule(
            stack_fn,
            ((left_batch, right_batch), 2),
            ((0, 0), 0),
            0,
        )
    with pytest.raises(ValueError, match="cannot map the stack axis"):
        contract.batching_rule(
            stack_fn,
            ((left_batch, right_batch), 0),
            ((0, 0), None),
            0,
        )


def test_program_ad_assembly_stack_convenience_contracts_and_direct_rules() -> None:
    """Stack convenience calls should expose exact fixed-shape assembly rules."""

    cases = (
        (
            "hstack",
            program_ad_assembly_hstack_derivative_rule,
            (np.array([1.0, -2.0]), np.array([0.5, 3.0, -1.5])),
            np.hstack,
        ),
        (
            "vstack",
            program_ad_assembly_vstack_derivative_rule,
            (np.array([1.0, -2.0]), np.array([0.5, 3.0])),
            np.vstack,
        ),
        (
            "column_stack",
            program_ad_assembly_column_stack_derivative_rule,
            (np.array([1.0, -2.0]), np.array([[0.5, 3.0], [-1.5, 2.0]])),
            np.column_stack,
        ),
        (
            "dstack",
            program_ad_assembly_dstack_derivative_rule,
            (np.array([[1.0, -2.0]]), np.array([[0.5, 3.0]])),
            np.dstack,
        ),
    )
    for name, factory, operands, numpy_function in cases:
        contract = primitive_contract_for(f"scpn.program_ad.assembly:{name}")
        contract = cast(Any, contract)
        assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", name, "1")
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.shape_rule is not None
        assert contract.shape_rule((operands,)) == np.asarray(numpy_function(operands)).shape
        assert contract.dtype_rule is not None
        assert contract.dtype_rule((operands,)) == "float64"
        assert contract.static_argument_rule is not None
        assert contract.static_argument_rule((operands,)) == (
            tuple(operand.shape for operand in operands),
            np.asarray(numpy_function(operands)).shape,
        )
        assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.assembly.{name}"
        assert contract.lowering_metadata["static_derivative_factory"] == factory.__name__
        assert contract.lowering_metadata["rust"] == "blocked_until_polyglot_assembly_ad"

        operand_shapes = tuple(operand.shape for operand in operands)
        values = np.concatenate([operand.reshape(-1) for operand in operands])
        tangent = np.linspace(-0.3, 0.4, values.size, dtype=np.float64)
        expected_value = np.asarray(numpy_function(operands), dtype=np.float64).reshape(-1)
        tangent_operands: list[FloatArray] = []
        offset = 0
        for shape in operand_shapes:
            size = int(np.prod(shape, dtype=np.int64))
            tangent_operands.append(tangent[offset : offset + size].reshape(shape))
            offset += size
        expected_jvp = np.asarray(
            numpy_function(tuple(tangent_operands)), dtype=np.float64
        ).reshape(-1)

        index_operands: list[IntArray] = []
        offset = 0
        for shape in operand_shapes:
            size = int(np.prod(shape, dtype=np.int64))
            index_operands.append(np.arange(offset, offset + size, dtype=np.int64).reshape(shape))
            offset += size
        selected = np.asarray(numpy_function(tuple(index_operands)), dtype=np.int64).reshape(-1)
        cotangent = np.linspace(0.25, 1.0, selected.size, dtype=np.float64)
        expected_vjp = np.zeros(values.size, dtype=np.float64)
        np.add.at(expected_vjp, selected, cotangent)

        rule = factory(operand_shapes)
        rule = cast(Any, rule)
        assert rule.name == f"program_ad_assembly_{name}_{len(operands)}_operands_direct_rule"
        _assert_allclose(rule.value_fn(values), expected_value)
        _assert_allclose(rule.jvp_rule(values, tangent), expected_jvp)
        assert rule.vjp_rule is not None
        _assert_allclose(rule.vjp_rule(values, cotangent), expected_vjp)


def test_program_ad_assembly_stack_convenience_batching_rules_map_operands() -> None:
    """Stack convenience batching should map compatible operand batch axes."""

    for name, function in (
        ("hstack", np.hstack),
        ("vstack", np.vstack),
        ("column_stack", np.column_stack),
        ("dstack", np.dstack),
    ):
        contract = primitive_contract_for(f"scpn.program_ad.assembly:{name}")
        contract = cast(Any, contract)
        assert contract.batching_rule is not None
        left = np.arange(6.0, dtype=np.float64).reshape(3, 2)
        right = np.arange(6.0, 12.0, dtype=np.float64).reshape(3, 2)
        batched = contract.batching_rule(
            function,
            ((left, right),),
            ((0, 0),),
            0,
        )
        expected = np.stack([function((left[index], right[index])) for index in range(3)])
        _assert_allclose(batched, expected)


def test_program_ad_assembly_append_contract_and_direct_rule() -> None:
    """np.append should expose a fail-closed assembly primitive direct rule."""

    source = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    values = np.array([[0.25], [-0.75]], dtype=np.float64)
    tangent_source = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    tangent_values = np.array([[-0.5], [0.25]], dtype=np.float64)
    cotangent = np.array([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=np.float64)
    flat_values = np.concatenate([source.reshape(-1), values.reshape(-1)])
    flat_tangent = np.concatenate([tangent_source.reshape(-1), tangent_values.reshape(-1)])

    contract = primitive_contract_for("scpn.program_ad.assembly:append")
    contract = cast(Any, contract)
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "append", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.append"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_append_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "source_shape:ranked_tensor_shape;values_shape:ranked_tensor_shape;axis"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((source, values, 1)) == (2, 3)
    assert contract.shape_rule((source, values, None)) == (6,)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((source, values, 1)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((source, values, 1)) == ((2, 2), (2, 1), 1)
    assert contract.static_argument_rule((source, values, None)) == ((2, 2), (2, 1), None)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_append_derivative_rule(source.shape, values.shape, axis=1)
    rule = cast(Any, rule)
    assert rule.name == "program_ad_assembly_append_axis1_direct_rule"
    _assert_allclose(
        rule.value_fn(flat_values),
        np.append(source, values, axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.jvp_rule(flat_values, flat_tangent),
        np.append(tangent_source, tangent_values, axis=1).reshape(-1),
    )
    _assert_allclose(
        rule.vjp_rule(flat_values, cotangent.reshape(-1)),
        np.concatenate([cotangent[:, :2].reshape(-1), cotangent[:, 2:].reshape(-1)]),
    )

    flat_rule = program_ad_assembly_append_derivative_rule(source.shape, values.shape, axis=None)
    flat_rule = cast(Any, flat_rule)
    assert flat_rule.name == "program_ad_assembly_append_axisflat_direct_rule"
    _assert_allclose(flat_rule.value_fn(flat_values), np.append(source, values))
    _assert_allclose(flat_rule.jvp_rule(flat_values, flat_tangent), flat_tangent)
    _assert_allclose(flat_rule.vjp_rule(flat_values, flat_values), flat_values)


def test_program_ad_assembly_append_batching_rule_maps_operand_batches() -> None:
    """Append batching should map source and values batches and keep axis static."""

    contract = primitive_contract_for("scpn.program_ad.assembly:append")
    contract = cast(Any, contract)
    assert contract.batching_rule is not None

    def append_fn(array: FloatArray, values: FloatArray, axis: int | None) -> FloatArray:
        return cast(FloatArray, np.append(array, values, axis=axis))

    source_batch = np.array(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-1.5, 0.25], [2.0, -0.75]],
        ],
        dtype=np.float64,
    )
    values_batch = np.array(
        [
            [[0.25], [-0.75]],
            [[1.5], [-2.5]],
        ],
        dtype=np.float64,
    )
    expected = np.stack(
        [
            np.append(source_batch[index], values_batch[index], axis=1)
            for index in range(source_batch.shape[0])
        ],
        axis=0,
    )

    _assert_allclose(
        contract.batching_rule(
            append_fn,
            (source_batch, values_batch, 2),
            (0, 0, None),
            0,
        ),
        expected,
    )
    _assert_allclose(
        contract.batching_rule(
            append_fn,
            (source_batch, values_batch, 2),
            (0, 0, None),
            1,
        ),
        np.moveaxis(expected, 0, 1),
    )
    with pytest.raises(ValueError, match="keeps axis static"):
        contract.batching_rule(append_fn, (source_batch, values_batch, 2), (0, 0, 0), 0)
    with pytest.raises(ValueError, match="cannot map the append axis"):
        contract.batching_rule(append_fn, (source_batch, values_batch, 0), (0, 0, None), 0)


def test_program_ad_assembly_block_contract_and_direct_rule() -> None:
    """np.block should expose a fail-closed nested assembly primitive direct rule."""

    top_left = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    top_right = np.array([[0.25], [-0.75]], dtype=np.float64)
    bottom_left = np.array([[1.5, -2.5]], dtype=np.float64)
    bottom_right = np.array([[0.125]], dtype=np.float64)
    tangent_top_left = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    tangent_top_right = np.array([[-0.5], [0.25]], dtype=np.float64)
    tangent_bottom_left = np.array([[0.75, -0.125]], dtype=np.float64)
    tangent_bottom_right = np.array([[0.625]], dtype=np.float64)
    cotangent = np.array(
        [[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2], [1.4, -1.6, 1.8]],
        dtype=np.float64,
    )
    layout = ((top_left, top_right), (bottom_left, bottom_right))
    layout_shapes = (
        (top_left.shape, top_right.shape),
        (bottom_left.shape, bottom_right.shape),
    )
    values = np.concatenate([array.reshape(-1) for row in layout for array in row])
    tangent = np.concatenate(
        [
            tangent_top_left.reshape(-1),
            tangent_top_right.reshape(-1),
            tangent_bottom_left.reshape(-1),
            tangent_bottom_right.reshape(-1),
        ]
    )

    contract = primitive_contract_for("scpn.program_ad.assembly:block")
    contract = cast(Any, contract)
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "block", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.block"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_block_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "layout_shapes:nested_ranked_tensor_shapes"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((layout,)) == (3, 3)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((layout,)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((layout,)) == (((2, 2), (2, 1)), ((1, 2), (1, 1)))
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_block_derivative_rule(layout_shapes)
    rule = cast(Any, rule)
    assert rule.name == "program_ad_assembly_block_4_operands_direct_rule"
    _assert_allclose(
        rule.value_fn(values),
        np.block([[top_left, top_right], [bottom_left, bottom_right]]).reshape(-1),
    )
    _assert_allclose(
        rule.jvp_rule(values, tangent),
        np.block(
            [
                [tangent_top_left, tangent_top_right],
                [tangent_bottom_left, tangent_bottom_right],
            ]
        ).reshape(-1),
    )
    _assert_allclose(
        rule.vjp_rule(values, cotangent.reshape(-1)),
        np.concatenate(
            [
                cotangent[:2, :2].reshape(-1),
                cotangent[:2, 2:].reshape(-1),
                cotangent[2:, :2].reshape(-1),
                cotangent[2:, 2:].reshape(-1),
            ]
        ),
    )


def test_program_ad_assembly_block_batching_rule_maps_nested_batches() -> None:
    """Block batching should map nested block leaves with matching axes."""

    contract = primitive_contract_for("scpn.program_ad.assembly:block")
    contract = cast(Any, contract)
    assert contract.batching_rule is not None

    def block_fn(layout: tuple[tuple[FloatArray, ...], ...]) -> FloatArray:
        return cast(FloatArray, np.block(layout))

    top_left_batch = np.array(
        [
            [[1.0, -2.0], [0.5, 3.0]],
            [[-1.5, 0.25], [2.0, -0.75]],
        ],
        dtype=np.float64,
    )
    top_right_batch = np.array(
        [
            [[0.25], [-0.75]],
            [[1.5], [-2.5]],
        ],
        dtype=np.float64,
    )
    bottom_left_batch = np.array(
        [
            [[1.5, -2.5]],
            [[0.75, -1.25]],
        ],
        dtype=np.float64,
    )
    bottom_right_batch = np.array([[[0.125]], [[-0.625]]], dtype=np.float64)
    layout = (
        (top_left_batch, top_right_batch),
        (bottom_left_batch, bottom_right_batch),
    )
    axes = (((0, 0), (0, 0)),)
    expected = np.stack(
        [
            np.block(
                [
                    [top_left_batch[index], top_right_batch[index]],
                    [bottom_left_batch[index], bottom_right_batch[index]],
                ]
            )
            for index in range(top_left_batch.shape[0])
        ],
        axis=0,
    )

    _assert_allclose(contract.batching_rule(block_fn, (layout,), axes, 0), expected)
    _assert_allclose(
        contract.batching_rule(block_fn, (layout,), axes, 1),
        np.moveaxis(expected, 0, 1),
    )
    with pytest.raises(ValueError, match="axes matching layout"):
        contract.batching_rule(block_fn, (layout,), (((0,),),), 0)
