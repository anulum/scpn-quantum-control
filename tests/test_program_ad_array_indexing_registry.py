# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD array indexing registry tests
"""Tests for Program AD array-indexing registry contracts and slicing adjoints."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(actual: object, expected: object, *, atol: float = 0.0) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, atol=atol)


def test_program_ad_basic_slicing_preserves_static_adjoint_paths() -> None:
    """Program AD basic slicing should preserve exact static index adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        block = matrix[..., 1:]
        expanded = matrix[:, :1][:, None, :]
        return np.sum(block * np.array([[1.0, 2.0], [3.0, 4.0]])) + 2.0 * expanded[1, 0, 0]

    result = whole_program_value_and_grad(
        objective,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        parameters=(
            Parameter("x0"),
            Parameter("x1"),
            Parameter("x2"),
            Parameter("x3"),
            Parameter("x4"),
            Parameter("x5"),
        ),
    )

    assert result.value == pytest.approx(55.0)
    _assert_allclose(result.gradient, [0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)


def test_program_ad_array_indexing_primitives_are_registry_policy_gated() -> None:
    """Static indexing and take should expose primitive registry contracts."""

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    contract = primitive_contract_for("scpn.program_ad.array:getitem")
    shape_rule = cast(Any, contract.shape_rule)
    dtype_rule = cast(Any, contract.dtype_rule)
    static_argument_rule = cast(Any, contract.static_argument_rule)

    assert contract.identity == PrimitiveIdentity("scpn.program_ad.array", "getitem", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["program_ad"] == "operator_intercepted_trace"
    assert (
        contract.lowering_metadata["mlir"]
        == "available: scpn_diff array dialect interchange; executable lowering blocked"
    )
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.array.getitem"
    assert contract.shape_rule is not None
    assert shape_rule((matrix, (slice(None), slice(1, None)))) == (2, 2)
    assert shape_rule((matrix, ([1, 0, 1], [2, 0, 2]))) == (3,)
    assert shape_rule((matrix, np.array([True, False]))) == (1, 3)
    assert contract.dtype_rule is not None
    assert dtype_rule((matrix, (slice(None), slice(1, None)))) == "float64"
    assert contract.static_argument_rule is not None
    assert static_argument_rule((matrix, (slice(None), slice(1, None)))) == (
        (slice(None), slice(1, None)),
    )
    assert static_argument_rule((matrix, ([1, 0, 1], [2, 0, 2]))) == (
        (
            ("static_index_array", "int64", (3,), (1, 0, 1)),
            ("static_index_array", "int64", (3,), (2, 0, 2)),
        ),
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    take_contract = primitive_contract_for("scpn.program_ad.array:take")
    take_shape_rule = cast(Any, take_contract.shape_rule)
    take_dtype_rule = cast(Any, take_contract.dtype_rule)
    take_static_argument_rule = cast(Any, take_contract.static_argument_rule)

    assert take_contract.identity == PrimitiveIdentity("scpn.program_ad.array", "take", "1")
    assert take_contract.lowering_metadata["mlir_op"] == "scpn_diff.array.take"
    assert (
        take_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_array_take_derivative_rule"
    )
    assert take_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;indices_axis_mode"
    )
    assert take_contract.shape_rule is not None
    assert take_shape_rule((matrix, np.array([1, 0]), 1)) == (2, 2)
    assert take_contract.dtype_rule is not None
    assert take_dtype_rule((matrix, np.array([1, 0]), 1)) == "float64"
    assert take_contract.static_argument_rule is not None
    assert take_static_argument_rule((matrix, np.array([1, 0]), 1)) == (
        (1, 0),
        1,
        "raise",
    )
    assert take_shape_rule((matrix, np.array([-1, 5]), None, "wrap")) == (2,)
    assert take_static_argument_rule((matrix, np.array([-1, 5]), None, "wrap")) == (
        (-1, 5),
        None,
        "wrap",
    )
    assert take_shape_rule((matrix, np.array([-2, 1, 8]), 1, "clip")) == (2, 3)
    assert take_static_argument_rule((matrix, np.array([-2, 1, 8]), 1, "clip")) == (
        (-2, 1, 8),
        1,
        "clip",
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(take_contract.identity)

    along_contract = primitive_contract_for("scpn.program_ad.array:take_along_axis")
    along_shape_rule = cast(Any, along_contract.shape_rule)
    along_dtype_rule = cast(Any, along_contract.dtype_rule)
    along_static_argument_rule = cast(Any, along_contract.static_argument_rule)

    assert along_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.array", "take_along_axis", "1"
    )
    assert along_contract.lowering_metadata["mlir_op"] == "scpn_diff.array.take_along_axis"
    assert (
        along_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_array_take_along_axis_derivative_rule"
    )
    assert along_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;indices_shape_axis"
    )
    assert along_contract.shape_rule is not None
    assert along_shape_rule((matrix, np.array([[2, 0, 2], [1, 1, 0]]), 1)) == (2, 3)
    assert along_contract.dtype_rule is not None
    assert along_dtype_rule((matrix, np.array([[2, 0, 2], [1, 1, 0]]), 1)) == "float64"
    assert along_contract.static_argument_rule is not None
    assert along_static_argument_rule((matrix, np.array([[2, 0, 2], [1, 1, 0]]), 1)) == (
        (2, 0, 2, 1, 1, 0),
        (2, 3),
        1,
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(along_contract.identity)

    delete_contract = primitive_contract_for("scpn.program_ad.array:delete")
    delete_shape_rule = cast(Any, delete_contract.shape_rule)
    delete_dtype_rule = cast(Any, delete_contract.dtype_rule)
    delete_static_argument_rule = cast(Any, delete_contract.static_argument_rule)

    assert delete_contract.identity == PrimitiveIdentity("scpn.program_ad.array", "delete", "1")
    assert delete_contract.lowering_metadata["mlir_op"] == "scpn_diff.array.delete"
    assert (
        delete_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_array_delete_derivative_rule"
    )
    assert delete_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;object_axis"
    )
    assert delete_contract.shape_rule is not None
    assert delete_shape_rule((matrix, np.array([1]), 1)) == (2, 2)
    assert delete_shape_rule((matrix, np.array([1, 4]), None)) == (4,)
    assert delete_contract.static_argument_rule is not None
    assert delete_static_argument_rule((matrix, np.array([1]), 1)) == ((1,), 1)
    assert delete_static_argument_rule((matrix, np.array([1, 4]), None)) == (
        (1, 4),
        None,
    )
    assert delete_contract.dtype_rule is not None
    assert delete_dtype_rule((matrix, np.array([1]), 1)) == "float64"
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(delete_contract.identity)

    pad_contract = primitive_contract_for("scpn.program_ad.array:pad")
    pad_shape_rule = cast(Any, pad_contract.shape_rule)
    pad_dtype_rule = cast(Any, pad_contract.dtype_rule)
    pad_static_argument_rule = cast(Any, pad_contract.static_argument_rule)

    assert pad_contract.identity == PrimitiveIdentity("scpn.program_ad.array", "pad", "1")
    assert pad_contract.lowering_metadata["mlir_op"] == "scpn_diff.array.pad"
    assert (
        pad_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_array_pad_derivative_rule"
    )
    assert pad_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;pad_width_constant_values"
    )
    assert pad_contract.shape_rule is not None
    assert pad_shape_rule((matrix, ((1, 0), (0, 2)), "constant", -1.0)) == (3, 5)
    assert pad_contract.static_argument_rule is not None
    assert pad_static_argument_rule((matrix, ((1, 0), (0, 2)), "constant", -1.0)) == (
        ((1, 0), (0, 2)),
        "constant",
        -1.0,
    )
    assert pad_contract.dtype_rule is not None
    assert pad_dtype_rule((matrix, ((1, 0), (0, 2)), "constant", -1.0)) == "float64"
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(pad_contract.identity)

    insert_contract = primitive_contract_for("scpn.program_ad.array:insert")
    insert_shape_rule = cast(Any, insert_contract.shape_rule)
    insert_dtype_rule = cast(Any, insert_contract.dtype_rule)
    insert_static_argument_rule = cast(Any, insert_contract.static_argument_rule)

    assert insert_contract.identity == PrimitiveIdentity("scpn.program_ad.array", "insert", "1")
    assert insert_contract.lowering_metadata["mlir_op"] == "scpn_diff.array.insert"
    assert (
        insert_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_array_insert_derivative_rule"
    )
    assert insert_contract.lowering_metadata["static_signature"] == (
        "source_shape:ranked_tensor_shape;object_values_axis"
    )
    assert insert_contract.shape_rule is not None
    assert insert_shape_rule((matrix, 1, np.array([-2.0, 3.0]), 1)) == (2, 4)
    assert insert_shape_rule((matrix, np.array([1, 4]), np.array([0.5, -1.5]), None)) == (8,)
    assert insert_contract.static_argument_rule is not None
    assert insert_static_argument_rule((matrix, 1, np.array([-2.0, 3.0]), 1)) == (
        1,
        ("static_insert_values", (2,), (-2.0, 3.0)),
        1,
    )
    assert insert_contract.dtype_rule is not None
    assert insert_dtype_rule((matrix, 1, np.array([-2.0, 3.0]), 1)) == "float64"
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(insert_contract.identity)


def test_program_ad_array_boundary_metadata_is_explicit() -> None:
    """Array-indexing contracts should expose fail-closed static gather boundaries."""

    expected_factories = {
        "getitem": "program_ad_array_getitem_derivative_rule",
        "take": "program_ad_array_take_derivative_rule",
        "take_along_axis": "program_ad_array_take_along_axis_derivative_rule",
        "delete": "program_ad_array_delete_derivative_rule",
        "pad": "program_ad_array_pad_derivative_rule",
        "insert": "program_ad_array_insert_derivative_rule",
    }
    expected_static_signatures = {
        "getitem": "source_shape:ranked_tensor_shape;index:static_gather_index",
        "take": "source_shape:ranked_tensor_shape;indices_axis_mode",
        "take_along_axis": "source_shape:ranked_tensor_shape;indices_shape_axis",
        "delete": "source_shape:ranked_tensor_shape;object_axis",
        "pad": "source_shape:ranked_tensor_shape;pad_width_constant_values",
        "insert": "source_shape:ranked_tensor_shape;object_values_axis",
    }
    expected_boundaries = {
        "getitem": "static_gather_index_scatter_add",
        "take": "static_integer_gather_scatter_add",
        "take_along_axis": "static_along_axis_gather_scatter_add",
        "delete": "static_delete_gather_scatter_add",
        "pad": "static_constant_pad_scatter_add",
        "insert": "static_constant_insert_scatter_add",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.array", name, "1")
        ).lowering_metadata
        assert metadata["static_derivative_factory"] == expected_factories[name]
        assert metadata["static_signature"] == expected_static_signatures[name]
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"
