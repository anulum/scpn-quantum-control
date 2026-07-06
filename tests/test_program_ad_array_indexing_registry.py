# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD array indexing registry tests
"""Tests for Program AD array-indexing registry contracts and slicing adjoints."""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import differentiable as differentiable_module
from scpn_quantum_control import program_ad_array_indexing as array_indexing
from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_delete_derivative_rule as direct_delete_derivative_rule,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_getitem_derivative_rule as direct_getitem_derivative_rule,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_insert_derivative_rule as direct_insert_derivative_rule,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_pad_derivative_rule as direct_pad_derivative_rule,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_take_along_axis_derivative_rule as direct_take_along_axis_derivative_rule,
)
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_take_derivative_rule as direct_take_derivative_rule,
)

DOCSTRING_SECTION_TARGETS: tuple[tuple[str, object, tuple[str, ...]], ...] = (
    (
        "program_ad_array_getitem_derivative_rule",
        array_indexing.program_ad_array_getitem_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_array_take_derivative_rule",
        array_indexing.program_ad_array_take_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_array_take_along_axis_derivative_rule",
        array_indexing.program_ad_array_take_along_axis_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_array_delete_derivative_rule",
        array_indexing.program_ad_array_delete_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_array_pad_derivative_rule",
        array_indexing.program_ad_array_pad_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "program_ad_array_insert_derivative_rule",
        array_indexing.program_ad_array_insert_derivative_rule,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "_program_ad_array_derivative_rule",
        array_indexing._program_ad_array_derivative_rule,
        ("Parameters", "Returns"),
    ),
    (
        "_program_ad_array_normalise_static_shape",
        array_indexing._program_ad_array_normalise_static_shape,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "_program_ad_array_shape_of",
        array_indexing._program_ad_array_shape_of,
        ("Parameters", "Returns", "Raises"),
    ),
    (
        "_program_ad_array_signature",
        array_indexing._program_ad_array_signature,
        ("Parameters", "Returns"),
    ),
    (
        "_program_ad_array_static_size",
        array_indexing._program_ad_array_static_size,
        ("Parameters", "Returns"),
    ),
    (
        "_register_program_ad_array_primitive_contracts",
        array_indexing._register_program_ad_array_primitive_contracts,
        ("Returns",),
    ),
    (
        "_require_program_ad_array_contract",
        array_indexing._require_program_ad_array_contract,
        ("Parameters", "Returns", "Raises"),
    ),
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_array_indexing_exported_docstrings_define_contract_sections() -> None:
    """Exported array-indexing helpers should document their public contracts."""
    for qualified_name, target, required_sections in DOCSTRING_SECTION_TARGETS:
        docstring = inspect.getdoc(target)
        assert docstring is not None, qualified_name
        for section in required_sections:
            assert f"{section}\n" in docstring, qualified_name


def test_program_ad_array_indexing_direct_module_exports_match_facade() -> None:
    """Static array-indexing factories should remain stable through the facade."""
    facade_exports = vars(differentiable_module)
    assert (
        facade_exports["_register_program_ad_array_primitive_contracts"]
        is array_indexing._register_program_ad_array_primitive_contracts
    )
    assert (
        facade_exports["_require_program_ad_array_contract"]
        is array_indexing._require_program_ad_array_contract
    )
    assert (
        facade_exports["_program_ad_array_shape_of"] is array_indexing._program_ad_array_shape_of
    )
    assert program_ad_array_getitem_derivative_rule is direct_getitem_derivative_rule
    assert program_ad_array_take_derivative_rule is direct_take_derivative_rule
    assert (
        program_ad_array_take_along_axis_derivative_rule is direct_take_along_axis_derivative_rule
    )
    assert (
        differentiable_module.program_ad_array_delete_derivative_rule
        is direct_delete_derivative_rule
    )
    assert differentiable_module.program_ad_array_pad_derivative_rule is direct_pad_derivative_rule
    assert (
        differentiable_module.program_ad_array_insert_derivative_rule
        is direct_insert_derivative_rule
    )


def test_program_ad_array_indexing_direct_module_fail_closed_boundaries() -> None:
    """Static direct-rule helpers should reject malformed array-indexing contracts."""
    contract_rule = array_indexing._program_ad_array_derivative_rule("getitem")
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        contract_rule.value_fn(np.array([1.0], dtype=np.float64))
    assert contract_rule.jvp_rule is not None
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        contract_rule.jvp_rule(
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
        )

    assert array_indexing._program_ad_array_normalise_static_shape("getitem", (2, 0)) == (2, 0)
    with pytest.raises(ValueError, match="non-negative dimensions"):
        array_indexing._program_ad_array_normalise_static_shape("getitem", (2, -1))
    assert array_indexing._program_ad_array_signature(()) == "scalar"
    with pytest.raises(ValueError, match="with 3 values"):
        array_indexing._program_ad_array_vector(
            "take",
            "values",
            np.array([1.0, 2.0], dtype=np.float64),
            expected_size=3,
        )

    with pytest.raises(ValueError, match="in-bounds indices"):
        program_ad_array_getitem_derivative_rule((2, 3), (slice(None), 4))
    with pytest.raises(ValueError, match="static integer slice bounds"):
        program_ad_array_getitem_derivative_rule((2, 3), slice(None, 1.5))
    with pytest.raises(ValueError, match="static integer or boolean"):
        program_ad_array_getitem_derivative_rule((2, 3), object())
    with pytest.raises(ValueError, match="static integer or boolean"):
        program_ad_array_getitem_derivative_rule((2, 3), np.array(True))

    with pytest.raises(ValueError, match="in-bounds indices"):
        program_ad_array_take_derivative_rule((3,), (3,), mode="raise")
    with pytest.raises(ValueError, match="out of bounds"):
        program_ad_array_take_derivative_rule((2, 3), (0,), axis=5, mode="wrap")
    with pytest.raises(ValueError, match="shape compatible"):
        program_ad_array_take_along_axis_derivative_rule(
            (2, 3),
            np.array([[0, 1]], dtype=np.int64),
            axis=0,
        )
    with pytest.raises(ValueError, match="static integer axis"):
        program_ad_array_take_along_axis_derivative_rule(
            (2, 3),
            np.array([[0, 1, 2]], dtype=np.int64),
            axis=cast(Any, True),
        )

    assert array_indexing._program_ad_array_delete_object(1, context="test") == 1
    assert array_indexing._program_ad_array_delete_object(np.array(1), context="test") == 1
    assert isinstance(
        array_indexing._program_ad_array_delete_object(np.array([True, False]), context="test"),
        np.ndarray,
    )
    assert isinstance(
        array_indexing._program_ad_array_delete_object(slice(0, 2), context="test"),
        slice,
    )
    with pytest.raises(ValueError, match="deletion selectors"):
        array_indexing._program_ad_array_delete_object(True, context="test")
    with pytest.raises(ValueError, match="slice bounds"):
        array_indexing._program_ad_array_delete_object(slice(0.0, 2), context="test")
    with pytest.raises(ValueError, match="deletion selectors"):
        array_indexing._program_ad_array_delete_object(np.array(["x"]), context="test")
    with pytest.raises(ValueError, match="in-bounds deletion selectors"):
        differentiable_module.program_ad_array_delete_derivative_rule((3,), (5,))

    assert array_indexing._program_ad_array_pad_mode("constant", context="test") == "constant"
    with pytest.raises(ValueError, match="constant mode only"):
        array_indexing._program_ad_array_pad_mode("edge", context="test")
    assert array_indexing._program_ad_array_pad_width(1, 2, context="test") == ((1, 1), (1, 1))
    assert array_indexing._program_ad_array_pad_width((1, 2), 2, context="test") == (
        (1, 2),
        (1, 2),
    )
    assert array_indexing._program_ad_array_pad_width(((1, 0), (0, 1)), 2, context="test") == (
        (1, 0),
        (0, 1),
    )
    assert (
        array_indexing._program_ad_array_pad_width(np.array([], dtype=np.int64), 0, context="test")
        == ()
    )
    with pytest.raises(ValueError, match="non-negative integer pad widths"):
        array_indexing._program_ad_array_pad_width(-1, 1, context="test")
    with pytest.raises(ValueError, match="scalar, pair, or per-axis"):
        array_indexing._program_ad_array_pad_width((1, 2, 3), 2, context="test")
    with pytest.raises(ValueError, match="static finite real constant_values"):
        array_indexing._program_ad_array_pad_constant_values(np.inf, context="test")
    with pytest.raises(ValueError, match="compatible with the source rank"):
        differentiable_module.program_ad_array_pad_derivative_rule(
            (2, 3),
            ((1, 0), (0, 1)),
            constant_values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    assert array_indexing._program_ad_array_insert_object(1, context="test") == 1
    assert isinstance(
        array_indexing._program_ad_array_insert_object(slice(0, 2), context="test"), slice
    )
    assert array_indexing._program_ad_array_insert_axis(None, 2, context="test") is None
    assert array_indexing._program_ad_array_insert_axis(-1, 2, context="test") == 1
    with pytest.raises(ValueError, match="insertion indices"):
        array_indexing._program_ad_array_insert_object(True, context="test")
    with pytest.raises(ValueError, match="insertion indices"):
        array_indexing._program_ad_array_insert_object(slice(0.0, 2), context="test")
    with pytest.raises(ValueError, match="insertion indices"):
        array_indexing._program_ad_array_insert_object(np.array(["x"]), context="test")
    with pytest.raises(ValueError, match="static finite real insert values"):
        array_indexing._program_ad_array_insert_values(np.nan, context="test")
    with pytest.raises(ValueError, match="static integer axis or None"):
        array_indexing._program_ad_array_insert_axis(True, 2, context="test")
    with pytest.raises(ValueError, match="compatible with the source shape"):
        differentiable_module.program_ad_array_insert_derivative_rule(
            (2, 3),
            1,
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            axis=1,
        )

    assert array_indexing._normalise_axis("axis", -1, 2) == 1
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        array_indexing._normalise_axis("axis", 0, 0)
    with pytest.raises(ValueError, match="out of bounds"):
        array_indexing._normalise_axis("axis", 2, 2)


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


def test_program_ad_array_static_derivative_factories_are_direct_kernels() -> None:
    """Static array-indexing factories should expose exact gather/scatter adjoints."""
    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    values = matrix.reshape(-1)
    tangent = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 1.25], dtype=np.float64)
    flat_values = np.arange(6.0, dtype=np.float64)

    getitem_rule = program_ad_array_getitem_derivative_rule((2, 3), (slice(None), 1))
    getitem_jvp = getitem_rule.jvp_rule
    getitem_vjp = getitem_rule.vjp_rule
    assert getitem_rule.name == "program_ad_array_getitem_2x3_static_direct_rule"
    assert getitem_jvp is not None
    assert getitem_vjp is not None
    _assert_allclose(
        getitem_rule.value_fn(values),
        matrix[:, 1].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        getitem_jvp(values, tangent),
        tangent.reshape(2, 3)[:, 1].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        getitem_vjp(values, np.array([1.5, -2.0], dtype=np.float64)),
        np.array([0.0, 1.5, 0.0, 0.0, -2.0, 0.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    advanced_getitem_rule = program_ad_array_getitem_derivative_rule(
        (2, 3), ([1, 0, 1], [2, 0, 2])
    )
    advanced_getitem_jvp = advanced_getitem_rule.jvp_rule
    advanced_getitem_vjp = advanced_getitem_rule.vjp_rule
    assert advanced_getitem_jvp is not None
    assert advanced_getitem_vjp is not None
    _assert_allclose(
        advanced_getitem_rule.value_fn(values),
        matrix[[1, 0, 1], [2, 0, 2]].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        advanced_getitem_jvp(values, tangent),
        tangent.reshape(2, 3)[[1, 0, 1], [2, 0, 2]].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        advanced_getitem_vjp(values, np.array([1.0, -0.5, 2.0], dtype=np.float64)),
        np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    take_rule = program_ad_array_take_derivative_rule((2, 3), (2, 0, 2), axis=1)
    take_jvp = take_rule.jvp_rule
    take_vjp = take_rule.vjp_rule
    assert take_rule.name == "program_ad_array_take_2x3_axis_1_static_direct_rule"
    assert take_jvp is not None
    assert take_vjp is not None
    expected_take = np.take(matrix, [2, 0, 2], axis=1)
    expected_take_tangent = np.take(tangent.reshape(2, 3), [2, 0, 2], axis=1)
    _assert_allclose(take_rule.value_fn(values), expected_take.reshape(-1))
    _assert_allclose(take_jvp(values, tangent), expected_take_tangent.reshape(-1))
    _assert_allclose(
        take_vjp(
            values,
            np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.25]], dtype=np.float64).reshape(-1),
        ),
        np.array([-0.5, 0.0, 3.0, 1.5, 0.0, -1.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="static integer or boolean"):
        program_ad_array_getitem_derivative_rule((2, 3), np.array([1.0, 0.0]))

    wrap_rule = program_ad_array_take_derivative_rule((6,), (-1, 6, 0), mode="wrap")
    wrap_jvp = wrap_rule.jvp_rule
    wrap_vjp = wrap_rule.vjp_rule
    assert wrap_rule.name == "program_ad_array_take_6_axis_flat_mode_wrap_static_direct_rule"
    assert wrap_jvp is not None
    assert wrap_vjp is not None
    _assert_allclose(wrap_rule.value_fn(flat_values), [5.0, 0.0, 0.0])
    _assert_allclose(
        wrap_jvp(
            flat_values,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        ),
        [5.0, 0.0, 0.0],
    )
    _assert_allclose(
        wrap_vjp(flat_values, np.array([0.5, -1.0, 2.0], dtype=np.float64)),
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    )

    clip_rule = program_ad_array_take_derivative_rule((6,), (-3, 2, 20), mode="clip")
    clip_vjp = clip_rule.vjp_rule
    assert clip_rule.name == "program_ad_array_take_6_axis_flat_mode_clip_static_direct_rule"
    assert clip_rule.jvp_rule is not None
    assert clip_vjp is not None
    _assert_allclose(clip_rule.value_fn(flat_values), [0.0, 2.0, 5.0])
    _assert_allclose(
        clip_vjp(flat_values, np.array([-0.25, 1.5, 0.75], dtype=np.float64)),
        [-0.25, 0.0, 1.5, 0.0, 0.0, 0.75],
    )
    with pytest.raises(ValueError, match="mode"):
        program_ad_array_take_derivative_rule((2, 3), (0,), axis=0, mode="not_a_mode")

    delete_rule = differentiable_module.program_ad_array_delete_derivative_rule(
        (2, 3), (1,), axis=1
    )
    delete_jvp = delete_rule.jvp_rule
    delete_vjp = delete_rule.vjp_rule
    assert delete_rule.name == "program_ad_array_delete_2x3_axis_1_static_direct_rule"
    assert delete_jvp is not None
    assert delete_vjp is not None
    _assert_allclose(delete_rule.value_fn(values), [0.0, 2.0, 3.0, 5.0])
    _assert_allclose(delete_jvp(values, tangent), [0.5, 0.25, 2.0, 1.25])
    _assert_allclose(
        delete_vjp(values, np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)),
        [1.0, 0.0, -2.0, 0.5, 0.0, 3.0],
    )

    flat_delete_rule = differentiable_module.program_ad_array_delete_derivative_rule((6,), (1, 4))
    flat_delete_vjp = flat_delete_rule.vjp_rule
    assert flat_delete_vjp is not None
    assert flat_delete_rule.name == "program_ad_array_delete_6_axis_flat_static_direct_rule"
    _assert_allclose(flat_delete_rule.value_fn(flat_values), [0.0, 2.0, 3.0, 5.0])
    _assert_allclose(
        flat_delete_vjp(flat_values, np.array([0.25, -0.75, 1.25, -1.5], dtype=np.float64)),
        [0.25, 0.0, -0.75, 1.25, 0.0, -1.5],
    )

    pad_rule = differentiable_module.program_ad_array_pad_derivative_rule(
        (2, 3), ((1, 0), (0, 1)), constant_values=-2.0
    )
    pad_jvp = pad_rule.jvp_rule
    pad_vjp = pad_rule.vjp_rule
    assert pad_rule.name == "program_ad_array_pad_2x3_static_constant_direct_rule"
    assert pad_jvp is not None
    assert pad_vjp is not None
    expected_pad = np.pad(matrix, ((1, 0), (0, 1)), constant_values=-2.0)
    expected_pad_tangent = np.pad(tangent.reshape(2, 3), ((1, 0), (0, 1)), constant_values=0.0)
    _assert_allclose(pad_rule.value_fn(values), expected_pad.reshape(-1))
    _assert_allclose(pad_jvp(values, tangent), expected_pad_tangent.reshape(-1))
    _assert_allclose(
        pad_vjp(
            values,
            np.array(
                [[0.5, -1.0, 2.0, 0.25], [1.5, -2.0, 0.75, 3.0], [-0.25, 0.5, 2.5, -1.5]],
                dtype=np.float64,
            ).reshape(-1),
        ),
        [1.5, -2.0, 0.75, -0.25, 0.5, 2.5],
    )

    insert_rule = differentiable_module.program_ad_array_insert_derivative_rule(
        (2, 3), 1, np.array([-2.0, 3.0]), axis=1
    )
    insert_jvp = insert_rule.jvp_rule
    insert_vjp = insert_rule.vjp_rule
    assert insert_rule.name == "program_ad_array_insert_2x3_axis_1_static_constant_direct_rule"
    assert insert_jvp is not None
    assert insert_vjp is not None
    expected_insert = np.insert(matrix, 1, np.array([-2.0, 3.0]), axis=1)
    expected_insert_tangent = np.insert(tangent.reshape(2, 3), 1, np.array([0.0, 0.0]), axis=1)
    _assert_allclose(insert_rule.value_fn(values), expected_insert.reshape(-1))
    _assert_allclose(insert_jvp(values, tangent), expected_insert_tangent.reshape(-1))
    _assert_allclose(
        insert_vjp(
            values,
            np.array([[1.0, 10.0, -2.0, 0.5], [3.0, -20.0, -1.5, 2.5]], dtype=np.float64).reshape(
                -1
            ),
        ),
        [1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
    )

    flat_insert_rule = differentiable_module.program_ad_array_insert_derivative_rule(
        (6,), (1, 4), np.array([0.5, -1.5])
    )
    flat_insert_vjp = flat_insert_rule.vjp_rule
    assert flat_insert_vjp is not None
    assert (
        flat_insert_rule.name == "program_ad_array_insert_6_axis_flat_static_constant_direct_rule"
    )
    _assert_allclose(
        flat_insert_rule.value_fn(flat_values),
        np.insert(flat_values, [1, 4], np.array([0.5, -1.5])),
    )
    _assert_allclose(
        flat_insert_vjp(
            flat_values,
            np.array([1.0, 10.0, -2.0, 0.5, 3.0, -20.0, -1.5, 2.5], dtype=np.float64),
        ),
        [1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
    )

    along_indices = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int64)
    along_weights = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.25]], dtype=np.float64)
    along_rule = program_ad_array_take_along_axis_derivative_rule((2, 3), along_indices, axis=1)
    along_jvp = along_rule.jvp_rule
    along_vjp = along_rule.vjp_rule
    assert along_rule.name == "program_ad_array_take_along_axis_2x3_axis_1_static_direct_rule"
    assert along_jvp is not None
    assert along_vjp is not None
    expected_along = np.take_along_axis(matrix, along_indices, axis=1)
    expected_along_tangent = np.take_along_axis(tangent.reshape(2, 3), along_indices, axis=1)
    _assert_allclose(along_rule.value_fn(values), expected_along.reshape(-1))
    _assert_allclose(along_jvp(values, tangent), expected_along_tangent.reshape(-1))
    _assert_allclose(
        along_vjp(values, along_weights.reshape(-1)),
        np.array([-0.5, 0.0, 3.0, -1.25, 1.75, 0.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="static integer indices"):
        program_ad_array_take_along_axis_derivative_rule((2, 3), np.array([[0.0, 1.0]]), axis=1)


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
