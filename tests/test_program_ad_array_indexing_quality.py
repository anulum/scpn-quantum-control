# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD array-indexing quality tests
"""Edge-contract tests for Program AD array-indexing registry helpers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import program_ad_array_indexing as array_indexing
from scpn_quantum_control.program_ad_registry import PrimitiveContract

_batch = array_indexing._program_ad_array_batching_rule
_insert_object = array_indexing._program_ad_array_insert_object
_take_along_static = array_indexing._program_ad_array_take_along_axis_static_arguments
_validate_dispatch = array_indexing._validate_program_ad_array_contract_dispatch


class TraceADArray:
    """Minimal structural trace array used to validate static-shape guards."""

    def __init__(self, shape: object) -> None:
        """Store a deliberately dynamic structural shape."""
        self.shape = shape


def _contract_with(**overrides: object) -> PrimitiveContract:
    """Return the registered getitem contract with selected fields replaced."""
    contract = array_indexing._require_program_ad_array_contract("getitem")
    return replace(contract, **overrides)  # type: ignore[arg-type]


def _registry_for(contract: PrimitiveContract) -> SimpleNamespace:
    """Return a registry stub that resolves one immutable contract."""
    return SimpleNamespace(require_contract=lambda _identity: contract)


def test_program_ad_array_low_level_validation_edges_fail_closed() -> None:
    """Low-level array helpers should reject incompatible static operands."""
    with pytest.raises(ValueError, match="axis-compatible"):
        array_indexing._program_ad_array_take_flat_indices((0,), [0], 0, "clip")
    with pytest.raises(ValueError, match="non-negative integer pad widths"):
        array_indexing._program_ad_array_pad_width([0.5, 1.5], 1, context="test")
    with pytest.raises(ValueError, match="scalar sources require empty"):
        array_indexing._program_ad_array_pad_width(1, 0, context="test")
    assert array_indexing._program_ad_array_pad_width([[1, 2]], 2, context="test") == (
        (1, 2),
        (1, 2),
    )
    with pytest.raises(ValueError, match="finite real constant_values"):
        array_indexing._program_ad_array_pad_constant_values("bad", context="test")
    insert_object = _insert_object(np.asarray(2), context="test")
    assert insert_object == 2
    with pytest.raises(ValueError, match="finite real insert values"):
        array_indexing._program_ad_array_insert_values("bad", context="test")
    with pytest.raises(ValueError, match="static integer or boolean"):
        array_indexing._validate_static_basic_index_selector(True)
    with pytest.raises(ValueError, match="trace shape must be static"):
        array_indexing._program_ad_array_shape_of(TraceADArray([2, 3]))
    assert array_indexing._program_ad_array_shape_of(TraceADArray((2, 3))) == (2, 3)
    assert array_indexing._program_ad_array_dtype_of(TraceADArray((2, 3))) == "float64"
    with pytest.raises(ValueError, match="real numeric arrays"):
        array_indexing._program_ad_array_dtype_of(np.asarray([1.0 + 2.0j]))


def test_program_ad_array_shape_rules_reject_invalid_static_dispatch() -> None:
    """Shape rules should reject malformed arity, selectors, axes, and bounds."""
    matrix = np.arange(6.0).reshape(2, 3)
    empty = np.empty((0,), dtype=np.float64)

    with pytest.raises(ValueError, match="getitem shape rule requires"):
        array_indexing._program_ad_array_getitem_shape((matrix,))
    with pytest.raises(ValueError, match="in-bounds indices"):
        array_indexing._program_ad_array_getitem_shape((matrix, 4))

    with pytest.raises(ValueError, match="take shape rule requires"):
        array_indexing._program_ad_array_take_shape((matrix,))
    with pytest.raises(ValueError, match="static integer indices"):
        array_indexing._program_ad_array_take_shape((matrix, [0.5]))
    with pytest.raises(ValueError, match="indices must be in bounds"):
        array_indexing._program_ad_array_take_shape((matrix, [9]))
    with pytest.raises(ValueError, match="axis-compatible indices"):
        array_indexing._program_ad_array_take_shape((empty, [0], 0, "wrap"))

    with pytest.raises(ValueError, match="take_along_axis shape rule requires"):
        array_indexing._program_ad_array_take_along_axis_shape((matrix, [0]))
    with pytest.raises(ValueError, match="static integer axis"):
        array_indexing._program_ad_array_take_along_axis_shape((matrix, [[0]], True))
    with pytest.raises(ValueError, match="static integer indices"):
        array_indexing._program_ad_array_take_along_axis_shape((matrix, [[0.0]], 1))
    with pytest.raises(ValueError, match="shape-compatible"):
        array_indexing._program_ad_array_take_along_axis_shape((matrix, [[[0]]], 1))

    with pytest.raises(ValueError, match="delete shape rule requires"):
        array_indexing._program_ad_array_delete_shape((matrix,))
    with pytest.raises(ValueError, match="static integer axis"):
        array_indexing._program_ad_array_delete_shape((matrix, 0, True))
    with pytest.raises(ValueError, match="in-bounds deletion selectors"):
        array_indexing._program_ad_array_delete_shape((matrix, 9, 1))

    with pytest.raises(ValueError, match="pad shape rule requires"):
        array_indexing._program_ad_array_pad_shape((matrix,))
    with pytest.raises(ValueError, match="insert shape rule requires"):
        array_indexing._program_ad_array_insert_shape((matrix, 0))
    with pytest.raises(ValueError, match="dtype rule requires"):
        array_indexing._program_ad_array_dtype_rule(())


def test_program_ad_array_static_rules_cover_canonical_edges() -> None:
    """Static rules should canonicalise supported values and reject dynamic ones."""
    matrix = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ValueError, match="getitem static rule requires"):
        array_indexing._program_ad_array_getitem_static_arguments((matrix,))
    with pytest.raises(ValueError, match="static integer or boolean"):
        array_indexing._program_ad_array_static_index_component(object())

    with pytest.raises(ValueError, match="take static rule requires"):
        array_indexing._program_ad_array_take_static_arguments((matrix,))
    with pytest.raises(ValueError, match="static integer indices"):
        array_indexing._program_ad_array_take_static_arguments((matrix, [0.5]))
    with pytest.raises(ValueError, match="static integer axis"):
        array_indexing._program_ad_array_take_static_arguments((matrix, [0], True))

    with pytest.raises(ValueError, match="take_along_axis static rule requires"):
        _take_along_static((matrix, [0]))
    with pytest.raises(ValueError, match="static integer indices"):
        _take_along_static((matrix, [[0.5]], 1))
    with pytest.raises(ValueError, match="static integer axis"):
        _take_along_static((matrix, [[0]], True))

    assert array_indexing._program_ad_array_delete_static_object(np.asarray(1)) == 1
    delete_slice = array_indexing._program_ad_array_delete_static_object(slice(0, 2))
    assert delete_slice == slice(0, 2)
    assert array_indexing._program_ad_array_delete_static_object([True, False]) == (
        "static_delete_mask",
        (2,),
        (True, False),
    )
    with pytest.raises(ValueError, match="delete static rule requires"):
        array_indexing._program_ad_array_delete_static_arguments((matrix,))
    with pytest.raises(ValueError, match="static integer axis"):
        array_indexing._program_ad_array_delete_static_arguments((matrix, 0, True))

    assert array_indexing._program_ad_array_pad_static_constants([1.0, 2.0]) == (
        "static_pad_constants",
        (2,),
        (1.0, 2.0),
    )
    with pytest.raises(ValueError, match="pad static rule requires"):
        array_indexing._program_ad_array_pad_static_arguments((matrix,))

    assert array_indexing._program_ad_array_insert_static_object(np.asarray(1)) == 1
    insert_slice = array_indexing._program_ad_array_insert_static_object(slice(0, 2))
    assert insert_slice == slice(0, 2)
    assert array_indexing._program_ad_array_insert_static_object([1, 2]) == (1, 2)
    assert array_indexing._program_ad_array_insert_static_values(1.5) == 1.5
    with pytest.raises(ValueError, match="insert static rule requires"):
        array_indexing._program_ad_array_insert_static_arguments((matrix, 0))


def test_program_ad_array_batching_rule_maps_only_array_operand() -> None:
    """Batching should map one real array and keep all selectors static."""
    matrix = np.arange(6.0).reshape(2, 3)
    function = lambda row, start: row[start:]  # noqa: E731

    with pytest.raises(ValueError, match="axes must match"):
        _batch(function, (matrix,), (), 0)
    with pytest.raises(ValueError, match="requires an array operand"):
        _batch(function, (), (), 0)
    with pytest.raises(ValueError, match="real numeric"):
        _batch(function, (np.asarray(["bad"]), 0), (0, None), 0)
    with pytest.raises(ValueError, match="array operand to be mapped"):
        _batch(function, (matrix, 1), (None, None), 0)
    with pytest.raises(ValueError, match="static non-array arguments"):
        _batch(function, (matrix, 1), (0, 0), 0)
    with pytest.raises(ValueError, match="out_axes is out of bounds"):
        _batch(function, (matrix, 1), (0, None), 4)

    result = _batch(function, (matrix, 1), (0, None), 1)
    np.testing.assert_array_equal(result, np.asarray([[1.0, 4.0], [2.0, 5.0]]))


def test_program_ad_array_contract_guards_reject_malformed_facets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry dispatch should reject every incomplete or malformed facet."""
    matrix = np.arange(6.0).reshape(2, 3)
    valid = array_indexing._require_program_ad_array_contract("getitem", (matrix, 1))
    assert array_indexing._require_program_ad_array_contract("getitem") == valid
    with pytest.raises(ValueError, match="identity registered"):
        array_indexing._require_program_ad_array_contract("missing")

    dispatch_cases = (
        (_contract_with(static_argument_rule=None), "missing static argument rule"),
        (_contract_with(shape_rule=None), "missing shape rule"),
        (_contract_with(dtype_rule=None), "missing dtype rule"),
        (
            _contract_with(static_argument_rule=cast(Any, lambda _args: "bad")),
            "static rule must return a tuple",
        ),
        (_contract_with(shape_rule=lambda _args: (-1,)), "non-negative integer"),
        (
            _contract_with(shape_rule=cast(Any, lambda _args: ("bad",))),
            "non-negative integer",
        ),
        (_contract_with(dtype_rule=cast(Any, lambda _args: 1)), "return a dtype name"),
        (_contract_with(dtype_rule=lambda _args: ""), "return a dtype name"),
    )
    for contract, message in dispatch_cases:
        with pytest.raises(ValueError, match=message):
            _validate_dispatch(contract, (matrix, 1))

    invalid_contracts = (
        (_contract_with(nondifferentiable_policy="other"), "invalid .* policy"),
        (_contract_with(effect="stateful"), "invalid .* effect"),
        (
            _contract_with(
                batching_rule=None,
                lowering_metadata={},
                shape_rule=None,
                dtype_rule=None,
                static_argument_rule=None,
            ),
            "missing batching_rule, lowering_metadata, mlir_op",
        ),
    )
    for contract, message in invalid_contracts:
        registry = _registry_for(contract)
        registry_name = "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY"
        monkeypatch.setattr(array_indexing, registry_name, registry)
        with pytest.raises(ValueError, match=message):
            array_indexing._require_program_ad_array_contract("getitem")


def test_program_ad_array_registration_preserves_existing_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated registration should leave existing primitive contracts untouched."""
    registry = SimpleNamespace(
        contract_for=lambda _identity: object(),
        register_transform=lambda _transform: pytest.fail("unexpected registration"),
    )
    monkeypatch.setattr(array_indexing, "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY", registry)
    array_indexing._register_program_ad_array_primitive_contracts()
