# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD shape transforms tests
# scpn-quantum-control -- Program AD shape transform tests
"""Tests for Program AD shape-transform adjoints and static contracts."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    Parameter,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_shape_transforms import (
    _program_ad_shape_derivative_rule,
    _register_program_ad_shape_primitive_contracts,
    _require_program_ad_shape_contract,
    program_ad_shape_atleast_1d_derivative_rule,
    program_ad_shape_atleast_2d_derivative_rule,
    program_ad_shape_atleast_3d_derivative_rule,
    program_ad_shape_expand_dims_derivative_rule,
    program_ad_shape_flip_derivative_rule,
    program_ad_shape_fliplr_derivative_rule,
    program_ad_shape_flipud_derivative_rule,
    program_ad_shape_moveaxis_derivative_rule,
    program_ad_shape_ravel_derivative_rule,
    program_ad_shape_repeat_derivative_rule,
    program_ad_shape_reshape_derivative_rule,
    program_ad_shape_roll_derivative_rule,
    program_ad_shape_rot90_derivative_rule,
    program_ad_shape_squeeze_derivative_rule,
    program_ad_shape_swapaxes_derivative_rule,
    program_ad_shape_tile_derivative_rule,
    program_ad_shape_transpose_derivative_rule,
)

_DOCSTRING_SECTION_TARGETS: tuple[
    tuple[Callable[..., object], tuple[str, ...]],
    ...,
] = (
    (_program_ad_shape_derivative_rule, ("Parameters", "Returns")),
    (_register_program_ad_shape_primitive_contracts, ("Returns",)),
    (_require_program_ad_shape_contract, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_atleast_1d_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_atleast_2d_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_atleast_3d_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_expand_dims_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_flip_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_fliplr_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_flipud_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_moveaxis_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_ravel_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_repeat_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_reshape_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_roll_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_rot90_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_squeeze_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_swapaxes_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_tile_derivative_rule, ("Parameters", "Returns", "Raises")),
    (program_ad_shape_transpose_derivative_rule, ("Parameters", "Returns", "Raises")),
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _contract_rules(contract: Any) -> tuple[Any, Any, Any]:
    """Return asserted registry validation callables for dynamic contracts."""
    assert contract.shape_rule is not None
    assert contract.dtype_rule is not None
    assert contract.static_argument_rule is not None
    return contract.shape_rule, contract.dtype_rule, contract.static_argument_rule


def _transform_rule_from_contract(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a mutable registry transform that exactly mirrors a contract."""
    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def test_program_ad_shape_public_docstrings_define_contract_sections() -> None:
    """Exported shape-transform helpers should document their runtime contracts."""
    for function, section_names in _DOCSTRING_SECTION_TARGETS:
        docstring = inspect.getdoc(function)
        assert docstring is not None, function.__name__
        for section_name in section_names:
            assert section_name in docstring, function.__name__


def test_program_ad_shape_direct_factories_have_dedicated_module_path() -> None:
    """Shape direct-rule factories should be available outside the facade."""
    from scpn_quantum_control import differentiable as differentiable_facade
    from scpn_quantum_control import program_ad_shape_transforms

    assert (
        program_ad_shape_transforms.program_ad_shape_reshape_derivative_rule
        is program_ad_shape_reshape_derivative_rule
    )
    assert (
        differentiable_facade.program_ad_shape_reshape_derivative_rule
        is program_ad_shape_reshape_derivative_rule
    )
    assert (
        program_ad_shape_transforms.program_ad_shape_repeat_derivative_rule
        is program_ad_shape_repeat_derivative_rule
    )
    assert (
        differentiable_facade.program_ad_shape_repeat_derivative_rule
        is program_ad_shape_repeat_derivative_rule
    )
    assert (
        program_ad_shape_transforms.program_ad_shape_atleast_3d_derivative_rule
        is program_ad_shape_atleast_3d_derivative_rule
    )
    assert (
        differentiable_facade.program_ad_shape_atleast_3d_derivative_rule
        is program_ad_shape_atleast_3d_derivative_rule
    )
    facade_exports = vars(differentiable_facade)
    assert (
        facade_exports["_register_program_ad_shape_primitive_contracts"]
        is program_ad_shape_transforms._register_program_ad_shape_primitive_contracts
    )
    assert (
        facade_exports["_require_program_ad_shape_contract"]
        is program_ad_shape_transforms._require_program_ad_shape_contract
    )


def test_program_ad_shape_primitives_are_registry_policy_gated() -> None:
    """Shape transforms should expose primitive registry contracts."""
    import scpn_quantum_control as scpn

    assert (
        scpn.program_ad_shape_expand_dims_derivative_rule
        is program_ad_shape_expand_dims_derivative_rule
    )
    assert (
        scpn.program_ad_shape_squeeze_derivative_rule is program_ad_shape_squeeze_derivative_rule
    )
    assert (
        scpn.program_ad_shape_swapaxes_derivative_rule is program_ad_shape_swapaxes_derivative_rule
    )
    assert (
        scpn.program_ad_shape_moveaxis_derivative_rule is program_ad_shape_moveaxis_derivative_rule
    )
    assert scpn.program_ad_shape_roll_derivative_rule is program_ad_shape_roll_derivative_rule
    assert scpn.program_ad_shape_rot90_derivative_rule is program_ad_shape_rot90_derivative_rule
    assert scpn.program_ad_shape_flip_derivative_rule is program_ad_shape_flip_derivative_rule
    assert scpn.program_ad_shape_flipud_derivative_rule is program_ad_shape_flipud_derivative_rule
    assert scpn.program_ad_shape_fliplr_derivative_rule is program_ad_shape_fliplr_derivative_rule
    assert scpn.program_ad_shape_repeat_derivative_rule is program_ad_shape_repeat_derivative_rule
    assert scpn.program_ad_shape_tile_derivative_rule is program_ad_shape_tile_derivative_rule
    assert scpn.program_ad_shape_atleast_1d_derivative_rule is (
        program_ad_shape_atleast_1d_derivative_rule
    )
    assert scpn.program_ad_shape_atleast_2d_derivative_rule is (
        program_ad_shape_atleast_2d_derivative_rule
    )
    assert scpn.program_ad_shape_atleast_3d_derivative_rule is (
        program_ad_shape_atleast_3d_derivative_rule
    )

    scalar = np.array(2.0, dtype=np.float64)
    vector = np.arange(3.0, dtype=np.float64)
    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    cube = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    singleton = matrix.reshape(1, 2, 3, 1)
    reshape_contract = primitive_contract_for("scpn.program_ad.shape:reshape")
    reshape_shape_rule, reshape_dtype_rule, reshape_static_rule = _contract_rules(reshape_contract)
    assert reshape_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "reshape", "1")
    assert reshape_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert reshape_contract.effect == "pure"
    assert reshape_contract.lowering_metadata["mlir_op"] == "scpn_diff.shape.reshape"
    assert reshape_shape_rule((matrix, (3, -1))) == (3, 2)
    assert reshape_dtype_rule((matrix, (3, -1))) == "float64"
    assert reshape_static_rule((matrix, (3, -1))) == ((3, 2),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(reshape_contract.identity)

    ravel_contract = primitive_contract_for("scpn.program_ad.shape:ravel")
    ravel_shape_rule, ravel_dtype_rule, ravel_static_rule = _contract_rules(ravel_contract)
    assert ravel_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "ravel", "1")
    assert ravel_shape_rule((matrix,)) == (6,)
    assert ravel_dtype_rule((matrix,)) == "float64"
    assert ravel_static_rule((matrix,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(ravel_contract.identity)

    transpose_contract = primitive_contract_for("scpn.program_ad.shape:transpose")
    transpose_shape_rule, transpose_dtype_rule, transpose_static_rule = _contract_rules(
        transpose_contract
    )
    assert transpose_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "transpose", "1"
    )
    assert transpose_shape_rule((matrix, (1, 0))) == (3, 2)
    assert transpose_dtype_rule((matrix, (1, 0))) == "float64"
    assert transpose_static_rule((matrix, (1, 0))) == ((1, 0),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(transpose_contract.identity)

    expand_contract = primitive_contract_for("scpn.program_ad.shape:expand_dims")
    expand_shape_rule, expand_dtype_rule, expand_static_rule = _contract_rules(expand_contract)
    assert expand_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "expand_dims", "1"
    )
    assert expand_shape_rule((matrix, (0, -1))) == (1, 2, 3, 1)
    assert expand_dtype_rule((matrix, (0, -1))) == "float64"
    assert expand_static_rule((matrix, (0, -1))) == ((0, 3),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(expand_contract.identity)

    squeeze_contract = primitive_contract_for("scpn.program_ad.shape:squeeze")
    squeeze_shape_rule, squeeze_dtype_rule, squeeze_static_rule = _contract_rules(squeeze_contract)
    assert squeeze_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "squeeze", "1")
    assert squeeze_shape_rule((singleton, (0, -1))) == (2, 3)
    assert squeeze_dtype_rule((singleton, (0, -1))) == "float64"
    assert squeeze_static_rule((singleton, (0, -1))) == ((0, 3),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(squeeze_contract.identity)

    swapaxes_contract = primitive_contract_for("scpn.program_ad.shape:swapaxes")
    swapaxes_shape_rule, swapaxes_dtype_rule, swapaxes_static_rule = _contract_rules(
        swapaxes_contract
    )
    assert swapaxes_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "swapaxes", "1"
    )
    assert swapaxes_shape_rule((cube, 0, -1)) == (4, 3, 2)
    assert swapaxes_dtype_rule((cube, 0, -1)) == "float64"
    assert swapaxes_static_rule((cube, 0, -1)) == (0, 2)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(swapaxes_contract.identity)

    moveaxis_contract = primitive_contract_for("scpn.program_ad.shape:moveaxis")
    moveaxis_shape_rule, moveaxis_dtype_rule, moveaxis_static_rule = _contract_rules(
        moveaxis_contract
    )
    assert moveaxis_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "moveaxis", "1"
    )
    assert moveaxis_shape_rule((cube, (0, 2), (2, 0))) == (4, 3, 2)
    assert moveaxis_dtype_rule((cube, (0, 2), (2, 0))) == "float64"
    assert moveaxis_static_rule((cube, (0, 2), (2, 0))) == (
        (0, 2),
        (2, 0),
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(moveaxis_contract.identity)

    roll_contract = primitive_contract_for("scpn.program_ad.shape:roll")
    roll_shape_rule, roll_dtype_rule, roll_static_rule = _contract_rules(roll_contract)
    assert roll_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "roll", "1")
    assert roll_shape_rule((cube, (1, -2), (0, 2))) == cube.shape
    assert roll_dtype_rule((cube, (1, -2), (0, 2))) == "float64"
    assert roll_static_rule((cube, (1, -2), (0, 2))) == ((1, -2), (0, 2))
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(roll_contract.identity)

    flip_contract = primitive_contract_for("scpn.program_ad.shape:flip")
    flip_shape_rule, flip_dtype_rule, flip_static_rule = _contract_rules(flip_contract)
    assert flip_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "flip", "1")
    assert flip_shape_rule((cube, (0, -1))) == cube.shape
    assert flip_dtype_rule((cube, (0, -1))) == "float64"
    assert flip_static_rule((cube, (0, -1))) == ((0, 2),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(flip_contract.identity)

    flipud_contract = primitive_contract_for("scpn.program_ad.shape:flipud")
    flipud_shape_rule, flipud_dtype_rule, flipud_static_rule = _contract_rules(flipud_contract)
    assert flipud_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "flipud", "1")
    assert flipud_shape_rule((cube,)) == cube.shape
    assert flipud_dtype_rule((cube,)) == "float64"
    assert flipud_static_rule((cube,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(flipud_contract.identity)

    fliplr_contract = primitive_contract_for("scpn.program_ad.shape:fliplr")
    fliplr_shape_rule, fliplr_dtype_rule, fliplr_static_rule = _contract_rules(fliplr_contract)
    assert fliplr_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "fliplr", "1")
    assert fliplr_shape_rule((cube,)) == cube.shape
    assert fliplr_dtype_rule((cube,)) == "float64"
    assert fliplr_static_rule((cube,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(fliplr_contract.identity)

    rot90_contract = primitive_contract_for("scpn.program_ad.shape:rot90")
    rot90_shape_rule, rot90_dtype_rule, rot90_static_rule = _contract_rules(rot90_contract)
    assert rot90_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "rot90", "1")
    assert rot90_shape_rule((matrix, 1, (0, 1))) == (3, 2)
    assert rot90_dtype_rule((matrix, 1, (0, 1))) == "float64"
    assert rot90_static_rule((matrix, 1, (0, 1))) == (1, (0, 1))
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(rot90_contract.identity)

    repeat_contract = primitive_contract_for("scpn.program_ad.shape:repeat")
    repeat_shape_rule, repeat_dtype_rule, repeat_static_rule = _contract_rules(repeat_contract)
    assert repeat_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "repeat", "1")
    assert repeat_shape_rule((matrix, (1, 2, 0), 1)) == (2, 3)
    assert repeat_dtype_rule((matrix, (1, 2, 0), 1)) == "float64"
    assert repeat_static_rule((matrix, (1, 2, 0), 1)) == ((1, 2, 0), 1)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(repeat_contract.identity)

    tile_contract = primitive_contract_for("scpn.program_ad.shape:tile")
    tile_shape_rule, tile_dtype_rule, tile_static_rule = _contract_rules(tile_contract)
    assert tile_contract.identity == PrimitiveIdentity("scpn.program_ad.shape", "tile", "1")
    assert tile_shape_rule((matrix, (2, 1))) == (4, 3)
    assert tile_dtype_rule((matrix, (2, 1))) == "float64"
    assert tile_static_rule((matrix, (2, 1))) == ((2, 1),)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(tile_contract.identity)

    atleast_1d_contract = primitive_contract_for("scpn.program_ad.shape:atleast_1d")
    atleast_1d_shape_rule, atleast_1d_dtype_rule, atleast_1d_static_rule = _contract_rules(
        atleast_1d_contract
    )
    assert atleast_1d_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "atleast_1d", "1"
    )
    assert atleast_1d_shape_rule((scalar,)) == (1,)
    assert atleast_1d_dtype_rule((scalar,)) == "float64"
    assert atleast_1d_static_rule((scalar,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(atleast_1d_contract.identity)

    atleast_2d_contract = primitive_contract_for("scpn.program_ad.shape:atleast_2d")
    atleast_2d_shape_rule, atleast_2d_dtype_rule, atleast_2d_static_rule = _contract_rules(
        atleast_2d_contract
    )
    assert atleast_2d_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "atleast_2d", "1"
    )
    assert atleast_2d_shape_rule((vector,)) == (1, 3)
    assert atleast_2d_dtype_rule((vector,)) == "float64"
    assert atleast_2d_static_rule((vector,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(atleast_2d_contract.identity)

    atleast_3d_contract = primitive_contract_for("scpn.program_ad.shape:atleast_3d")
    atleast_3d_shape_rule, atleast_3d_dtype_rule, atleast_3d_static_rule = _contract_rules(
        atleast_3d_contract
    )
    assert atleast_3d_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.shape", "atleast_3d", "1"
    )
    assert atleast_3d_shape_rule((matrix,)) == (2, 3, 1)
    assert atleast_3d_dtype_rule((matrix,)) == "float64"
    assert atleast_3d_static_rule((matrix,)) == ()
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(atleast_3d_contract.identity)


def test_program_ad_shape_boundary_metadata_is_explicit() -> None:
    """Shape contracts should expose fail-closed static-layout boundaries."""
    expected_factories = {
        "atleast_1d": "program_ad_shape_atleast_1d_derivative_rule",
        "atleast_2d": "program_ad_shape_atleast_2d_derivative_rule",
        "atleast_3d": "program_ad_shape_atleast_3d_derivative_rule",
        "expand_dims": "program_ad_shape_expand_dims_derivative_rule",
        "flip": "program_ad_shape_flip_derivative_rule",
        "fliplr": "program_ad_shape_fliplr_derivative_rule",
        "flipud": "program_ad_shape_flipud_derivative_rule",
        "moveaxis": "program_ad_shape_moveaxis_derivative_rule",
        "repeat": "program_ad_shape_repeat_derivative_rule",
        "reshape": "program_ad_shape_reshape_derivative_rule",
        "ravel": "program_ad_shape_ravel_derivative_rule",
        "roll": "program_ad_shape_roll_derivative_rule",
        "rot90": "program_ad_shape_rot90_derivative_rule",
        "squeeze": "program_ad_shape_squeeze_derivative_rule",
        "swapaxes": "program_ad_shape_swapaxes_derivative_rule",
        "tile": "program_ad_shape_tile_derivative_rule",
        "transpose": "program_ad_shape_transpose_derivative_rule",
    }
    expected_static_signatures = {
        "atleast_1d": "source_shape:ranked_tensor_shape",
        "atleast_2d": "source_shape:ranked_tensor_shape",
        "atleast_3d": "source_shape:ranked_tensor_shape",
        "expand_dims": "source_shape:ranked_tensor_shape;axis",
        "flip": "source_shape:ranked_tensor_shape;axis",
        "fliplr": "source_shape:rank_ge_2",
        "flipud": "source_shape:rank_ge_1",
        "moveaxis": "source_shape:ranked_tensor_shape;source_destination",
        "repeat": "source_shape:ranked_tensor_shape;repeats_axis",
        "reshape": "source_shape:ranked_tensor_shape;target_shape",
        "ravel": "source_shape:ranked_tensor_shape",
        "roll": "source_shape:ranked_tensor_shape;shift_axis",
        "rot90": "source_shape:ranked_tensor_shape;k_axes",
        "squeeze": "source_shape:ranked_tensor_shape;axis",
        "swapaxes": "source_shape:ranked_tensor_shape;axis1_axis2",
        "tile": "source_shape:ranked_tensor_shape;reps",
        "transpose": "source_shape:ranked_tensor_shape;axes",
    }
    expected_boundaries = {
        "atleast_1d": "static_rank_promotion",
        "atleast_2d": "static_rank_promotion",
        "atleast_3d": "static_rank_promotion",
        "expand_dims": "static_singleton_axis_insertion",
        "flip": "static_axis_flip_permutation",
        "fliplr": "static_second_axis_flip_permutation",
        "flipud": "static_first_axis_flip_permutation",
        "moveaxis": "static_axis_move_permutation",
        "repeat": "static_repeat_scatter_add",
        "reshape": "element_count_preserving_static_shape",
        "ravel": "contiguous_flat_view_shape",
        "roll": "static_integer_roll_permutation",
        "rot90": "static_quarter_turn_axis_permutation",
        "squeeze": "static_singleton_axis_removal",
        "swapaxes": "static_axis_swap_permutation",
        "tile": "static_tile_scatter_add",
        "transpose": "static_axis_permutation",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.shape", name, "1")
        ).lowering_metadata
        assert metadata["static_derivative_factory"] == expected_factories[name]
        assert metadata["static_signature"] == expected_static_signatures[name]
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_shape_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported shape primitives must execute through registry validation rules."""
    originals = {
        name: primitive_contract_for(f"scpn.program_ad.shape:{name}")
        for name in (
            "reshape",
            "ravel",
            "transpose",
            "expand_dims",
            "roll",
            "rot90",
            "repeat",
            "flip",
            "flipud",
            "fliplr",
            "squeeze",
            "swapaxes",
            "moveaxis",
            "tile",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
        )
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        original_shape_rule = original.shape_rule
        original_dtype_rule = original.dtype_rule
        original_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], tuple[int, ...]] = original_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return wrapped_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[[tuple[object, ...]], str] = original_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return wrapped_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            wrapped_rule: Callable[
                [tuple[object, ...]], tuple[object, ...]
            ] = original_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return wrapped_rule(args)

        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=original.identity,
                derivative_rule=original.derivative_rule,
                batching_rule=original.batching_rule,
                lowering_rule=original.lowering_rule,
                lowering_metadata=original.lowering_metadata,
                shape_rule=shape_rule,
                dtype_rule=dtype_rule,
                static_argument_rule=static_argument_rule,
                nondifferentiable_policy=original.nondifferentiable_policy,
                effect=original.effect,
            ),
            overwrite=True,
        )
    try:

        def objective(values: Any) -> object:
            reshaped = np.reshape(values, (2, 2))
            swapped = np.swapaxes(reshaped, 0, 1)
            moved = np.moveaxis(swapped, 0, 1)
            repeated = np.repeat(moved, repeats=(1, 1), axis=0)
            tiled = np.tile(repeated, reps=(1, 1))
            rolled = np.roll(tiled, shift=(1, -1), axis=(0, 1))
            rotated = np.rot90(rolled, k=1, axes=(0, 1))
            flipped = np.flip(rotated, axis=(0, 1))
            flipped_ud = np.flipud(flipped)
            flipped_lr = np.fliplr(flipped_ud)
            flattened = np.ravel(np.transpose(flipped_lr))
            promoted = np.atleast_3d(np.atleast_2d(np.atleast_1d(flattened)))
            return np.sum(np.squeeze(np.expand_dims(promoted, axis=0), axis=0))

        result = whole_program_value_and_grad(
            objective,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(10.0)
    assert calls == {
        "reshape": {"shape", "dtype", "static"},
        "ravel": {"shape", "dtype", "static"},
        "transpose": {"shape", "dtype", "static"},
        "expand_dims": {"shape", "dtype", "static"},
        "roll": {"shape", "dtype", "static"},
        "rot90": {"shape", "dtype", "static"},
        "repeat": {"shape", "dtype", "static"},
        "flip": {"shape", "dtype", "static"},
        "flipud": {"shape", "dtype", "static"},
        "fliplr": {"shape", "dtype", "static"},
        "squeeze": {"shape", "dtype", "static"},
        "swapaxes": {"shape", "dtype", "static"},
        "moveaxis": {"shape", "dtype", "static"},
        "tile": {"shape", "dtype", "static"},
        "atleast_1d": {"shape", "dtype", "static"},
        "atleast_2d": {"shape", "dtype", "static"},
        "atleast_3d": {"shape", "dtype", "static"},
    }


def test_program_ad_shape_static_derivative_factories_are_direct_kernels() -> None:
    """Static shape-transform factories should expose exact value, JVP, and VJP rules."""
    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    values = matrix.reshape(-1)
    tangent = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 1.25], dtype=np.float64)
    cotangent_reshape = np.arange(1.0, 7.0, dtype=np.float64).reshape(3, 2)

    reshape_rule = program_ad_shape_reshape_derivative_rule((2, 3), (3, 2))
    assert reshape_rule.name == "program_ad_shape_reshape_2x3_to_3x2_direct_rule"
    assert reshape_rule.jvp_rule is not None
    assert reshape_rule.vjp_rule is not None
    _assert_allclose(
        reshape_rule.value_fn(values),
        matrix.reshape(3, 2).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        reshape_rule.jvp_rule(values, tangent),
        tangent.reshape(2, 3).reshape(3, 2).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        reshape_rule.vjp_rule(values, cotangent_reshape.reshape(-1)),
        cotangent_reshape.reshape(2, 3).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    ravel_rule = program_ad_shape_ravel_derivative_rule((2, 3))
    assert ravel_rule.name == "program_ad_shape_ravel_2x3_direct_rule"
    assert ravel_rule.jvp_rule is not None
    assert ravel_rule.vjp_rule is not None
    _assert_allclose(ravel_rule.value_fn(values), values, rtol=0.0, atol=0.0)
    _assert_allclose(ravel_rule.jvp_rule(values, tangent), tangent, rtol=0.0, atol=0.0)
    _assert_allclose(
        ravel_rule.vjp_rule(values, cotangent_reshape.reshape(-1)),
        cotangent_reshape.reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    transpose_rule = program_ad_shape_transpose_derivative_rule((2, 3), (1, 0))
    assert transpose_rule.name == "program_ad_shape_transpose_2x3_axes_1_0_direct_rule"
    assert transpose_rule.jvp_rule is not None
    assert transpose_rule.vjp_rule is not None
    _assert_allclose(
        transpose_rule.value_fn(values),
        matrix.transpose(1, 0).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        transpose_rule.jvp_rule(values, tangent),
        tangent.reshape(2, 3).transpose(1, 0).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        transpose_rule.vjp_rule(values, cotangent_reshape.reshape(-1)),
        cotangent_reshape.transpose(1, 0).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="same element count"):
        program_ad_shape_reshape_derivative_rule((2, 3), (4, 2))
    with pytest.raises(ValueError, match="permutation"):
        program_ad_shape_transpose_derivative_rule((2, 3), (0, 0))

    expand_rule = program_ad_shape_expand_dims_derivative_rule((2, 3), (0, -1))
    assert expand_rule.name == "program_ad_shape_expand_dims_2x3_axes_0_3_direct_rule"
    assert expand_rule.jvp_rule is not None
    assert expand_rule.vjp_rule is not None
    _assert_allclose(
        expand_rule.value_fn(values),
        np.expand_dims(matrix, axis=(0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        expand_rule.jvp_rule(values, tangent),
        np.expand_dims(tangent.reshape(2, 3), axis=(0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        expand_rule.vjp_rule(values, np.arange(1.0, 7.0, dtype=np.float64)),
        np.arange(1.0, 7.0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    singleton_values = np.expand_dims(matrix, axis=(0, -1)).reshape(-1)
    squeeze_rule = program_ad_shape_squeeze_derivative_rule((1, 2, 3, 1), (0, -1))
    assert squeeze_rule.name == "program_ad_shape_squeeze_1x2x3x1_axes_0_3_direct_rule"
    assert squeeze_rule.jvp_rule is not None
    assert squeeze_rule.vjp_rule is not None
    _assert_allclose(
        squeeze_rule.value_fn(singleton_values),
        matrix.reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        squeeze_rule.jvp_rule(singleton_values, np.expand_dims(tangent.reshape(2, 3), (0, -1))),
        tangent.reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        squeeze_rule.vjp_rule(singleton_values, np.arange(1.0, 7.0, dtype=np.float64)),
        np.expand_dims(np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3), (0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="length one"):
        program_ad_shape_squeeze_derivative_rule((2, 3), 0)

    empty_axis_squeeze = program_ad_shape_squeeze_derivative_rule((1, 2, 1), ())
    assert empty_axis_squeeze.name == "program_ad_shape_squeeze_1x2x1_axes_none_direct_rule"
    empty_axis_values = np.array([1.0, -2.0], dtype=np.float64)
    _assert_allclose(
        empty_axis_squeeze.value_fn(empty_axis_values),
        empty_axis_values,
        rtol=0.0,
        atol=0.0,
    )

    cube = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    cube_values = cube.reshape(-1)
    cube_tangent = np.linspace(-1.0, 1.0, cube.size, dtype=np.float64)
    cube_cotangent = np.linspace(0.25, 2.25, cube.size, dtype=np.float64).reshape(4, 3, 2)

    swapaxes_rule = program_ad_shape_swapaxes_derivative_rule((2, 3, 4), 0, -1)
    assert swapaxes_rule.name == "program_ad_shape_swapaxes_2x3x4_axes_0_2_direct_rule"
    assert swapaxes_rule.jvp_rule is not None
    assert swapaxes_rule.vjp_rule is not None
    _assert_allclose(
        swapaxes_rule.value_fn(cube_values),
        np.swapaxes(cube, 0, -1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        swapaxes_rule.jvp_rule(cube_values, cube_tangent),
        np.swapaxes(cube_tangent.reshape(2, 3, 4), 0, -1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        swapaxes_rule.vjp_rule(cube_values, cube_cotangent.reshape(-1)),
        np.swapaxes(cube_cotangent, 0, -1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    moveaxis_rule = program_ad_shape_moveaxis_derivative_rule((2, 3, 4), (0, 2), (2, 0))
    assert (
        moveaxis_rule.name
        == "program_ad_shape_moveaxis_2x3x4_source_0_2_destination_2_0_direct_rule"
    )
    assert moveaxis_rule.jvp_rule is not None
    assert moveaxis_rule.vjp_rule is not None
    _assert_allclose(
        moveaxis_rule.value_fn(cube_values),
        np.moveaxis(cube, (0, 2), (2, 0)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        moveaxis_rule.jvp_rule(cube_values, cube_tangent),
        np.moveaxis(cube_tangent.reshape(2, 3, 4), (0, 2), (2, 0)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        moveaxis_rule.vjp_rule(cube_values, cube_cotangent.reshape(-1)),
        np.moveaxis(cube_cotangent, (2, 0), (0, 2)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="lengths must match"):
        program_ad_shape_moveaxis_derivative_rule((2, 3, 4), (0, 1), (2,))

    roll_rule = program_ad_shape_roll_derivative_rule((2, 3, 4), shift=(1, -2), axis=(0, 2))
    assert roll_rule.name == "program_ad_shape_roll_2x3x4_shift_1_-2_axis_0_2_direct_rule"
    assert roll_rule.jvp_rule is not None
    assert roll_rule.vjp_rule is not None
    _assert_allclose(
        roll_rule.value_fn(cube_values),
        np.roll(cube, shift=(1, -2), axis=(0, 2)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        roll_rule.jvp_rule(cube_values, cube_tangent),
        np.roll(cube_tangent.reshape(2, 3, 4), shift=(1, -2), axis=(0, 2)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        roll_rule.vjp_rule(cube_values, cube_tangent),
        np.roll(cube_tangent.reshape(2, 3, 4), shift=(-1, 2), axis=(0, 2)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    flip_rule = program_ad_shape_flip_derivative_rule((2, 3, 4), axis=(0, -1))
    assert flip_rule.name == "program_ad_shape_flip_2x3x4_axis_0_2_direct_rule"
    assert flip_rule.jvp_rule is not None
    assert flip_rule.vjp_rule is not None
    _assert_allclose(
        flip_rule.value_fn(cube_values),
        np.flip(cube, axis=(0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        flip_rule.jvp_rule(cube_values, cube_tangent),
        np.flip(cube_tangent.reshape(2, 3, 4), axis=(0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        flip_rule.vjp_rule(cube_values, cube_tangent),
        np.flip(cube_tangent.reshape(2, 3, 4), axis=(0, -1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    flipud_rule = program_ad_shape_flipud_derivative_rule((2, 3, 4))
    assert flipud_rule.name == "program_ad_shape_flipud_2x3x4_direct_rule"
    assert flipud_rule.jvp_rule is not None
    assert flipud_rule.vjp_rule is not None
    _assert_allclose(
        flipud_rule.value_fn(cube_values),
        np.flipud(cube).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        flipud_rule.jvp_rule(cube_values, cube_tangent),
        np.flipud(cube_tangent.reshape(2, 3, 4)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        flipud_rule.vjp_rule(cube_values, cube_tangent),
        np.flipud(cube_tangent.reshape(2, 3, 4)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    fliplr_rule = program_ad_shape_fliplr_derivative_rule((2, 3, 4))
    assert fliplr_rule.name == "program_ad_shape_fliplr_2x3x4_direct_rule"
    assert fliplr_rule.jvp_rule is not None
    assert fliplr_rule.vjp_rule is not None
    _assert_allclose(
        fliplr_rule.value_fn(cube_values),
        np.fliplr(cube).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        fliplr_rule.jvp_rule(cube_values, cube_tangent),
        np.fliplr(cube_tangent.reshape(2, 3, 4)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        fliplr_rule.vjp_rule(cube_values, cube_tangent),
        np.fliplr(cube_tangent.reshape(2, 3, 4)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="rank-1"):
        program_ad_shape_flipud_derivative_rule(())
    with pytest.raises(ValueError, match="rank-2"):
        program_ad_shape_fliplr_derivative_rule((3,))

    rot90_cotangent = np.arange(1.0, 7.0, dtype=np.float64).reshape(3, 2)
    rot90_rule = program_ad_shape_rot90_derivative_rule((2, 3), k=1, axes=(0, 1))
    assert rot90_rule.name == "program_ad_shape_rot90_2x3_k_1_axes_0_1_direct_rule"
    assert rot90_rule.jvp_rule is not None
    assert rot90_rule.vjp_rule is not None
    _assert_allclose(
        rot90_rule.value_fn(values),
        np.rot90(matrix, k=1, axes=(0, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        rot90_rule.jvp_rule(values, tangent),
        np.rot90(tangent.reshape(2, 3), k=1, axes=(0, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        rot90_rule.vjp_rule(values, rot90_cotangent.reshape(-1)),
        np.rot90(rot90_cotangent, k=-1, axes=(0, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )

    repeat_rule = program_ad_shape_repeat_derivative_rule((2, 3), repeats=(1, 2, 0), axis=1)
    assert repeat_rule.name == "program_ad_shape_repeat_2x3_repeats_1_2_0_axis_1_direct_rule"
    assert repeat_rule.jvp_rule is not None
    assert repeat_rule.vjp_rule is not None
    repeat_cotangent = np.arange(1.0, 7.0, dtype=np.float64)
    source_indices = np.arange(matrix.size, dtype=np.int64).reshape(matrix.shape)
    repeated_indices = np.repeat(source_indices, (1, 2, 0), axis=1).reshape(-1)
    expected_repeat_adjoint = np.zeros(matrix.size, dtype=np.float64)
    np.add.at(expected_repeat_adjoint, repeated_indices, repeat_cotangent)
    _assert_allclose(
        repeat_rule.value_fn(values),
        np.repeat(matrix, (1, 2, 0), axis=1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        repeat_rule.jvp_rule(values, tangent),
        np.repeat(tangent.reshape(2, 3), (1, 2, 0), axis=1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        repeat_rule.vjp_rule(values, repeat_cotangent),
        expected_repeat_adjoint,
        rtol=0.0,
        atol=0.0,
    )

    tile_rule = program_ad_shape_tile_derivative_rule((2, 3), reps=(2, 1))
    assert tile_rule.name == "program_ad_shape_tile_2x3_reps_2_1_direct_rule"
    assert tile_rule.jvp_rule is not None
    assert tile_rule.vjp_rule is not None
    tile_cotangent = np.arange(1.0, 13.0, dtype=np.float64)
    tiled_indices = np.tile(source_indices, (2, 1)).reshape(-1)
    expected_tile_adjoint = np.zeros(matrix.size, dtype=np.float64)
    np.add.at(expected_tile_adjoint, tiled_indices, tile_cotangent)
    _assert_allclose(
        tile_rule.value_fn(values),
        np.tile(matrix, (2, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        tile_rule.jvp_rule(values, tangent),
        np.tile(tangent.reshape(2, 3), (2, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        tile_rule.vjp_rule(values, tile_cotangent),
        expected_tile_adjoint,
        rtol=0.0,
        atol=0.0,
    )

    vector_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    vector_tangent = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    atleast_1d_rule = program_ad_shape_atleast_1d_derivative_rule((3,))
    assert atleast_1d_rule.name == "program_ad_shape_atleast_1d_3_to_3_direct_rule"
    assert atleast_1d_rule.jvp_rule is not None
    _assert_allclose(atleast_1d_rule.value_fn(vector_values), vector_values)
    _assert_allclose(atleast_1d_rule.jvp_rule(vector_values, vector_tangent), vector_tangent)

    atleast_2d_rule = program_ad_shape_atleast_2d_derivative_rule((3,))
    assert atleast_2d_rule.name == "program_ad_shape_atleast_2d_3_to_1x3_direct_rule"
    assert atleast_2d_rule.jvp_rule is not None
    assert atleast_2d_rule.vjp_rule is not None
    _assert_allclose(
        atleast_2d_rule.value_fn(vector_values),
        np.atleast_2d(vector_values).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        atleast_2d_rule.jvp_rule(vector_values, vector_tangent),
        np.atleast_2d(vector_tangent).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        atleast_2d_rule.vjp_rule(vector_values, np.arange(1.0, 4.0, dtype=np.float64)),
        np.arange(1.0, 4.0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    atleast_3d_rule = program_ad_shape_atleast_3d_derivative_rule((2, 3))
    assert atleast_3d_rule.name == "program_ad_shape_atleast_3d_2x3_to_2x3x1_direct_rule"
    assert atleast_3d_rule.jvp_rule is not None
    _assert_allclose(
        atleast_3d_rule.value_fn(values),
        np.atleast_3d(matrix).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    _assert_allclose(
        atleast_3d_rule.jvp_rule(values, tangent),
        np.atleast_3d(tangent.reshape(2, 3)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )


def test_program_ad_shape_direct_factories_fail_closed_on_static_boundary_edges() -> None:
    """Shape direct-rule factories should reject invalid static transform contracts."""
    trace_contract = _program_ad_shape_derivative_rule("reshape")
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        trace_contract.value_fn(np.array([1.0], dtype=np.float64))
    assert trace_contract.jvp_rule is not None
    with pytest.raises(ValueError, match="operator-intercepted trace dispatch"):
        trace_contract.jvp_rule(
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="non-negative dimensions"):
        program_ad_shape_reshape_derivative_rule((-1,), (1,))
    reshape_rule = program_ad_shape_reshape_derivative_rule((2,), (2,))
    with pytest.raises(ValueError, match="requires values with 2 values"):
        reshape_rule.value_fn(np.array([1.0], dtype=np.float64))

    transpose_default_rule = program_ad_shape_transpose_derivative_rule((2, 3))
    _assert_allclose(
        transpose_default_rule.value_fn(np.arange(6.0, dtype=np.float64)),
        np.arange(6.0, dtype=np.float64).reshape(2, 3).transpose().reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="axis out of bounds"):
        program_ad_shape_expand_dims_derivative_rule((2,), 3)
    with pytest.raises(ValueError, match="source axes must be static integers"):
        program_ad_shape_moveaxis_derivative_rule((2, 3), cast(Any, object()), 0)

    with pytest.raises(ValueError, match="repeat counts must be static"):
        program_ad_shape_repeat_derivative_rule((2,), object(), axis=None)
    with pytest.raises(ValueError, match="repeat counts must be static"):
        program_ad_shape_repeat_derivative_rule((2,), -1, axis=None)
    flat_repeat_rule = program_ad_shape_repeat_derivative_rule((2, 3), 2, axis=None)
    _assert_allclose(
        flat_repeat_rule.value_fn(np.arange(6.0, dtype=np.float64)),
        np.repeat(np.arange(6.0, dtype=np.float64), 2),
        rtol=0.0,
        atol=0.0,
    )

    scalar_tile_rule = program_ad_shape_tile_derivative_rule((2,), 2)
    _assert_allclose(
        scalar_tile_rule.value_fn(np.array([1.0, 2.0], dtype=np.float64)),
        np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="tile reps must be static"):
        program_ad_shape_tile_derivative_rule((2,), (True,))

    axis_broadcast_roll_rule = program_ad_shape_roll_derivative_rule((2, 3), shift=1, axis=(0, 1))
    _assert_allclose(
        axis_broadcast_roll_rule.value_fn(np.arange(6.0, dtype=np.float64)),
        np.roll(np.arange(6.0, dtype=np.float64).reshape(2, 3), (1, 1), axis=(0, 1)).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    flat_roll_rule = program_ad_shape_roll_derivative_rule((2, 3), shift=1, axis=None)
    _assert_allclose(
        flat_roll_rule.value_fn(np.arange(6.0, dtype=np.float64)),
        np.roll(np.arange(6.0, dtype=np.float64).reshape(2, 3), 1).reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="roll shift must be static"):
        program_ad_shape_roll_derivative_rule((2, 3), shift=object(), axis=(0,))

    with pytest.raises(ValueError, match="rot90 k must be a static integer"):
        program_ad_shape_rot90_derivative_rule((2, 3), k=True, axes=(0, 1))
    with pytest.raises(ValueError, match="rot90 axes must contain exactly two axes"):
        program_ad_shape_rot90_derivative_rule((2, 3), k=1, axes=(0,))

    atleast_2d_scalar_rule = program_ad_shape_atleast_2d_derivative_rule(())
    _assert_allclose(
        atleast_2d_scalar_rule.value_fn(np.array([3.0], dtype=np.float64)),
        np.array([3.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    atleast_2d_matrix_rule = program_ad_shape_atleast_2d_derivative_rule((2, 3))
    _assert_allclose(
        atleast_2d_matrix_rule.value_fn(np.arange(6.0, dtype=np.float64)),
        np.arange(6.0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    atleast_3d_scalar_rule = program_ad_shape_atleast_3d_derivative_rule(())
    _assert_allclose(
        atleast_3d_scalar_rule.value_fn(np.array([4.0], dtype=np.float64)),
        np.array([4.0], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    atleast_3d_cube_rule = program_ad_shape_atleast_3d_derivative_rule((2, 3, 4))
    _assert_allclose(
        atleast_3d_cube_rule.value_fn(np.arange(24.0, dtype=np.float64)),
        np.arange(24.0, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_program_ad_squeeze_expand_dims_preserve_exact_adjoint() -> None:
    """Program AD shape-only transforms should preserve exact element adjoints."""

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (1, 2, 1, 3))
        squeezed = np.squeeze(tensor, axis=(0, 2))
        method_squeezed = tensor.squeeze()
        expanded = np.expand_dims(squeezed[1], axis=(0, 1))
        first_row = squeezed[0]
        method_expanded = (
            first_row.expand_dims(axis=1)
            if hasattr(first_row, "expand_dims")
            else np.expand_dims(first_row, axis=1)
        )
        return (
            np.sum(squeezed * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            + np.sum(method_squeezed[0] * np.array([0.5, 1.5, 2.5]))
            + np.sum(expanded * np.array([[[7.0, 11.0, 13.0]]]))
            + np.sum(method_expanded * np.array([[17.0], [19.0], [23.0]]))
        )

    values = np.array([0.2, -0.3, 0.4, 1.1, -1.2, 1.3], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = np.array([18.5, 22.5, 28.5, 11.0, 16.0, 19.0], dtype=np.float64)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_squeeze_expand_dims_fail_closed_axes() -> None:
    """Program AD shape-only transforms should reject invalid axis semantics."""
    with pytest.raises(ValueError, match="squeeze axis must have length one"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.squeeze(np.reshape(values, (2, 1)), axis=0)),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="expand_dims axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.expand_dims(values, axis=(0, 0))),
            np.array([1.0, 2.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="expand_dims axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, values).expand_dims(axis=(True,))),
            np.array([1.0, 2.0], dtype=np.float64),
        )


def test_program_ad_axis_permutations_preserve_exact_adjoint() -> None:
    """Program AD rank-N axis permutations should preserve exact element adjoints."""
    weights_swap = np.arange(24.0, dtype=np.float64).reshape(4, 3, 2) / 7.0
    weights_method = np.linspace(-1.5, 2.0, 24, dtype=np.float64).reshape(2, 4, 3)
    weights_move = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(4, 3, 2)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        swapped = np.swapaxes(tensor, 0, 2)
        method_swapped = tensor.swapaxes(1, 2)
        moved = np.moveaxis(tensor, source=(0, 2), destination=(2, 0))
        return (
            np.sum(swapped * weights_swap)
            + np.sum(method_swapped * weights_method)
            + np.sum(moved * weights_move)
        )

    values = np.linspace(-0.75, 1.5, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.swapaxes(weights_swap, 0, 2)
        + np.swapaxes(weights_method, 1, 2)
        + np.moveaxis(weights_move, source=(2, 0), destination=(0, 2))
    ).reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_roll_preserves_exact_adjoint() -> None:
    """Program AD static roll permutations should preserve exact element adjoints."""
    weights_flat = np.linspace(-2.0, 1.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_axes = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(2, 3, 4)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        flat_roll = np.roll(tensor, shift=5)
        axis_roll = np.roll(tensor, shift=(1, -2), axis=(0, 2))
        return np.sum(flat_roll * weights_flat) + np.sum(axis_roll * weights_axes)

    values = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.roll(weights_flat.reshape(-1), shift=-5).reshape(2, 3, 4)
        + np.roll(weights_axes, shift=(-1, 2), axis=(0, 2))
    ).reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_repeat_accumulates_exact_adjoint() -> None:
    """Program AD repeat should accumulate adjoints from repeated source elements."""
    flat_repeats = (1, 2, 0, 3, 1, 2)
    weights_flat = np.linspace(-2.0, 2.0, sum(flat_repeats), dtype=np.float64)
    axis_repeats = (2, 1, 3)
    weights_axis = np.linspace(0.5, 3.5, 12, dtype=np.float64).reshape(2, 6)
    weights_method = np.linspace(-1.25, 1.75, 12, dtype=np.float64).reshape(4, 3)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        flat = np.repeat(matrix, flat_repeats)
        axis_repeat = np.repeat(matrix, axis_repeats, axis=1)
        method_repeat = matrix.repeat(2, axis=0)
        return (
            np.sum(flat * weights_flat)
            + np.sum(axis_repeat * weights_axis)
            + np.sum(method_repeat * weights_method)
        )

    values = np.linspace(-0.8, 0.9, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    flat_indices = np.repeat(np.arange(6, dtype=np.int64), flat_repeats)
    expected = np.zeros(6, dtype=np.float64)
    np.add.at(expected, flat_indices, weights_flat)
    axis_indices = np.repeat(np.arange(6, dtype=np.int64).reshape(2, 3), axis_repeats, axis=1)
    np.add.at(expected, axis_indices.reshape(-1), weights_axis.reshape(-1))
    method_indices = np.repeat(np.arange(6, dtype=np.int64).reshape(2, 3), 2, axis=0)
    np.add.at(expected, method_indices.reshape(-1), weights_method.reshape(-1))

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_repeat_fails_closed_invalid_static_contracts() -> None:
    """Program AD repeat should reject invalid static repeat contracts."""
    with pytest.raises(ValueError, match="repeat counts must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(values, True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="repeat counts length must match selected axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(np.reshape(values, (2, 2)), (1, 2, 3), axis=1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="repeat axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.repeat(np.reshape(values, (2, 2)), 2, axis=2)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_tile_accumulates_exact_adjoint() -> None:
    """Program AD tile should accumulate adjoints from every tiled source use."""
    weights_matrix = np.linspace(-2.5, 3.5, 24, dtype=np.float64).reshape(4, 6)
    weights_promoted = np.linspace(0.25, 2.25, 36, dtype=np.float64).reshape(3, 2, 6)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        tiled = np.tile(matrix, (2, 2))
        promoted = np.tile(matrix, (3, 1, 2))
        return np.sum(tiled * weights_matrix) + np.sum(promoted * weights_promoted)

    values = np.linspace(-0.6, 1.1, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    source = np.arange(6, dtype=np.int64).reshape(2, 3)
    expected = np.zeros(6, dtype=np.float64)
    np.add.at(expected, np.tile(source, (2, 2)).reshape(-1), weights_matrix.reshape(-1))
    np.add.at(
        expected,
        np.tile(source.reshape(1, 2, 3), (3, 1, 2)).reshape(-1),
        weights_promoted.reshape(-1),
    )

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_tile_fails_closed_invalid_static_contracts() -> None:
    """Program AD tile should reject dynamic or invalid repetition contracts."""
    with pytest.raises(ValueError, match="tile reps must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="tile reps must be static non-negative integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, (2, -1))),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="tile reps must contain at least one axis"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tile(values, ())),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_atleast_rank_transforms_preserve_exact_adjoint() -> None:
    """Program AD atleast transforms should preserve derivatives through rank lifts."""
    vector_weights_2d = np.linspace(-1.5, 2.5, 6, dtype=np.float64).reshape(1, 6)
    vector_weights_3d = np.linspace(0.25, 2.75, 6, dtype=np.float64).reshape(1, 6, 1)
    matrix_weights = np.linspace(-2.0, 2.0, 6, dtype=np.float64).reshape(2, 3, 1)
    multi_left_weights = np.linspace(-0.75, 1.25, 6, dtype=np.float64)
    multi_right_weights = np.linspace(1.5, 3.0, 3, dtype=np.float64).reshape(1, 3)

    def objective(values: Any) -> object:
        vector = values[:6]
        matrix = np.reshape(values[:6], (2, 3))
        left, right = np.atleast_1d(vector, values[1:4])
        return (
            np.sum(np.atleast_2d(vector) * vector_weights_2d)
            + np.sum(np.atleast_3d(vector) * vector_weights_3d)
            + np.sum(np.atleast_3d(matrix) * matrix_weights)
            + np.sum(left * multi_left_weights)
            + np.sum(np.atleast_2d(right) * multi_right_weights)
        )

    values = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = np.zeros(6, dtype=np.float64)
    expected += vector_weights_2d.reshape(-1)
    expected += vector_weights_3d.reshape(-1)
    expected += matrix_weights.reshape(-1)
    expected += multi_left_weights
    expected[1:4] += multi_right_weights.reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_atleast_rank_transforms_fail_closed_invalid_contracts() -> None:
    """Program AD atleast transforms should reject non-NumPy keyword contracts."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.atleast_2d)(values, dtype=float)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_reshape_inferred_dimension_preserves_exact_adjoint() -> None:
    """Program AD reshape should support one inferred dimension exactly."""
    matrix_weights = np.linspace(-2.0, 2.0, 6, dtype=np.float64).reshape(2, 3)
    method_weights = np.linspace(0.5, 3.5, 6, dtype=np.float64).reshape(2, 3)
    promoted_weights = np.linspace(-1.25, 1.75, 6, dtype=np.float64).reshape(3, 2)

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (-1, 3))
        method_matrix = values.reshape(2, -1)
        promoted = np.reshape(values, (3, -1))
        return (
            np.sum(matrix * matrix_weights)
            + np.sum(method_matrix * method_weights)
            + np.sum(promoted * promoted_weights)
        )

    values = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected = (
        matrix_weights.reshape(-1) + method_weights.reshape(-1) + promoted_weights.reshape(-1)
    )
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_reshape_inferred_dimension_fails_closed_invalid_contracts() -> None:
    """Program AD reshape should reject ambiguous or size-losing inferred shapes."""
    with pytest.raises(ValueError, match="at most one inferred dimension"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (-1, -1))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="inferred dimension must preserve size"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (4, -1))),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dimensions must be non-negative or -1"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.reshape(values, (2, -2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_rot90_preserves_exact_adjoint() -> None:
    """Program AD rot90 permutations should preserve exact element adjoints."""
    weights_default = np.linspace(-2.0, 1.0, 12, dtype=np.float64).reshape(4, 3)
    weights_axes = np.linspace(0.5, 3.5, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_negative = np.linspace(-1.5, 2.5, 24, dtype=np.float64).reshape(2, 4, 3)

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:12], (3, 4))
        tensor = np.reshape(values, (2, 3, 4))
        return (
            np.sum(np.rot90(matrix) * weights_default)
            + np.sum(np.rot90(tensor, k=2, axes=(0, 1)) * weights_axes)
            + np.sum(np.rot90(tensor, k=-1, axes=(1, 2)) * weights_negative)
        )

    values = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected.reshape(-1)[:12] += np.rot90(weights_default, k=-1).reshape(-1)
    expected += np.rot90(weights_axes, k=-2, axes=(0, 1))
    expected += np.rot90(weights_negative, k=1, axes=(1, 2))

    _assert_allclose(result.gradient, expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected.reshape(-1), rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_rot90_fails_closed_invalid_static_contracts() -> None:
    """Program AD rot90 should reject invalid rotation contracts."""
    rot90 = cast(Any, np.rot90)
    with pytest.raises(ValueError, match="rot90 k must be a static integer"):
        whole_program_value_and_grad(
            lambda values: np.sum(rot90(np.reshape(values, (2, 2)), k=True)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="rot90 axes must contain exactly two axes"):
        whole_program_value_and_grad(
            lambda values: np.sum(rot90(np.reshape(values, (2, 2, 1)), axes=(0, 1, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="rot90 axes axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.rot90(np.reshape(values, (2, 2)), axes=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_flip_family_preserves_exact_adjoint() -> None:
    """Program AD flip-family permutations should preserve exact element adjoints."""
    weights_all = np.linspace(-1.0, 2.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_axis = np.linspace(0.25, 3.25, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_tuple = np.linspace(-2.5, 1.5, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_ud = np.linspace(1.0, 4.0, 24, dtype=np.float64).reshape(2, 3, 4)
    weights_lr = np.linspace(-3.0, -0.25, 24, dtype=np.float64).reshape(2, 3, 4)

    def objective(values: Any) -> object:
        tensor = np.reshape(values, (2, 3, 4))
        return (
            np.sum(np.flip(tensor) * weights_all)
            + np.sum(np.flip(tensor, axis=1) * weights_axis)
            + np.sum(np.flip(tensor, axis=(0, 2)) * weights_tuple)
            + np.sum(np.flipud(tensor) * weights_ud)
            + np.sum(np.fliplr(tensor) * weights_lr)
        )

    values = np.linspace(-1.25, 1.25, 24, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )
    expected = (
        np.flip(weights_all)
        + np.flip(weights_axis, axis=1)
        + np.flip(weights_tuple, axis=(0, 2))
        + np.flipud(weights_ud)
        + np.fliplr(weights_lr)
    ).reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_flip_family_fails_closed_invalid_axes() -> None:
    """Program AD flip-family permutations should reject invalid axes."""
    flip = cast(Any, np.flip)
    with pytest.raises(ValueError, match="flip axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(flip(values, axis=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="flip axis axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.flip(np.reshape(values, (2, 2)), axis=(0, 0))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="fliplr requires at least rank-2 arrays"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.fliplr(values)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )


def test_program_ad_roll_fails_closed_invalid_static_contracts() -> None:
    """Program AD roll should reject dynamic or inconsistent permutation contracts."""
    roll = cast(Any, np.roll)
    with pytest.raises(ValueError, match="roll shift must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(roll(values, shift=True)),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="roll shift and axis lengths must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.roll(np.reshape(values, (2, 2)), shift=(1, 2), axis=0)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="roll axis out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.roll(np.reshape(values, (2, 2)), shift=1, axis=2)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )


def test_program_ad_axis_permutations_fail_closed_invalid_axes() -> None:
    """Program AD axis permutations should reject invalid static axis contracts."""
    with pytest.raises(ValueError, match="swapaxes axes must be static integers"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.reshape(values, (2, 2))).swapaxes(True, 1)),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="moveaxis source and destination lengths must match"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.moveaxis(np.reshape(values, (2, 2, 1)), (0, 1), (2,))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="moveaxis source axes must be unique"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.moveaxis(np.reshape(values, (2, 2, 1)), (0, 0), (1, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )
