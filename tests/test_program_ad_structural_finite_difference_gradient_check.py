# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD structural finite difference gradient check tests
# scpn-quantum-control -- Structural Program AD finite-difference gradient checks
"""Independent finite-difference cross-checks of structural Program AD rules.

The elementwise, reduction, product, and linalg primitive rules carry the
numerical derivatives and are cross-checked in
``test_program_ad_finite_difference_gradient_check``. This module covers the
complementary structural (shape-transforming) and routing (scatter/gather)
families, whose derivatives are linear index permutations rather than
closed-form calculus:

* shape transforms -- reshape, ravel, transpose, expand_dims, squeeze,
  swapaxes, moveaxis, roll, flip, flipud, fliplr, repeat, tile, atleast_*;
* array indexing -- getitem, take, take_along_axis, delete, pad, insert;
* assembly -- split, tril, triu, diagonal;
* broadcast -- broadcast_to, broadcast_arrays;
* stack/block -- concatenate, stack, hstack, vstack, column_stack, dstack,
  append, block.

Each rule maps a flat input vector to a flat output vector by a fixed linear
gather (forward) whose transpose is the scatter (reverse). Because the map is
linear the finite-difference Jacobian equals the exact Jacobian to rounding, so
these checks pin the routing tables (which index goes where, and the transpose
identity between ``jvp`` and ``vjp``) rather than a Taylor approximation. The
per-family registry tests assert against hand-written expected arrays; a
transposed axis or an off-by-one gather index can leave a rule and its expected
array wrong the same way, which an independent finite-difference anchor catches.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_finite_difference import finite_difference_jacobian
from scpn_quantum_control.program_ad_array_indexing import (
    program_ad_array_delete_derivative_rule,
    program_ad_array_getitem_derivative_rule,
    program_ad_array_insert_derivative_rule,
    program_ad_array_pad_derivative_rule,
    program_ad_array_take_along_axis_derivative_rule,
    program_ad_array_take_derivative_rule,
)
from scpn_quantum_control.program_ad_assembly_primitives import (
    program_ad_assembly_diagonal_derivative_rule,
    program_ad_assembly_split_derivative_rule,
    program_ad_assembly_tril_derivative_rule,
    program_ad_assembly_triu_derivative_rule,
)
from scpn_quantum_control.program_ad_broadcast_assembly import (
    program_ad_assembly_broadcast_arrays_derivative_rule,
    program_ad_assembly_broadcast_to_derivative_rule,
)
from scpn_quantum_control.program_ad_registry import CustomDerivativeRule
from scpn_quantum_control.program_ad_shape_transforms import (
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
    program_ad_shape_squeeze_derivative_rule,
    program_ad_shape_swapaxes_derivative_rule,
    program_ad_shape_tile_derivative_rule,
    program_ad_shape_transpose_derivative_rule,
)
from scpn_quantum_control.program_ad_stack_block_assembly import (
    program_ad_assembly_append_derivative_rule,
    program_ad_assembly_block_derivative_rule,
    program_ad_assembly_column_stack_derivative_rule,
    program_ad_assembly_concatenate_derivative_rule,
    program_ad_assembly_dstack_derivative_rule,
    program_ad_assembly_hstack_derivative_rule,
    program_ad_assembly_stack_derivative_rule,
    program_ad_assembly_vstack_derivative_rule,
)

_RNG = np.random.default_rng(20260701)


def _values(*shapes: tuple[int, ...]) -> NDArray[np.float64]:
    """Return a flat vector packing one random operand per shape.

    Multi-operand structural primitives (concatenate, stack, append, block,
    broadcast_arrays) pack every operand into a single contiguous flat input in
    declaration order; this mirrors that packing so a rule built from those
    operand shapes receives a conformant input vector.
    """

    return np.concatenate(
        [_RNG.normal(size=shape).reshape(-1).astype(np.float64) for shape in shapes]
    )


def _analytic_jvp_jacobian(
    rule: CustomDerivativeRule, values: NDArray[np.float64], out_size: int
) -> NDArray[np.float64]:
    """Assemble the forward-rule Jacobian one column per unit input tangent."""

    assert rule.jvp_rule is not None
    jacobian = np.zeros((out_size, values.size), dtype=np.float64)
    for column in range(values.size):
        tangent = np.zeros(values.size, dtype=np.float64)
        tangent[column] = 1.0
        jacobian[:, column] = np.asarray(rule.jvp_rule(values, tangent), dtype=np.float64).reshape(
            -1
        )
    return jacobian


def _analytic_vjp_jacobian(
    rule: CustomDerivativeRule, values: NDArray[np.float64], out_size: int
) -> NDArray[np.float64]:
    """Assemble the reverse-rule Jacobian one row per unit output cotangent."""

    assert rule.vjp_rule is not None
    jacobian = np.zeros((out_size, values.size), dtype=np.float64)
    for row in range(out_size):
        cotangent = np.zeros(out_size, dtype=np.float64)
        cotangent[row] = 1.0
        jacobian[row, :] = np.asarray(rule.vjp_rule(values, cotangent), dtype=np.float64).reshape(
            -1
        )
    return jacobian


def _assert_routing_rule_matches_finite_difference(
    rule: CustomDerivativeRule,
    values: NDArray[np.float64],
    *,
    atol: float = 1e-8,
    step: float = 1e-6,
) -> None:
    """Assert forward and reverse routing rules match the FD Jacobian.

    A structural rule is an exact linear gather, so the central finite
    difference reproduces the analytic Jacobian to rounding and the ``jvp``
    Jacobian must equal the ``vjp`` Jacobian exactly (the routing table and its
    transpose). Both are asserted on absolute tolerance because the exact
    Jacobian entries are ``0`` or ``1``.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.asarray(rule.value_fn(values), dtype=np.float64).reshape(-1)
    finite_difference = finite_difference_jacobian(rule.value_fn, values, step=step)
    analytic_forward = _analytic_jvp_jacobian(rule, values, out.size)
    analytic_reverse = _analytic_vjp_jacobian(rule, values, out.size)
    np.testing.assert_allclose(analytic_forward, finite_difference, rtol=0.0, atol=atol)
    np.testing.assert_allclose(analytic_reverse, finite_difference, rtol=0.0, atol=atol)
    # Forward and reverse are exact transposes of one linear map: identical.
    np.testing.assert_allclose(analytic_forward, analytic_reverse, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Shape transforms
# --------------------------------------------------------------------------- #
_SHAPE_RULES: dict[str, tuple[CustomDerivativeRule, tuple[int, ...]]] = {
    "reshape": (program_ad_shape_reshape_derivative_rule((2, 3), (3, 2)), (2, 3)),
    "ravel": (program_ad_shape_ravel_derivative_rule((2, 3)), (2, 3)),
    "transpose_2d": (program_ad_shape_transpose_derivative_rule((2, 3)), (2, 3)),
    "transpose_axes": (
        program_ad_shape_transpose_derivative_rule((2, 3, 4), (2, 0, 1)),
        (2, 3, 4),
    ),
    "expand_dims": (program_ad_shape_expand_dims_derivative_rule((3,), 0), (3,)),
    "expand_dims_multi": (program_ad_shape_expand_dims_derivative_rule((3,), (0, 2)), (3,)),
    "squeeze": (program_ad_shape_squeeze_derivative_rule((1, 3, 1)), (1, 3, 1)),
    "squeeze_axis": (program_ad_shape_squeeze_derivative_rule((1, 3, 1), 0), (1, 3, 1)),
    "swapaxes": (program_ad_shape_swapaxes_derivative_rule((2, 3, 4), 0, 2), (2, 3, 4)),
    "moveaxis": (program_ad_shape_moveaxis_derivative_rule((2, 3, 4), 0, 2), (2, 3, 4)),
    "roll": (program_ad_shape_roll_derivative_rule((5,), 2), (5,)),
    "roll_axis": (program_ad_shape_roll_derivative_rule((2, 3), 1, 1), (2, 3)),
    "flip": (program_ad_shape_flip_derivative_rule((5,)), (5,)),
    "flip_axis": (program_ad_shape_flip_derivative_rule((2, 3), 0), (2, 3)),
    "flipud": (program_ad_shape_flipud_derivative_rule((4, 2)), (4, 2)),
    "fliplr": (program_ad_shape_fliplr_derivative_rule((2, 4)), (2, 4)),
    "repeat": (program_ad_shape_repeat_derivative_rule((3,), 2), (3,)),
    "repeat_axis": (program_ad_shape_repeat_derivative_rule((2, 3), 2, 1), (2, 3)),
    "tile": (program_ad_shape_tile_derivative_rule((3,), 2), (3,)),
    "tile_2d": (program_ad_shape_tile_derivative_rule((2, 2), (1, 2)), (2, 2)),
    "atleast_1d": (program_ad_shape_atleast_1d_derivative_rule((3,)), (3,)),
    "atleast_2d": (program_ad_shape_atleast_2d_derivative_rule((3,)), (3,)),
    "atleast_3d": (program_ad_shape_atleast_3d_derivative_rule((3,)), (3,)),
}


@pytest.mark.parametrize("name", sorted(_SHAPE_RULES))
def test_shape_transform_rule_matches_finite_difference(name: str) -> None:
    rule, shape = _SHAPE_RULES[name]
    _assert_routing_rule_matches_finite_difference(rule, _values(shape))


# --------------------------------------------------------------------------- #
# Array indexing
# --------------------------------------------------------------------------- #
_INDEXING_RULES: dict[str, tuple[CustomDerivativeRule, tuple[int, ...]]] = {
    "getitem_slice": (program_ad_array_getitem_derivative_rule((4,), slice(1, 3)), (4,)),
    "getitem_2d": (program_ad_array_getitem_derivative_rule((3, 4), (slice(None), 1)), (3, 4)),
    "take": (program_ad_array_take_derivative_rule((5,), [0, 2, 4]), (5,)),
    "take_axis": (program_ad_array_take_derivative_rule((2, 3), [0, 2], axis=1), (2, 3)),
    "take_along_axis": (
        program_ad_array_take_along_axis_derivative_rule(
            (2, 3), np.array([[0, 2, 1], [1, 0, 2]]), axis=1
        ),
        (2, 3),
    ),
    "delete": (program_ad_array_delete_derivative_rule((5,), 2), (5,)),
    "delete_axis": (program_ad_array_delete_derivative_rule((2, 3), 1, axis=1), (2, 3)),
    "pad": (program_ad_array_pad_derivative_rule((3,), (1, 2)), (3,)),
    "pad_2d": (program_ad_array_pad_derivative_rule((2, 2), ((1, 0), (0, 1))), (2, 2)),
    "insert": (program_ad_array_insert_derivative_rule((4,), 1, 0.0), (4,)),
}


@pytest.mark.parametrize("name", sorted(_INDEXING_RULES))
def test_array_indexing_rule_matches_finite_difference(name: str) -> None:
    rule, shape = _INDEXING_RULES[name]
    _assert_routing_rule_matches_finite_difference(rule, _values(shape))


# --------------------------------------------------------------------------- #
# Assembly (split / triangular / diagonal)
# --------------------------------------------------------------------------- #
_ASSEMBLY_RULES: dict[str, tuple[CustomDerivativeRule, tuple[int, ...]]] = {
    "split": (program_ad_assembly_split_derivative_rule((6,), 3), (6,)),
    "split_axis": (program_ad_assembly_split_derivative_rule((2, 4), 2, axis=1), (2, 4)),
    "tril": (program_ad_assembly_tril_derivative_rule((3, 3)), (3, 3)),
    "tril_k": (program_ad_assembly_tril_derivative_rule((3, 3), k=1), (3, 3)),
    "triu": (program_ad_assembly_triu_derivative_rule((3, 3)), (3, 3)),
    "diagonal": (program_ad_assembly_diagonal_derivative_rule((3, 3)), (3, 3)),
    "diagonal_offset": (program_ad_assembly_diagonal_derivative_rule((3, 3), offset=1), (3, 3)),
}


@pytest.mark.parametrize("name", sorted(_ASSEMBLY_RULES))
def test_assembly_rule_matches_finite_difference(name: str) -> None:
    rule, shape = _ASSEMBLY_RULES[name]
    _assert_routing_rule_matches_finite_difference(rule, _values(shape))


# --------------------------------------------------------------------------- #
# Broadcast
# --------------------------------------------------------------------------- #
def test_broadcast_to_rule_matches_finite_difference() -> None:
    rule = program_ad_assembly_broadcast_to_derivative_rule((3,), (2, 3))
    _assert_routing_rule_matches_finite_difference(rule, _values((3,)))


def test_broadcast_arrays_rule_matches_finite_difference() -> None:
    rule = program_ad_assembly_broadcast_arrays_derivative_rule([(3,), (2, 1)])
    _assert_routing_rule_matches_finite_difference(rule, _values((3,), (2, 1)))


# --------------------------------------------------------------------------- #
# Stack / block (multi-operand packing)
# --------------------------------------------------------------------------- #
_STACK_RULES: dict[str, tuple[CustomDerivativeRule, tuple[tuple[int, ...], ...]]] = {
    "concatenate": (
        program_ad_assembly_concatenate_derivative_rule([(2,), (3,)]),
        ((2,), (3,)),
    ),
    "concatenate_axis": (
        program_ad_assembly_concatenate_derivative_rule([(2, 2), (2, 3)], axis=1),
        ((2, 2), (2, 3)),
    ),
    "stack": (program_ad_assembly_stack_derivative_rule([(3,), (3,)]), ((3,), (3,))),
    "stack_axis": (
        program_ad_assembly_stack_derivative_rule([(2, 3), (2, 3)], axis=1),
        ((2, 3), (2, 3)),
    ),
    "hstack": (program_ad_assembly_hstack_derivative_rule([(2,), (3,)]), ((2,), (3,))),
    "vstack": (program_ad_assembly_vstack_derivative_rule([(3,), (3,)]), ((3,), (3,))),
    "column_stack": (
        program_ad_assembly_column_stack_derivative_rule([(3,), (3,)]),
        ((3,), (3,)),
    ),
    "dstack": (program_ad_assembly_dstack_derivative_rule([(3,), (3,)]), ((3,), (3,))),
    "append": (program_ad_assembly_append_derivative_rule((4,), (2,)), ((4,), (2,))),
    "block": (
        program_ad_assembly_block_derivative_rule([(2, 2), (2, 3)]),
        ((2, 2), (2, 3)),
    ),
}


@pytest.mark.parametrize("name", sorted(_STACK_RULES))
def test_stack_block_rule_matches_finite_difference(name: str) -> None:
    rule, shapes = _STACK_RULES[name]
    _assert_routing_rule_matches_finite_difference(rule, _values(*shapes))
