# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD shape transform tests
"""Tests for Program AD shape-transform adjoints and static contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


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
