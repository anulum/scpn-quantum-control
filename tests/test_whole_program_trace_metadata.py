# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program trace metadata tests
# scpn-quantum-control -- Whole-program AD trace operand metadata tests
"""Tests for static normalisation of whole-program AD trace shapes and axes.

Each pure shape/axis normalisation helper is exercised on its accepting forms
(scalar and sequence operands, negative-axis wrapping, inferred reshape
dimensions) and on every fail-closed validation branch (out-of-range axes,
duplicate axes, non-integer or non-iterable operands, size-preservation
violations, and incompatible broadcasts).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from scpn_quantum_control.whole_program_trace_metadata import (
    _broadcast_shape,
    _normalise_axis,
    _normalise_axis_permutation_axes,
    _normalise_axis_permutation_axis,
    _normalise_repeat_count,
    _normalise_repeat_counts,
    _normalise_roll_shift_scalar,
    _normalise_roll_shift_tuple,
    _normalise_rot90_axes,
    _normalise_rot90_k,
    _normalise_shape_transform_axes,
    _normalise_sort_axis,
    _normalise_tile_reps,
    _normalise_trace_broadcast_shape,
    _normalise_trace_reshape_shape,
    _normalise_trapezoid_axis,
)

# ------------------------------------------------------------------ normalise_axis


def test_normalise_axis_wraps_and_validates() -> None:
    """Normalise axis wraps and validates."""
    assert _normalise_axis("axis", 1, 3) == 1
    assert _normalise_axis("axis", -1, 3) == 2


def test_normalise_axis_rejects_scalar_rank() -> None:
    """Normalise axis rejects scalar rank."""
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        _normalise_axis("axis", 0, 0)


def test_normalise_axis_rejects_out_of_bounds() -> None:
    """Normalise axis rejects out of bounds."""
    with pytest.raises(ValueError, match="out of bounds"):
        _normalise_axis("axis", 5, 3)


# -------------------------------------------------------------------- reshape_shape


def test_reshape_shape_accepts_scalar_and_sequence() -> None:
    """Reshape shape accepts scalar and sequence."""
    assert _normalise_trace_reshape_shape(6, 6) == (6,)
    assert _normalise_trace_reshape_shape((2, 3), 6) == (2, 3)


def test_reshape_shape_infers_single_dimension() -> None:
    """Reshape shape infers single dimension."""
    assert _normalise_trace_reshape_shape((2, -1), 6) == (2, 3)


def test_reshape_shape_rejects_multiple_inferred() -> None:
    """Reshape shape rejects multiple inferred."""
    with pytest.raises(ValueError, match="at most one inferred"):
        _normalise_trace_reshape_shape((-1, -1), 6)


def test_reshape_shape_rejects_dimension_below_minus_one() -> None:
    """Reshape shape rejects dimension below minus one."""
    with pytest.raises(ValueError, match="non-negative or -1"):
        _normalise_trace_reshape_shape((2, -2), 6)


def test_reshape_shape_rejects_zero_product_inference() -> None:
    """Reshape shape rejects zero product inference."""
    with pytest.raises(ValueError, match="infer dimension from zero product"):
        _normalise_trace_reshape_shape((0, -1), 6)


def test_reshape_shape_rejects_non_divisible_inference() -> None:
    """Reshape shape rejects non divisible inference."""
    with pytest.raises(ValueError, match="must preserve size"):
        _normalise_trace_reshape_shape((4, -1), 6)


def test_reshape_shape_rejects_size_mismatch() -> None:
    """Reshape shape rejects size mismatch."""
    with pytest.raises(ValueError, match="must preserve size"):
        _normalise_trace_reshape_shape((2, 2), 6)


def test_reshape_shape_rejects_string() -> None:
    """Reshape shape rejects string."""
    with pytest.raises(ValueError, match="static integer or shape tuple"):
        _normalise_trace_reshape_shape("23", 6)


def test_reshape_shape_rejects_non_integer_dimension() -> None:
    """Reshape shape rejects non integer dimension."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_trace_reshape_shape((2, 1.5), 6)


# ------------------------------------------------------------------ broadcast_shape


def test_broadcast_shape_accepts_scalar_and_sequence() -> None:
    """Broadcast shape accepts scalar and sequence."""
    assert _normalise_trace_broadcast_shape(4) == (4,)
    assert _normalise_trace_broadcast_shape((2, 3)) == (2, 3)


def test_broadcast_shape_rejects_negative() -> None:
    """Broadcast shape rejects negative."""
    with pytest.raises(ValueError, match="non-negative"):
        _normalise_trace_broadcast_shape((2, -1))


def test_broadcast_shape_rejects_non_integer() -> None:
    """Broadcast shape rejects non integer."""
    with pytest.raises(ValueError, match="requires an integer shape"):
        _normalise_trace_broadcast_shape("23")


# ----------------------------------------------------------- shape_transform_axes


def test_shape_transform_axes_normalises_and_sorts() -> None:
    """Shape transform axes normalises and sorts."""
    assert _normalise_shape_transform_axes("squeeze", (2, -3), output_rank=3) == (0, 2)
    assert _normalise_shape_transform_axes("squeeze", 1, output_rank=3) == (1,)


def test_shape_transform_axes_rejects_out_of_bounds() -> None:
    """Shape transform axes rejects out of bounds."""
    with pytest.raises(ValueError, match="out of bounds"):
        _normalise_shape_transform_axes("squeeze", 5, output_rank=3)


def test_shape_transform_axes_rejects_duplicates() -> None:
    """Shape transform axes rejects duplicates."""
    with pytest.raises(ValueError, match="must be unique"):
        _normalise_shape_transform_axes("squeeze", (1, 1), output_rank=3)


def test_shape_transform_axes_rejects_non_integer() -> None:
    """Shape transform axes rejects non integer."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_shape_transform_axes("squeeze", cast(Any, (1.0,)), output_rank=3)


# -------------------------------------------------------- axis_permutation_axis


def test_axis_permutation_axis_wraps() -> None:
    """Axis permutation axis wraps."""
    assert _normalise_axis_permutation_axis("moveaxis", -1, rank=3) == 2


def test_axis_permutation_axis_rejects_bool() -> None:
    """Axis permutation axis rejects bool."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_axis_permutation_axis("moveaxis", True, rank=3)


def test_axis_permutation_axis_rejects_out_of_bounds() -> None:
    """Axis permutation axis rejects out of bounds."""
    with pytest.raises(ValueError, match="out of bounds"):
        _normalise_axis_permutation_axis("moveaxis", 9, rank=3)


# ------------------------------------------------------- axis_permutation_axes


def test_axis_permutation_axes_accepts_scalar_and_sequence() -> None:
    """Axis permutation axes accepts scalar and sequence."""
    assert _normalise_axis_permutation_axes("moveaxis", 1, rank=3, role="source") == (1,)
    assert _normalise_axis_permutation_axes("moveaxis", (0, 2), rank=3, role="source") == (0, 2)


def test_axis_permutation_axes_rejects_non_iterable() -> None:
    """Axis permutation axes rejects non iterable."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_axis_permutation_axes("moveaxis", object(), rank=3, role="source")


def test_axis_permutation_axes_rejects_duplicates() -> None:
    """Axis permutation axes rejects duplicates."""
    with pytest.raises(ValueError, match="must be unique"):
        _normalise_axis_permutation_axes("moveaxis", (1, 1), rank=3, role="source")


# --------------------------------------------------------------------- repeat


def test_repeat_count_accepts_non_negative() -> None:
    """Repeat count accepts non negative."""
    assert _normalise_repeat_count(3) == 3


def test_repeat_count_rejects_bool() -> None:
    """Repeat count rejects bool."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_repeat_count(True)


def test_repeat_count_rejects_negative() -> None:
    """Repeat count rejects negative."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_repeat_count(-1)


def test_repeat_counts_scalar_and_sequence() -> None:
    """Repeat counts scalar and sequence."""
    assert _normalise_repeat_counts(2, 3) == 2
    assert _normalise_repeat_counts((1, 2, 3), 3) == (1, 2, 3)


def test_repeat_counts_rejects_length_mismatch() -> None:
    """Repeat counts rejects length mismatch."""
    with pytest.raises(ValueError, match="length must match"):
        _normalise_repeat_counts((1, 2), 3)


def test_repeat_counts_rejects_non_iterable() -> None:
    """Repeat counts rejects non iterable."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_repeat_counts(object(), 3)


# ----------------------------------------------------------------------- tile


def test_tile_reps_scalar_and_sequence() -> None:
    """Tile reps scalar and sequence."""
    assert _normalise_tile_reps(2) == (2,)
    assert _normalise_tile_reps((2, 3)) == (2, 3)


def test_tile_reps_rejects_empty() -> None:
    """Tile reps rejects empty."""
    with pytest.raises(ValueError, match="at least one axis"):
        _normalise_tile_reps(())


def test_tile_reps_rejects_non_integer() -> None:
    """Tile reps rejects non integer."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_tile_reps((2, 1.0))


def test_tile_reps_rejects_negative() -> None:
    """Tile reps rejects negative."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_tile_reps((2, -1))


def test_tile_reps_rejects_non_iterable() -> None:
    """Tile reps rejects non iterable."""
    with pytest.raises(ValueError, match="non-negative integers"):
        _normalise_tile_reps(object())


# ----------------------------------------------------------------------- roll


def test_roll_shift_scalar_accepts_integer() -> None:
    """Roll shift scalar accepts integer."""
    assert _normalise_roll_shift_scalar(-2) == -2


def test_roll_shift_scalar_rejects_bool() -> None:
    """Roll shift scalar rejects bool."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_roll_shift_scalar(True)


def test_roll_shift_tuple_broadcasts_scalar() -> None:
    """Roll shift tuple broadcasts scalar."""
    assert _normalise_roll_shift_tuple(2, 3) == (2, 2, 2)


def test_roll_shift_tuple_accepts_matching_sequence() -> None:
    """Roll shift tuple accepts matching sequence."""
    assert _normalise_roll_shift_tuple((1, -1), 2) == (1, -1)


def test_roll_shift_tuple_rejects_length_mismatch() -> None:
    """Roll shift tuple rejects length mismatch."""
    with pytest.raises(ValueError, match="axis lengths must match"):
        _normalise_roll_shift_tuple((1, 2), 3)


def test_roll_shift_tuple_rejects_non_iterable() -> None:
    """Roll shift tuple rejects non iterable."""
    with pytest.raises(ValueError, match="static integers"):
        _normalise_roll_shift_tuple(object(), 3)


# ----------------------------------------------------------------------- rot90


def test_rot90_k_accepts_integer() -> None:
    """Rot90 k accepts integer."""
    assert _normalise_rot90_k(3) == 3


def test_rot90_k_rejects_bool() -> None:
    """Rot90 k rejects bool."""
    with pytest.raises(ValueError, match="static integer"):
        _normalise_rot90_k(False)


def test_rot90_axes_accepts_two_axes() -> None:
    """Rot90 axes accepts two axes."""
    assert _normalise_rot90_axes((0, 1), rank=3) == (0, 1)


def test_rot90_axes_rejects_wrong_count() -> None:
    """Rot90 axes rejects wrong count."""
    with pytest.raises(ValueError, match="exactly two axes"):
        _normalise_rot90_axes((0, 1, 2), rank=3)


# ----------------------------------------------------------------------- sort


def test_sort_axis_wraps() -> None:
    """Sort axis wraps."""
    assert _normalise_sort_axis(-1, 3) == 2


def test_sort_axis_rejects_bool() -> None:
    """Sort axis rejects bool."""
    with pytest.raises(ValueError, match="static integer or None"):
        _normalise_sort_axis(True, 3)


def test_sort_axis_rejects_out_of_bounds() -> None:
    """Sort axis rejects out of bounds."""
    with pytest.raises(ValueError, match="out of bounds"):
        _normalise_sort_axis(5, 3)


# ------------------------------------------------------------------ broadcast


def test_broadcast_shape_combines_compatible_shapes() -> None:
    """Broadcast shape combines compatible shapes."""
    assert _broadcast_shape((1, 3), (2, 1)) == (2, 3)


def test_broadcast_shape_rejects_incompatible() -> None:
    """Broadcast shape rejects incompatible."""
    with pytest.raises(ValueError, match="NumPy broadcasting rules"):
        _broadcast_shape((2, 3), (4, 5))


# ----------------------------------------------------------------- trapezoid


def test_trapezoid_axis_normalises() -> None:
    """Trapezoid axis normalises."""
    assert _normalise_trapezoid_axis(-1, 3) == 2


def test_trapezoid_axis_rejects_non_integer() -> None:
    """Trapezoid axis rejects non integer."""
    with pytest.raises(ValueError, match="must be a static integer"):
        _normalise_trapezoid_axis(1.0, 3)


def test_trapezoid_axis_rejects_out_of_bounds() -> None:
    """Trapezoid axis rejects out of bounds."""
    with pytest.raises(ValueError, match="np.trapezoid axis out of bounds"):
        _normalise_trapezoid_axis(5, 3)


def test_trapezoid_axis_rejects_scalar_rank() -> None:
    """Trapezoid axis rejects scalar rank."""
    with pytest.raises(ValueError, match="cannot map over a scalar"):
        _normalise_trapezoid_axis(0, 0)
