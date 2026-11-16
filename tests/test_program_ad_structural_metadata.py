# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — structural metadata of traced arrays in whole-program AD
"""The shape, rank, size and length of a traced array are static metadata.

They are plain integers fixed by the input's structure, so an objective may size a
reshape or a loop with them and pass them through ``int()`` or ``round()``. An
integer conversion or a rounding of a traced value stays refused before execution.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control import whole_program_value_and_grad
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects

_INTEGER = "active integer shape or index conversion is unsupported"
_ROUNDING = "active rounding loses the derivative and is unsupported"


def _findings(objective: Callable[..., object]) -> tuple[tuple[str, str], ...]:
    """Return the semantic and detail of every effect finding, in source order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    return tuple(
        (finding.semantic, finding.detail) for finding in find_objective_effects(objective, tree)
    )


def test_a_side_derived_from_the_traced_size_reshapes_and_differentiates() -> None:
    """The side of a square matrix derived from the parameter count needs no static literal."""

    def determinant(values: Any) -> object:
        side = int(round(float(np.sqrt(values.size))))
        return np.linalg.det(np.reshape(values, (side, side)))

    def squares(values: Any) -> object:
        side = int(round(float(np.sqrt(values.size))))
        grid = values.reshape(side, side)
        return np.sum(grid * grid)

    assert _findings(determinant) == () and _findings(squares) == ()
    point = np.array([3.0, -2.0, 1.0, 4.0])
    result = whole_program_value_and_grad(determinant, point, trace=False)
    assert result.value == pytest.approx(14.0, abs=1e-12)
    np.testing.assert_allclose(result.gradient, [4.0, -1.0, 2.0, 3.0], atol=1e-12)
    grid = np.arange(1.0, 10.0)
    result = whole_program_value_and_grad(squares, grid, trace=False)
    assert result.value == pytest.approx(float(grid @ grid))
    np.testing.assert_allclose(result.gradient, 2.0 * grid)


def test_shape_rank_size_and_length_pass_through_integer_conversions() -> None:
    """Each metadata form, also of a derived array, is admitted where an integer is needed."""

    def by_shape(values: Any) -> object:
        return values[0] * int(values.shape[0])

    def by_rank(values: Any) -> object:
        return values[0] * int(values.ndim)

    def by_size(values: Any) -> object:
        return values[0] * int(values.size)

    def by_length(values: Any) -> object:
        return values[0] * int(len(values))

    def by_derived_size(values: Any) -> object:
        return values[0] * int(np.sqrt(values * values).size)

    def by_rounded_half(values: Any) -> object:
        return values[0] * round(values.size / 2) + values[1] * round(2.6)

    point = np.array([3.0, -2.0, 1.0, 4.0])
    expected = {
        by_shape: (12.0, 4.0),
        by_rank: (3.0, 1.0),
        by_size: (12.0, 4.0),
        by_length: (12.0, 4.0),
        by_derived_size: (12.0, 4.0),
    }
    for objective, (value, slope) in expected.items():
        assert _findings(objective) == ()
        result = whole_program_value_and_grad(objective, point, trace=False)
        assert result.value == value
        np.testing.assert_array_equal(result.gradient, [slope, 0.0, 0.0, 0.0])
    assert _findings(by_rounded_half) == ()
    result = whole_program_value_and_grad(by_rounded_half, point, trace=False)
    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, [2.0, 3.0, 0.0, 0.0])


def test_an_integer_conversion_or_rounding_of_a_traced_value_stays_refused() -> None:
    """A traced value, positional or by keyword, may not become an integer or be rounded."""

    def converted(values: Any) -> object:
        return values[0] * int(values[1])

    def converted_sum(values: Any) -> object:
        return values[0] * int(np.sum(values))

    def rounded(values: Any) -> object:
        return values[0] * round(values[1])

    def rounded_digits(values: Any) -> object:
        return values[0] * round(values[1], 2)

    def rounded_keyword(values: Any) -> object:
        return values[0] * round(number=values[1])

    refusals: tuple[tuple[Callable[..., object], str], ...] = (
        (converted, _INTEGER),
        (converted_sum, _INTEGER),
        (rounded, _ROUNDING),
        (rounded_digits, _ROUNDING),
        (rounded_keyword, _ROUNDING),
    )
    for objective, detail in refusals:
        assert _findings(objective) == (("dynamic_integer", detail),)
        with pytest.raises(ValueError, match=detail):
            whole_program_value_and_grad(objective, [3.0, -2.0], trace=False)
