# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole program AD finite difference gradient check tests
# scpn-quantum-control -- Whole-program AD finite-difference gradient checks
"""Independent finite-difference cross-checks of whole-program forward AD.

``whole_program_grad`` differentiates an executed Python/NumPy program by
operator-intercepted exact forward AD: parameters enter as derivative-carrying
trace values and the program runs unchanged, so loops, data-dependent branches,
list mutation, aliasing, and supported scalar ufuncs all propagate exact
tangents. The runtime tests in ``test_whole_program_ad_runtime`` assert the
gradient against hand-derived closed-form expressions (for example
``cos(0.25)``); a hand-derivation slip can leave the interception rule and the
expected expression wrong the same way.

These tests add an independent anchor: the identical objective is run once
through ``whole_program_grad`` (forward AD over trace values) and once through
``finite_difference_gradient`` (central differences over plain floats), and the
two gradients must agree. Because whole-program AD is exact, the agreement is to
finite-difference rounding (~1e-10), not merely to a loose tolerance. Each
objective is evaluated at an interior point so no data-dependent branch or
``maximum``/``minimum`` tie flips under the finite-difference perturbation, and
every ``log``/``sqrt``/``arcsin``/``arccos`` argument stays inside its domain --
keeping the whole perturbation stencil inside one differentiable program branch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_finite_difference import finite_difference_gradient
from scpn_quantum_control.whole_program_ad_api import whole_program_grad


def _assert_whole_program_grad_matches_finite_difference(
    objective: Any,
    values: NDArray[np.float64],
    *,
    atol: float = 1e-6,
    step: float = 1e-6,
) -> None:
    """Assert forward-AD and finite-difference gradients of ``objective`` agree."""

    values = np.asarray(values, dtype=np.float64)
    analytic = whole_program_grad(objective, values, trace=False)
    numeric = finite_difference_gradient(objective, values, step=step)
    np.testing.assert_allclose(analytic, numeric, rtol=0.0, atol=atol)


# --------------------------------------------------------------------------- #
# Objective programs -- one executed Python/NumPy program per differentiable
# construct. Written to run unchanged over trace values (forward AD) and over
# plain floats (finite differences); the shared source is what makes the two
# gradients comparable.
# --------------------------------------------------------------------------- #
def _smooth_trig_poly(values: Any) -> object:
    """Straight-line arithmetic and scalar ufuncs, no control flow."""

    return np.sin(values[0]) * values[1] + np.exp(values[2]) - values[0] * values[2] ** 2


def _loop_sum_of_squares(values: Any) -> object:
    """``for`` loop accumulating a smooth reduction."""

    total = values[0] * 0.0
    for value in values:
        total = total + value * value
    return total


def _loop_index_sin(values: Any) -> object:
    """``for`` loop mixing the enumerate index with a ufunc."""

    total = values[0] * 0.0
    for index, value in enumerate(values):
        total = total + index * np.sin(value)
    return total


def _branch_program(values: Any) -> object:
    """Data-dependent branch; evaluated away from the ``0`` boundary."""

    if values[0] > 0.0:
        return np.sin(values[0]) + values[1] ** 2
    return np.cos(values[0]) - values[1] ** 2


def _alias_list_mutation(values: Any) -> object:
    """List aliasing and in-place mutation feeding the accumulator."""

    history = [values[0]]
    alias = history
    total = values[0] * 0.0
    for item in range(3):
        alias.append(values[1] * item)
        total = total + history[item]
    return np.sin(values[0]) + total


def _rational_log_reciprocal(values: Any) -> object:
    """Division, ``log`` and ``reciprocal`` in one expression."""

    return values[0] / (1.0 + values[1] * values[1]) + np.log(values[2]) + np.reciprocal(values[3])


def _running_product(values: Any) -> object:
    """``for`` loop accumulating a running product."""

    product = values[0] * 0.0 + 1.0
    for value in values:
        product = product * value
    return product


def _maximum_tanh(values: Any) -> object:
    """``maximum`` with a clear winner plus ``tanh``."""

    return np.maximum(values[0], values[1]) + np.tanh(values[2])


def _nested_ufuncs(values: Any) -> object:
    """Nested composition of ``square``/``exp``/``log1p``/``sqrt``."""

    return np.log1p(np.square(values[0]) + np.exp(values[1])) + np.sqrt(values[2])


def _inverse_trig(values: Any) -> object:
    """``arcsin``/``arccos`` evaluated inside their domain."""

    return np.arcsin(values[0]) * np.arccos(values[1]) + values[2] * values[2]


def _while_loop_cos(values: Any) -> object:
    """``while`` loop with an integer counter driving indexing and a ufunc."""

    total = values[0] * 0.0
    count = 0
    while count < 3:
        total = total + values[count] * np.cos(values[count])
        count = count + 1
    return total


def _variable_power(values: Any) -> object:
    """Variable exponent with a positive base plus a constant-exponent power."""

    return values[0] ** values[1] + values[2] ** 2.0


_WHOLE_PROGRAM_CASES: dict[str, tuple[Any, NDArray[np.float64]]] = {
    "smooth_trig_poly": (_smooth_trig_poly, np.array([0.3, -0.7, 0.4])),
    "loop_sum_of_squares": (_loop_sum_of_squares, np.array([0.5, -0.6, 1.1, 0.2])),
    "loop_index_sin": (_loop_index_sin, np.array([0.25, 0.75, -0.4])),
    "branch_positive": (_branch_program, np.array([0.5, 0.3])),
    "branch_negative": (_branch_program, np.array([-0.5, 0.3])),
    "alias_list_mutation": (_alias_list_mutation, np.array([0.5, 0.25])),
    "rational_log_reciprocal": (_rational_log_reciprocal, np.array([0.7, 0.4, 1.6, 0.9])),
    "running_product": (_running_product, np.array([0.6, 0.8, 1.2, 0.7])),
    "maximum_tanh": (_maximum_tanh, np.array([0.9, 0.2, 0.4])),
    "nested_ufuncs": (_nested_ufuncs, np.array([0.5, -0.3, 1.4])),
    "inverse_trig": (_inverse_trig, np.array([0.3, -0.2, 0.5])),
    "while_loop_cos": (_while_loop_cos, np.array([0.4, 0.7, -0.5])),
    "variable_power": (_variable_power, np.array([1.3, 0.7, 0.5])),
}


@pytest.mark.parametrize("name", sorted(_WHOLE_PROGRAM_CASES))
def test_whole_program_grad_matches_finite_difference(name: str) -> None:
    objective, values = _WHOLE_PROGRAM_CASES[name]
    _assert_whole_program_grad_matches_finite_difference(objective, values)
