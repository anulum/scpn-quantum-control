# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — frozen effect dispatch contracts
"""Exercise frozen native identities through public derivative capture and replay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control import TraceADArray, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.execution_reservations import active_reserved_bytes


@pytest.mark.parametrize("route", ["dot", "outer", "stack", "concatenate"])
def test_native_dispatch_output_slots_refuse_caller_owned_storage(route: str) -> None:
    """Different frozen native output slots refuse before writing an external array.

    Parameters
    ----------
    route
        Native function with an explicitly positioned output operand.

    """
    if route == "dot":
        state = np.array(7.0)
        invoke = cast(Callable[..., object], np.dot)

        def objective(values: TraceADArray) -> object:
            invoke(np.ones(2), np.ones(2), state)
            return values[0]

    elif route == "outer":
        state = np.array([[7.0]])
        invoke = cast(Callable[..., object], np.outer)

        def objective(values: TraceADArray) -> object:
            invoke(np.ones(1), np.ones(1), state)
            return values[0]

    elif route == "stack":
        state = np.array([[7.0], [7.0]])
        invoke = cast(Callable[..., object], np.stack)

        def objective(values: TraceADArray) -> object:
            invoke((np.ones(1), np.ones(1)), 0, state)
            return values[0]

    else:
        state = np.array([7.0, 7.0])
        invoke = cast(Callable[..., object], np.concatenate)

        def objective(values: TraceADArray) -> object:
            invoke((np.ones(1), np.ones(1)), 0, state)
            return values[0]

    original = state.copy()
    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="captured_mutation.*line="):
        whole_program_value_and_grad(objective, [3.0], trace=False)
    np.testing.assert_array_equal(state, original)
    assert active_reserved_bytes() == baseline


def test_frozen_native_dot_output_preserves_owned_storage_and_replay() -> None:
    """A native dot writes its owned scalar output without detaching active dependence."""

    def objective(values: TraceADArray) -> object:
        output = np.zeros(())
        np.dot(np.ones(2), np.ones(2), out=output)
        return values[0] * output.item()

    result = whole_program_value_and_grad(objective, [3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [2.0])


def test_frozen_unary_identity_preserves_nonlinear_derivative_and_replay() -> None:
    """An admitted NumPy identity still reaches the real nonlinear primitive registry."""
    sine = cast(Callable[[object], object], np.sin)

    def objective(values: TraceADArray) -> object:
        return sine(values[0])

    result = whole_program_value_and_grad(objective, [0.5], trace=False)
    assert result.value == pytest.approx(float(np.sin(0.5)), abs=1e-14)
    np.testing.assert_allclose(result.gradient, [np.cos(0.5)], rtol=1e-14)
    np.testing.assert_allclose(program_adjoint_replay_gradient(result), [np.cos(0.5)], rtol=1e-14)


@pytest.mark.parametrize("angle", [0.3, -0.6])
def test_frozen_identity_allocator_is_admitted_with_exact_value_and_derivative(
    angle: float,
) -> None:
    """A constant identity matrix enters an objective as a passive operand.

    The objective is the leading entry of ``I - v v^T`` for the unit vector
    ``v = (sin t, 0, cos t)``; its value is ``cos(t)**2`` and its derivative
    ``-sin(2 t)``.

    Parameters
    ----------
    angle
        Angle ``t`` at which the value and the gradient are compared.

    """

    def objective(values: Any) -> object:
        theta = values[0]
        direction = np.stack((np.sin(theta), theta * 0.0, np.cos(theta)))
        return (np.eye(3) - np.outer(direction, direction))[0, 0]

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [angle], trace=False)
    assert result.value == pytest.approx(np.cos(angle) ** 2, abs=1e-14)
    np.testing.assert_allclose(result.gradient, [-np.sin(2.0 * angle)], rtol=0.0, atol=1e-14)
    assert active_reserved_bytes() == baseline
