# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for pseudo-arclength continuation past fold points
"""Module-specific tests for :mod:`kuramoto_pseudo_arclength`.

The defining contract is that the continuation traces a branch *through* a fold where
natural-parameter continuation stalls: on the textbook parabola ``x^2 + λ = 0`` and on the
canonical Kuramoto locking saddle-node ``K sin φ − Δω = 0`` the branch turns around in the
parameter and continues onto the second solution branch, staying on the residual to machine
precision. The branch container, the input contract and the corrector-failure path are also
exercised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.kuramoto_pseudo_arclength import (
    PseudoArclengthBranch,
    pseudo_arclength_continuation,
)


def _parabola_residual(state: NDArray[np.float64], parameter: float) -> NDArray[np.float64]:
    return np.array([state[0] ** 2 + parameter])


def _parabola_state_jacobian(state: NDArray[np.float64], parameter: float) -> NDArray[np.float64]:
    return np.array([[2.0 * state[0]]])


def _parabola_parameter_derivative(
    state: NDArray[np.float64], parameter: float
) -> NDArray[np.float64]:
    return np.array([1.0])


def test_traces_through_the_textbook_fold() -> None:
    branch = pseudo_arclength_continuation(
        _parabola_residual,
        _parabola_state_jacobian,
        _parabola_parameter_derivative,
        np.array([1.0]),
        -1.0,
        step=0.15,
        n_steps=40,
    )
    x = branch.states[:, 0]
    parameter = branch.parameters
    # the parameter rises to the fold at λ = 0 and turns back down
    assert parameter.max() == pytest.approx(0.0, abs=1e-2)
    assert parameter[-1] < parameter.max()
    # both branches (x > 0 and x < 0) are traced — natural-parameter continuation cannot do this
    assert (x > 0.1).any() and (x < -0.1).any()
    # the points stay on the solution branch G = 0
    assert float(np.max(np.abs(x**2 + parameter))) < 1e-8


def test_traces_through_the_kuramoto_locking_saddle_node() -> None:
    detuning = 1.0

    def residual(state: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
        return np.array([coupling * np.sin(state[0]) - detuning])

    def state_jacobian(state: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
        return np.array([[coupling * np.cos(state[0])]])

    def parameter_derivative(state: NDArray[np.float64], coupling: float) -> NDArray[np.float64]:
        return np.array([np.sin(state[0])])

    branch = pseudo_arclength_continuation(
        residual,
        state_jacobian,
        parameter_derivative,
        np.array([np.arcsin(detuning / 3.0)]),
        3.0,
        step=0.1,
        n_steps=60,
        parameter_direction=-1.0,
    )
    phi = branch.states[:, 0]
    coupling = branch.parameters
    # the locked branch folds at the saddle-node K = Δω; the coupling does not drop below it
    assert coupling.min() == pytest.approx(detuning, abs=1e-2)
    # the stable (φ < π/2) and unstable (φ > π/2) locked phases are both traced
    assert (phi < np.pi / 2).any() and (phi > np.pi / 2).any()
    assert float(np.max(np.abs(coupling * np.sin(phi) - detuning))) < 1e-8


def test_branch_container() -> None:
    branch = pseudo_arclength_continuation(
        _parabola_residual,
        _parabola_state_jacobian,
        _parabola_parameter_derivative,
        np.array([1.0]),
        -1.0,
        step=0.1,
        n_steps=10,
    )
    assert isinstance(branch, PseudoArclengthBranch)
    assert branch.states.shape == (11, 1)
    assert branch.parameters.shape == (11,)
    assert branch.n_points == 11
    assert branch.arclengths == pytest.approx(0.1 * np.arange(11))


def test_corrector_failure_raises() -> None:
    # one Newton iteration is not enough to correct a large step on the nonlinear parabola
    with pytest.raises(RuntimeError, match="corrector failed to converge"):
        pseudo_arclength_continuation(
            _parabola_residual,
            _parabola_state_jacobian,
            _parabola_parameter_derivative,
            np.array([1.0]),
            -1.0,
            step=2.0,
            n_steps=5,
            max_newton_iterations=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"initial_state": np.zeros((2, 2))}, "initial_state must be a non-empty"),
        ({"step": 0.0}, "step must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
        ({"parameter_direction": 2.0}, "parameter_direction must be"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "initial_state": np.array([1.0]),
        "initial_parameter": -1.0,
        "step": 0.1,
        "n_steps": 10,
        "parameter_direction": 1.0,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        pseudo_arclength_continuation(
            _parabola_residual,
            _parabola_state_jacobian,
            _parabola_parameter_derivative,
            call["initial_state"],
            call["initial_parameter"],
            step=call["step"],
            n_steps=call["n_steps"],
            parameter_direction=call["parameter_direction"],
        )
