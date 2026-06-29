# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for reduced-order optimal control of the collective phase
"""Module-specific tests for :mod:`kuramoto_collective_control`.

The contracts: with zero forcing the controlled flow reproduces the shipped autonomous Ott–Antonsen
reduction exactly; the discrete-adjoint gradient of the terminal control objective matches finite
differences to machine precision (the property that makes it control-grade); gradient descent on the
forcing genuinely steers the collective order parameter towards the target with a monotone cost; and
the trajectory container and input contract behave.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.kuramoto_collective_control import (
    ForcedCollectiveTrajectory,
    collective_control_value_and_grad,
    integrate_forced_collective,
    optimise_collective_forcing,
)
from scpn_quantum_control.accel.kuramoto_ott_antonsen import ott_antonsen_trajectory

_K = 3.0
_DELTA = 0.5
_DT = 0.05
_CENTRE = 0.4
_Z0 = 0.6 + 0.1j


def test_zero_forcing_reproduces_autonomous_ott_antonsen() -> None:
    n_steps = 40
    trajectory = integrate_forced_collective(
        _Z0, np.zeros(n_steps, dtype=np.complex128), _K, _DELTA, _DT, centre=_CENTRE
    )
    reference = ott_antonsen_trajectory(_Z0, _K, _DELTA, _DT, n_steps, centre=_CENTRE)
    assert trajectory.order_parameter == pytest.approx(reference, abs=1e-12)


def test_adjoint_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    n_steps = 30
    forcing: NDArray[np.complex128] = np.asarray(
        0.2 * (rng.standard_normal(n_steps) + 1j * rng.standard_normal(n_steps)),
        dtype=np.complex128,
    )
    grads = collective_control_value_and_grad(
        _Z0, forcing, _K, _DELTA, _DT, target=0j, control_weight=0.1, centre=_CENTRE
    )

    eps = 1e-6
    finite = np.zeros(n_steps, dtype=np.complex128)

    def cost(series: NDArray[np.complex128]) -> float:
        return collective_control_value_and_grad(
            _Z0, series, _K, _DELTA, _DT, target=0j, control_weight=0.1, centre=_CENTRE
        ).cost

    for index in range(n_steps):
        for unit in (1.0 + 0j, 1j):
            plus = forcing.copy()
            minus = forcing.copy()
            plus[index] += eps * unit
            minus[index] -= eps * unit
            derivative = (cost(plus) - cost(minus)) / (2.0 * eps)
            finite[index] += derivative if unit == 1.0 else 1j * derivative
    assert grads.forcing_gradient == pytest.approx(finite, abs=1e-7)


def test_gradient_descent_desynchronises_with_monotone_cost() -> None:
    n_steps = 30
    forcing, history = optimise_collective_forcing(
        _Z0,
        _K,
        _DELTA,
        _DT,
        n_steps,
        target=0j,
        control_weight=1e-3,
        learning_rate=0.5,
        n_iterations=200,
        centre=_CENTRE,
    )
    uncontrolled = integrate_forced_collective(
        _Z0, np.zeros(n_steps, dtype=np.complex128), _K, _DELTA, _DT, centre=_CENTRE
    )
    controlled = integrate_forced_collective(_Z0, forcing, _K, _DELTA, _DT, centre=_CENTRE)
    # the optimised forcing drives |z(T)| below the uncontrolled value (towards the target 0)
    assert abs(controlled.terminal_order_parameter) < abs(uncontrolled.terminal_order_parameter)
    # the cost decreases monotonically and strictly overall
    assert np.all(np.diff(history) <= 1e-9)
    assert history[-1] < history[0]


def test_trajectory_container() -> None:
    n_steps = 12
    trajectory = integrate_forced_collective(
        _Z0, np.full(n_steps, 0.05 + 0.0j), _K, _DELTA, _DT, centre=_CENTRE
    )
    assert isinstance(trajectory, ForcedCollectiveTrajectory)
    assert trajectory.order_parameter.shape == (n_steps + 1,)
    assert trajectory.times.shape == (n_steps + 1,)
    assert trajectory.times[-1] == pytest.approx(n_steps * _DT)
    assert trajectory.terminal_order_parameter == trajectory.order_parameter[-1]


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("integrate", {"z0": 1.5 + 0j}, r"\|z0\| must not exceed 1"),
        ("integrate", {"forcing": np.zeros((2, 2), dtype=np.complex128)}, "forcing must be"),
        ("integrate", {"coupling": 0.0}, "coupling must be positive"),
        ("integrate", {"half_width": 0.0}, "half_width must be positive"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("grad", {"control_weight": -1.0}, "control_weight must be non-negative"),
        ("optimise", {"n_steps": 0}, "n_steps must be positive"),
        ("optimise", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("optimise", {"n_iterations": 0}, "n_iterations must be positive"),
        ("optimise", {"initial_forcing": np.zeros(3, dtype=np.complex128)}, "initial_forcing"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    base_integrate: dict[str, Any] = {
        "z0": _Z0,
        "forcing": np.zeros(5, dtype=np.complex128),
        "coupling": _K,
        "half_width": _DELTA,
        "dt": _DT,
    }
    with pytest.raises(ValueError, match=message):
        if call == "integrate":
            base_integrate.update(kwargs)
            integrate_forced_collective(
                base_integrate["z0"],
                base_integrate["forcing"],
                base_integrate["coupling"],
                base_integrate["half_width"],
                base_integrate["dt"],
            )
        elif call == "grad":
            collective_control_value_and_grad(
                _Z0,
                np.zeros(5, dtype=np.complex128),
                _K,
                _DELTA,
                _DT,
                target=0j,
                control_weight=kwargs["control_weight"],
            )
        else:
            optimise_kwargs: dict[str, Any] = {
                "n_steps": 5,
                "learning_rate": 0.5,
                "n_iterations": 10,
            }
            optimise_kwargs.update(kwargs)
            optimise_collective_forcing(
                _Z0,
                _K,
                _DELTA,
                _DT,
                optimise_kwargs["n_steps"],
                target=0j,
                control_weight=0.1,
                learning_rate=optimise_kwargs["learning_rate"],
                n_iterations=optimise_kwargs["n_iterations"],
                initial_forcing=optimise_kwargs.get("initial_forcing"),
            )
