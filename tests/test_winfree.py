# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Winfree / unified Kuramoto–Winfree model
"""Module-specific tests for :mod:`winfree`.

The contracts: the interpolation endpoints are exact — ``q = 0`` reproduces the all-to-all Kuramoto
field and ``q = 1`` reproduces the analytic Winfree mean-field; the field is affine in the
interpolation; the dense mean-field Jacobian matches finite differences across the family; the
Winfree limit reaches oscillator death (the symmetry-breaking regime a difference-coupled model
cannot); and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force
from scpn_quantum_control.accel.winfree import (
    WinfreeTrajectory,
    integrate_winfree,
    winfree_field,
    winfree_jacobian,
)

_N = 8
_EPS = 2.0


def _problem(seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=_N), rng.standard_normal(_N) * 0.3


def test_kuramoto_endpoint_q_zero() -> None:
    phases, omega = _problem()
    coupling = _EPS / _N * np.ones((_N, _N))
    np.fill_diagonal(coupling, 0.0)
    expected = omega + networked_kuramoto_force(phases, coupling)
    assert winfree_field(phases, omega, _EPS, 0.0) == pytest.approx(expected, abs=1e-12)


def test_winfree_endpoint_q_one() -> None:
    phases, omega = _problem()
    analytic = omega - _EPS * np.sin(phases) * (1.0 + np.mean(np.cos(phases)))
    assert winfree_field(phases, omega, _EPS, 1.0) == pytest.approx(analytic, abs=1e-12)


def test_field_is_affine_in_the_interpolation() -> None:
    phases, omega = _problem(1)
    endpoint_kuramoto = winfree_field(phases, omega, _EPS, 0.0)
    endpoint_winfree = winfree_field(phases, omega, _EPS, 1.0)
    for q in (0.25, 0.5, 0.8):
        blended = (1.0 - q) * endpoint_kuramoto + q * endpoint_winfree
        assert winfree_field(phases, omega, _EPS, q) == pytest.approx(blended, abs=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.6, 1.0])
def test_jacobian_matches_finite_differences(q: float) -> None:
    phases, omega = _problem(2)
    jacobian = winfree_jacobian(phases, omega, _EPS, q)
    eps = 1e-6
    finite = np.zeros((_N, _N), dtype=np.float64)
    for column in range(_N):
        plus = phases.copy()
        minus = phases.copy()
        plus[column] += eps
        minus[column] -= eps
        finite[:, column] = (
            winfree_field(plus, omega, _EPS, q) - winfree_field(minus, omega, _EPS, q)
        ) / (2.0 * eps)
    assert jacobian == pytest.approx(finite, abs=1e-7)


def test_winfree_limit_reaches_oscillation_death() -> None:
    phases = np.random.default_rng(3).uniform(-1.0, 1.0, size=_N)
    trajectory = integrate_winfree(phases, np.zeros(_N), 3.0, 1.0, 0.01, 15000)
    assert isinstance(trajectory, WinfreeTrajectory)
    final = trajectory.terminal_phases
    velocity = winfree_field(final, np.zeros(_N), 3.0, 1.0)
    # oscillations have ceased at the synchronised Winfree fixed point
    assert np.max(np.abs(velocity)) < 1e-6
    assert np.mean(np.cos(final)) > 0.99


def test_trajectory_container() -> None:
    phases, omega = _problem(4)
    trajectory = integrate_winfree(phases, omega, 1.0, 0.5, 0.05, 12)
    assert trajectory.phases.shape == (13, _N)
    assert trajectory.times.shape == (13,)
    assert trajectory.times[-1] == pytest.approx(12 * 0.05)
    assert trajectory.terminal_phases == pytest.approx(trajectory.phases[-1])


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("field", {"phases": np.zeros(1)}, "phases must be a one-dimensional"),
        ("field", {"phases": np.zeros((2, 2))}, "phases must be a one-dimensional"),
        ("field", {"omega": np.zeros(4)}, "omega must have shape"),
        ("field", {"phases": np.full(_N, np.nan)}, "phases must be finite"),
        ("field", {"omega": np.full(_N, np.inf)}, "omega must be finite"),
        ("field", {"coupling": np.inf}, "coupling must be finite"),
        ("field", {"interpolation": -0.1}, r"interpolation must be in \[0, 1\]"),
        ("field", {"interpolation": 1.5}, r"interpolation must be in \[0, 1\]"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"dt": np.inf}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    phases, omega = _problem()
    with pytest.raises(ValueError, match=message):
        if call == "field":
            args: dict[str, Any] = {
                "phases": phases,
                "omega": omega,
                "coupling": _EPS,
                "interpolation": 0.5,
            }
            args.update(kwargs)
            winfree_field(args["phases"], args["omega"], args["coupling"], args["interpolation"])
        else:
            args = {"dt": 0.05, "n_steps": 10}
            args.update(kwargs)
            integrate_winfree(phases, omega, _EPS, 0.5, args["dt"], args["n_steps"])
