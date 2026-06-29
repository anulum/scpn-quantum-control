# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for SDRE feedback control of Kuramoto networks
"""Module-specific tests for :mod:`kuramoto_sdre_control`.

The contracts: the SDRE feedback drives a perturbed network onto its target configuration; the gain
reduces exactly to the LQR at the target (where the state-dependent coefficient is the network
Jacobian); the closed loop is Hurwitz at the target; the control at the target is the pure
feed-forward and holds the equilibrium; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from scpn_quantum_control.accel.kuramoto_network_control import ControlledNetworkTrajectory
from scpn_quantum_control.accel.kuramoto_sdre_control import (
    integrate_sdre_controlled_kuramoto,
    kuramoto_sdre_gain,
    sdre_control_input,
)
from scpn_quantum_control.accel.networked_kuramoto import (
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)

_N = 8


def _problem(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(_N, _N))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "omega": rng.standard_normal(_N) * 0.3,
        "coupling": coupling,
        "target": rng.uniform(0.0, 2.0 * np.pi, size=_N),
        "rng": rng,
    }


def test_feedback_drives_the_network_onto_the_target() -> None:
    problem = _problem()
    start = problem["target"] + problem["rng"].standard_normal(_N) * 0.8
    trajectory = integrate_sdre_controlled_kuramoto(
        start,
        problem["target"],
        problem["omega"],
        problem["coupling"],
        0.02,
        1500,
        state_cost=1.0,
        control_cost=1.0,
    )
    assert isinstance(trajectory, ControlledNetworkTrajectory)
    assert np.max(np.abs(trajectory.terminal_phases - problem["target"])) < 1e-6


def test_gain_reduces_to_lqr_at_the_target() -> None:
    problem = _problem(1)
    gain = kuramoto_sdre_gain(
        problem["target"],
        problem["target"],
        problem["coupling"],
        state_cost=2.0,
        control_cost=0.5,
    )
    jacobian = networked_kuramoto_jacobian(problem["target"], problem["coupling"])
    actuation = np.eye(_N)
    riccati = solve_continuous_are(jacobian, actuation, 2.0 * actuation, 0.5 * actuation)
    assert gain == pytest.approx(riccati / 0.5, abs=1e-9)


def test_closed_loop_is_hurwitz_at_the_target() -> None:
    problem = _problem(2)
    gain = kuramoto_sdre_gain(
        problem["target"],
        problem["target"],
        problem["coupling"],
        state_cost=1.0,
        control_cost=1.0,
    )
    jacobian = networked_kuramoto_jacobian(problem["target"], problem["coupling"])
    closed_loop = jacobian - gain  # A - B G with B = I
    assert np.max(np.linalg.eigvals(closed_loop).real) < 0.0


def test_control_at_the_target_is_the_feed_forward_and_holds() -> None:
    problem = _problem(3)
    control = sdre_control_input(
        problem["target"],
        problem["target"],
        problem["omega"],
        problem["coupling"],
        state_cost=1.0,
        control_cost=1.0,
    )
    feed_forward = -(
        problem["omega"] + networked_kuramoto_force(problem["target"], problem["coupling"])
    )
    assert control == pytest.approx(feed_forward, abs=1e-9)
    # starting exactly at the target, the network holds there
    held = integrate_sdre_controlled_kuramoto(
        problem["target"],
        problem["target"],
        problem["omega"],
        problem["coupling"],
        0.02,
        200,
        state_cost=1.0,
        control_cost=1.0,
    )
    assert held.terminal_phases == pytest.approx(problem["target"], abs=1e-9)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("gain", {"phases": np.zeros(1)}, "phases must be a one-dimensional"),
        ("gain", {"target_phases": np.zeros(_N + 1)}, "target_phases must have shape"),
        ("gain", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("gain", {"phases": np.full(_N, np.nan)}, "must be finite"),
        ("gain", {"state_cost": 0.0}, "state_cost must be positive"),
        ("gain", {"control_cost": 0.0}, "control_cost must be positive"),
        ("control", {"omega": np.zeros(_N + 1)}, "omega must have shape"),
        ("control", {"omega": np.full(_N, np.inf)}, "omega must be finite"),
        ("integrate", {"omega": np.zeros(_N + 1)}, "omega must have shape"),
        ("integrate", {"omega": np.full(_N, np.inf)}, "omega must be finite"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"dt": np.inf}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    problem = _problem()
    base: dict[str, Any] = {
        "phases": problem["target"],
        "target_phases": problem["target"],
        "omega": problem["omega"],
        "coupling": problem["coupling"],
        "state_cost": 1.0,
        "control_cost": 1.0,
        "dt": 0.02,
        "n_steps": 10,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        if call == "gain":
            kuramoto_sdre_gain(
                base["phases"],
                base["target_phases"],
                base["coupling"],
                state_cost=base["state_cost"],
                control_cost=base["control_cost"],
            )
        elif call == "control":
            sdre_control_input(
                base["phases"],
                base["target_phases"],
                base["omega"],
                base["coupling"],
                state_cost=base["state_cost"],
                control_cost=base["control_cost"],
            )
        else:
            integrate_sdre_controlled_kuramoto(
                base["phases"],
                base["target_phases"],
                base["omega"],
                base["coupling"],
                base["dt"],
                base["n_steps"],
                state_cost=base["state_cost"],
                control_cost=base["control_cost"],
            )
