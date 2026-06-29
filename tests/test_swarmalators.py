# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for swarmalators (space + phase oscillators)
"""Module-specific tests for :mod:`swarmalators`.

The contracts: the canonical model settles into its named states — static synchrony (phase coherence
→ 1), the static phase wave (a rainbow order parameter → 1 while the ordinary coherence stays low,
the hallmark a phase-only model cannot produce) and static asynchrony (both ≈ 0); the rainbow order
parameters take their exact values on constructed configurations; the field and trajectory containers
behave; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import kuramoto
from scpn_quantum_control.accel.swarmalators import (
    SwarmalatorTrajectory,
    integrate_swarmalators,
    swarmalator_field,
    swarmalator_order_parameters,
)

_N = 60
_DT = 0.1


def _initial(seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(_N, 2))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=_N)
    return positions, phases


def test_static_synchrony_state() -> None:
    positions, phases = _initial(0)
    trajectory = integrate_swarmalators(positions, phases, 0.1, 1.0, _DT, 3000)
    assert isinstance(trajectory, SwarmalatorTrajectory)
    order = swarmalator_order_parameters(trajectory.terminal_positions, trajectory.terminal_phases)
    # strong space-weighted phase coupling synchronises the phases
    assert order.phase_coherence > 0.99


def test_static_phase_wave_state() -> None:
    positions, phases = _initial(0)
    # J > 0, K = 0: phase locks to spatial angle — a rainbow order parameter saturates while the
    # ordinary phase coherence stays low (a state with no phase-only analogue)
    trajectory = integrate_swarmalators(positions, phases, 1.0, 0.0, _DT, 4000)
    order = swarmalator_order_parameters(trajectory.terminal_positions, trajectory.terminal_phases)
    assert max(order.s_plus, order.s_minus) > 0.9
    assert order.phase_coherence < 0.3


def test_static_asynchrony_state() -> None:
    positions, phases = _initial(0)
    trajectory = integrate_swarmalators(positions, phases, 0.1, -1.0, _DT, 3000)
    order = swarmalator_order_parameters(trajectory.terminal_positions, trajectory.terminal_phases)
    assert order.phase_coherence < 0.2
    assert max(order.s_plus, order.s_minus) < 0.2


def test_rainbow_order_parameters_on_constructed_states() -> None:
    spatial_angle = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False, dtype=np.float64)
    positions = np.column_stack([np.cos(spatial_angle), np.sin(spatial_angle)])
    # phase locked to the spatial angle (θ = φ) → S_- = 1, S_+ ≈ 0, coherence ≈ 0
    locked = swarmalator_order_parameters(positions, spatial_angle)
    assert locked.s_minus == pytest.approx(1.0)
    assert locked.s_plus == pytest.approx(0.0, abs=1e-9)
    assert locked.phase_coherence == pytest.approx(0.0, abs=1e-9)
    # uniform phase → ordinary coherence 1
    synchronous = swarmalator_order_parameters(positions, np.full(_N, 0.7))
    assert synchronous.phase_coherence == pytest.approx(1.0)


def test_field_and_trajectory_shapes() -> None:
    positions, phases = _initial(1)
    position_velocity, phase_velocity = swarmalator_field(positions, phases, 1.0, 1.0)
    assert position_velocity.shape == (_N, 2)
    assert phase_velocity.shape == (_N,)
    assert np.all(np.isfinite(position_velocity)) and np.all(np.isfinite(phase_velocity))
    trajectory = integrate_swarmalators(positions, phases, 0.5, 0.5, _DT, 10)
    assert trajectory.positions.shape == (11, _N, 2)
    assert trajectory.phases.shape == (11, _N)
    assert trajectory.times[-1] == pytest.approx(10 * _DT)
    assert trajectory.terminal_positions == pytest.approx(trajectory.positions[-1])


def test_field_is_translation_equivariant_and_rotation_covariant() -> None:
    positions, phases = _initial(2)
    base_position_velocity, base_phase_velocity = swarmalator_field(positions, phases, 0.7, -0.4)

    translated_position_velocity, translated_phase_velocity = swarmalator_field(
        positions + np.array([10.0, -3.0]), phases, 0.7, -0.4
    )
    assert translated_position_velocity == pytest.approx(base_position_velocity)
    assert translated_phase_velocity == pytest.approx(base_phase_velocity)

    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    rotated_positions = positions @ rotation.T
    rotated_position_velocity, rotated_phase_velocity = swarmalator_field(
        rotated_positions, phases, 0.7, -0.4
    )
    assert rotated_position_velocity == pytest.approx(base_position_velocity @ rotation.T)
    assert rotated_phase_velocity == pytest.approx(base_phase_velocity)


def test_public_kuramoto_facade_exports_swarmalators() -> None:
    positions, phases = _initial(3)
    direct_position_velocity, direct_phase_velocity = swarmalator_field(
        positions, phases, 0.2, 0.3
    )
    facade_position_velocity, facade_phase_velocity = kuramoto.swarmalator_field(
        positions, phases, 0.2, 0.3
    )
    assert facade_position_velocity == pytest.approx(direct_position_velocity)
    assert facade_phase_velocity == pytest.approx(direct_phase_velocity)

    trajectory = kuramoto.integrate_swarmalators(positions, phases, 0.2, 0.3, 0.01, 2)
    assert isinstance(trajectory, kuramoto.SwarmalatorTrajectory)
    order = kuramoto.swarmalator_order_parameters(
        trajectory.terminal_positions, trajectory.terminal_phases
    )
    assert isinstance(order, kuramoto.SwarmalatorOrderParameters)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("field", {"positions": np.zeros((1, 2))}, "positions must be an"),
        ("field", {"positions": np.zeros((5, 3))}, "positions must be an"),
        ("field", {"phases": np.zeros(4)}, "phases must have shape"),
        (
            "field",
            {
                "positions": np.array(
                    [[0.0, 0.0], [np.nan, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                )
            },
            "positions must be finite",
        ),
        ("field", {"phases": np.full(5, np.nan)}, "phases must be finite"),
        (
            "field",
            {"positions": np.vstack([np.zeros((2, 2)), np.ones((3, 2))])},
            "positions must be pairwise distinct",
        ),
        ("field", {"coupling_phase": np.inf}, "coupling_phase must be finite"),
        ("field", {"coupling_space": np.inf}, "coupling_space must be finite"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"dt": np.inf}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
        ("order", {"positions": np.zeros((3, 3))}, "positions must be an"),
        ("order", {"phases": np.zeros(3)}, "phases must have shape"),
        (
            "order",
            {
                "positions": np.array(
                    [[np.inf, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]
                )
            },
            "positions must be finite",
        ),
        ("order", {"phases": np.array([np.nan, 0.0, 0.0, 0.0, 0.0])}, "phases must be finite"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    positions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(5, 2))
    phases = np.zeros(5)
    with pytest.raises(ValueError, match=message):
        if call == "field":
            args: dict[str, Any] = {
                "positions": positions,
                "phases": phases,
                "coupling_phase": 1.0,
                "coupling_space": 1.0,
            }
            args.update(kwargs)
            swarmalator_field(
                args["positions"], args["phases"], args["coupling_phase"], args["coupling_space"]
            )
        elif call == "integrate":
            args = {"dt": 0.1, "n_steps": 10}
            args.update(kwargs)
            integrate_swarmalators(positions, phases, 1.0, 1.0, args["dt"], args["n_steps"])
        else:
            args = {"positions": positions, "phases": phases}
            args.update(kwargs)
            swarmalator_order_parameters(args["positions"], args["phases"])
