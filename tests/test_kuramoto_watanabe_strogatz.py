# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Watanabe–Strogatz exact finite-N reduction
"""Module-specific tests for :mod:`kuramoto_watanabe_strogatz`.

The reduction is validated on its defining contracts: the reconstructed phases match a direct
N-oscillator integration of the identical-oscillator mean-field Kuramoto model, the SU(1,1)
Möbius invariant is conserved along the flow, the constants of motion are the initial points,
and in the thermodynamic limit the order parameter follows the identical-oscillator
Ott–Antonsen equation (cross-checked against the repository's :func:`ott_antonsen_field`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel import ott_antonsen_field
from scpn_quantum_control.accel.kuramoto_watanabe_strogatz import (
    WatanabeStrogatzTrajectory,
    integrate_watanabe_strogatz,
    watanabe_strogatz_constants,
    watanabe_strogatz_invariant,
    watanabe_strogatz_order_parameter,
    watanabe_strogatz_phases,
)

_OMEGA = 1.0
_COUPLING = 2.0
_DT = 0.01
_N_STEPS = 400


def _direct_identical_kuramoto(
    initial_phases: NDArray[np.float64], dt: float, n_steps: int
) -> NDArray[np.float64]:
    """Direct RK4 integration of θ̇_j = ω + Im(K Z e^{-iθ_j}) for identical oscillators."""
    theta = initial_phases.copy()

    def field(phase: NDArray[np.float64]) -> NDArray[np.float64]:
        order_parameter = np.mean(np.exp(1j * phase))
        return np.asarray(
            _OMEGA + (_COUPLING * order_parameter * np.exp(-1j * phase)).imag, dtype=np.float64
        )

    for _ in range(n_steps):
        k1 = field(theta)
        k2 = field(theta + 0.5 * dt * k1)
        k3 = field(theta + 0.5 * dt * k2)
        k4 = field(theta + dt * k3)
        theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return theta


def test_reconstruction_matches_direct_integration() -> None:
    rng = np.random.default_rng(1)
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=12)
    trajectory = integrate_watanabe_strogatz(
        initial_phases, omega=_OMEGA, coupling=_COUPLING, dt=_DT, n_steps=_N_STEPS
    )
    direct = _direct_identical_kuramoto(initial_phases, _DT, _N_STEPS)
    residual = np.angle(np.exp(1j * (trajectory.terminal_phases - direct)))
    assert float(np.max(np.abs(residual))) < 1e-7


def test_su11_invariant_is_conserved() -> None:
    rng = np.random.default_rng(2)
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=8)
    trajectory = integrate_watanabe_strogatz(
        initial_phases, omega=_OMEGA, coupling=_COUPLING, dt=_DT, n_steps=_N_STEPS
    )
    invariants = np.abs(trajectory.alpha) ** 2 - np.abs(trajectory.beta) ** 2
    assert np.allclose(invariants, 1.0, atol=1e-8)
    assert watanabe_strogatz_invariant(
        complex(trajectory.alpha[-1]), complex(trajectory.beta[-1])
    ) == pytest.approx(1.0, abs=1e-8)


def test_constants_are_the_initial_points() -> None:
    initial_phases = np.array([0.1, 1.3, -2.0, 2.5], dtype=np.float64)
    constants = watanabe_strogatz_constants(initial_phases)
    assert constants == pytest.approx(np.exp(1j * initial_phases))
    # the identity map reconstructs the initial phases
    reconstructed = watanabe_strogatz_phases(1.0 + 0.0j, 0.0 + 0.0j, constants)
    assert np.angle(np.exp(1j * (reconstructed - initial_phases))) == pytest.approx(
        np.zeros(4), abs=1e-12
    )


def test_order_parameter_helper_and_initial_value() -> None:
    initial_phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float64)
    constants = watanabe_strogatz_constants(initial_phases)
    # four symmetric points have a vanishing order parameter
    assert watanabe_strogatz_order_parameter(1.0 + 0.0j, 0.0 + 0.0j, constants) == pytest.approx(
        0.0, abs=1e-12
    )


def test_trajectory_shapes_and_terminal_properties() -> None:
    rng = np.random.default_rng(3)
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
    trajectory = integrate_watanabe_strogatz(
        initial_phases, omega=_OMEGA, coupling=_COUPLING, dt=_DT, n_steps=50
    )
    assert isinstance(trajectory, WatanabeStrogatzTrajectory)
    assert trajectory.alpha.shape == (51,)
    assert trajectory.phases.shape == (51, 6)
    assert trajectory.order_parameter.shape == (51,)
    assert trajectory.terminal_phases.shape == (6,)
    assert isinstance(trajectory.terminal_order_parameter, complex)
    assert trajectory.order_parameter[0] == pytest.approx(np.mean(np.exp(1j * initial_phases)))


def test_thermodynamic_limit_matches_ott_antonsen() -> None:
    rng = np.random.default_rng(4)
    n_oscillators = 20000
    # a partially coherent initial state
    initial_phases = 0.6 * np.sin(rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)) + rng.uniform(
        0.0, 2.0 * np.pi, size=n_oscillators
    )
    n_steps = 300
    ws = integrate_watanabe_strogatz(
        initial_phases, omega=_OMEGA, coupling=_COUPLING, dt=_DT, n_steps=n_steps
    )

    reduced = ws.order_parameter[0]
    ott_antonsen = [reduced]
    for _ in range(n_steps):

        def field(z: complex) -> complex:
            return ott_antonsen_field(z, _COUPLING, 0.0, centre=_OMEGA)

        k1 = field(reduced)
        k2 = field(reduced + 0.5 * _DT * k1)
        k3 = field(reduced + 0.5 * _DT * k2)
        k4 = field(reduced + _DT * k3)
        reduced = reduced + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ott_antonsen.append(reduced)

    # the WS order parameter converges to the OA equation at the finite-N sampling floor (~1/sqrt N)
    assert float(np.max(np.abs(ws.order_parameter - np.array(ott_antonsen)))) < 0.02


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_phases": np.zeros((2, 2))}, "initial_phases must be a non-empty"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "initial_phases": np.array([0.1, 0.2, 0.3]),
        "omega": _OMEGA,
        "coupling": _COUPLING,
        "dt": _DT,
        "n_steps": 10,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        integrate_watanabe_strogatz(
            call["initial_phases"],
            omega=call["omega"],
            coupling=call["coupling"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
