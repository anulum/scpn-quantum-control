# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for coordinated-reset desynchronisation control
"""Module-specific tests for :mod:`kuramoto_coordinated_reset`.

The control contracts are validated directly: coordinated reset collapses a synchronised
population's order parameter; on a plastic (Hebbian) substrate the desynchronised state survives a
perturbation (the carryover) whereas a static synchronising coupling re-synchronises; and the
desynchronising terminal objective is differentiable through the controlled flow, with the
forward-mode gradient matching a central finite difference on every channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel import order_parameter, order_parameter_gradient
from scpn_quantum_control.accel.kuramoto_coordinated_reset import (
    CoordinatedResetGradients,
    CoordinatedResetTrajectory,
    coordinated_reset_phases,
    coordinated_reset_sites,
    coordinated_reset_terminal_value_and_grad,
    integrate_coordinated_reset,
)


def test_sites_and_reset_phases() -> None:
    assert coordinated_reset_sites(6, 3).tolist() == [0, 1, 2, 0, 1, 2]
    assert coordinated_reset_phases(4) == pytest.approx(np.array([0.0, 0.5, 1.0, 1.5]) * np.pi)


@pytest.mark.parametrize(
    ("n_oscillators", "n_sites"),
    [(6, 1), (3, 5)],
)
def test_sites_validation(n_oscillators: int, n_sites: int) -> None:
    with pytest.raises(ValueError):
        coordinated_reset_sites(n_oscillators, n_sites)


def test_coordinated_reset_collapses_the_order_parameter() -> None:
    rng = np.random.default_rng(0)
    n_oscillators = 40
    initial_phases = rng.normal(0.0, 0.05, size=n_oscillators)  # nearly synchronised
    omega = np.zeros(n_oscillators)
    coupling = np.full((n_oscillators, n_oscillators), 6.0 / n_oscillators)
    np.fill_diagonal(coupling, 0.0)
    trajectory = integrate_coordinated_reset(
        initial_phases, omega, coupling, n_sites=4, dt=0.01, steps_per_pulse=8, n_cycles=20
    )
    assert isinstance(trajectory, CoordinatedResetTrajectory)
    assert trajectory.order_parameter_series[0] > 0.95
    assert trajectory.terminal_order_parameter < 1e-3


def test_plastic_carryover_outlasts_stimulation() -> None:
    rng = np.random.default_rng(0)
    n_oscillators = 60
    initial_phases = rng.normal(0.0, 0.05, size=n_oscillators)
    omega = np.zeros(n_oscillators)
    coupling = np.full((n_oscillators, n_oscillators), 6.0 / n_oscillators)
    np.fill_diagonal(coupling, 0.0)
    shared: dict[str, Any] = dict(
        n_sites=4,
        dt=0.01,
        steps_per_pulse=8,
        n_cycles=20,
        free_cycles=15,
        free_perturbation=0.3,
        seed=1,
    )
    static = integrate_coordinated_reset(
        initial_phases, omega, coupling, plasticity_rate=0.0, **shared
    )
    plastic = integrate_coordinated_reset(
        initial_phases, omega, coupling, plasticity_rate=0.5, **shared
    )
    # the static synchronising coupling recovers after the perturbation; the rewired plastic one does not
    assert static.order_parameter_series[-1] > 0.9
    assert plastic.order_parameter_series[-1] < 1e-2
    assert plastic.terminal_coupling.mean() < coupling.mean()


def test_trajectory_shapes_and_metadata() -> None:
    rng = np.random.default_rng(2)
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=8)
    omega = np.zeros(8)
    coupling = np.full((8, 8), 0.5)
    np.fill_diagonal(coupling, 0.0)
    trajectory = integrate_coordinated_reset(
        initial_phases,
        omega,
        coupling,
        n_sites=2,
        dt=0.02,
        steps_per_pulse=3,
        n_cycles=4,
        free_cycles=2,
    )
    total_steps = (4 + 2) * 2 * 3
    assert trajectory.order_parameter_series.shape == (total_steps + 1,)
    assert trajectory.stimulation_steps == 4 * 2 * 3
    assert trajectory.site_assignment.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert np.array_equal(trajectory.terminal_coupling, coupling)  # static run leaves coupling


def test_desync_objective_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    n_oscillators = 6
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    omega = rng.standard_normal(n_oscillators)
    raw = rng.uniform(0.0, 1.0, size=(n_oscillators, n_oscillators))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    shared: dict[str, Any] = dict(n_sites=2, dt=0.02, steps_per_pulse=3, n_cycles=2)

    value, grads = coordinated_reset_terminal_value_and_grad(
        initial_phases,
        omega,
        coupling,
        objective=lambda theta: float(order_parameter(theta)),
        objective_grad=lambda theta: order_parameter_gradient(theta),
        **shared,
    )
    assert isinstance(grads, CoordinatedResetGradients)

    def loss(
        phases: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        coupling_matrix: NDArray[np.float64],
    ) -> float:
        return float(
            order_parameter(
                integrate_coordinated_reset(
                    phases, frequencies, coupling_matrix, **shared
                ).terminal_phases
            )
        )

    assert value == pytest.approx(loss(initial_phases, omega, coupling))

    def central_difference(base: NDArray[np.float64], setter: Any) -> NDArray[np.float64]:
        eps = 1e-6
        grad = np.zeros_like(base)
        for index in np.ndindex(base.shape):
            high = base.copy()
            high[index] += eps
            low = base.copy()
            low[index] -= eps
            grad[index] = (setter(high) - setter(low)) / (2.0 * eps)
        return grad

    fd_phases = central_difference(initial_phases, lambda x: loss(x, omega, coupling))
    fd_omega = central_difference(omega, lambda x: loss(initial_phases, x, coupling))
    fd_coupling = central_difference(coupling, lambda x: loss(initial_phases, omega, x))

    assert grads.initial_phases == pytest.approx(fd_phases, abs=1e-7)
    assert grads.omega == pytest.approx(fd_omega, abs=1e-7)
    assert grads.coupling == pytest.approx(fd_coupling, abs=1e-7)


def test_objective_grad_wrong_shape_raises() -> None:
    initial_phases = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    omega = np.zeros(4)
    coupling = np.full((4, 4), 0.5)
    np.fill_diagonal(coupling, 0.0)
    with pytest.raises(ValueError, match="objective_grad must return"):
        coordinated_reset_terminal_value_and_grad(
            initial_phases,
            omega,
            coupling,
            n_sites=2,
            dt=0.02,
            steps_per_pulse=2,
            n_cycles=1,
            objective=lambda theta: 0.0,
            objective_grad=lambda theta: np.zeros(theta.size + 1),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"initial_phases": np.zeros((2, 2))}, "initial_phases must be a non-empty"),
        ({"omega": np.zeros(3)}, "omega must have shape"),
        ({"coupling": np.zeros((3, 3))}, "coupling must have shape"),
        ({"n_sites": 1}, "n_sites must satisfy"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"steps_per_pulse": 0}, "steps_per_pulse must be positive"),
        ({"n_cycles": 0}, "n_cycles must be positive"),
        ({"plasticity_rate": -1.0}, "plasticity_rate must be non-negative"),
        ({"free_cycles": -1}, "free_cycles must be non-negative"),
        ({"free_perturbation": -1.0}, "free_perturbation must be non-negative"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "initial_phases": np.array([0.1, 0.2, 0.3, 0.4]),
        "omega": np.zeros(4),
        "coupling": np.full((4, 4), 0.5),
        "n_sites": 2,
        "dt": 0.02,
        "steps_per_pulse": 2,
        "n_cycles": 1,
    }
    np.fill_diagonal(call["coupling"], 0.0)
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        integrate_coordinated_reset(
            call["initial_phases"],
            call["omega"],
            call["coupling"],
            n_sites=call["n_sites"],
            dt=call["dt"],
            steps_per_pulse=call["steps_per_pulse"],
            n_cycles=call["n_cycles"],
            plasticity_rate=call.get("plasticity_rate", 0.0),
            free_cycles=call.get("free_cycles", 0),
            free_perturbation=call.get("free_perturbation", 0.0),
        )
