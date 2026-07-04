# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the symplectic inertial Kuramoto integrator
"""Module-specific tests for :mod:`kuramoto_symplectic_inertial`.

The symplectic contract is the long-time energy behaviour: in the undamped Hamiltonian limit the
Verlet energy error stays bounded (no secular growth) whereas the RK4 integrator's drifts, while
over short times the two agree to second order. The damped flow dissipates energy monotonically,
and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import kuramoto_interaction_energy
from oscillatools.accel.kuramoto_inertial import (
    InertialTrajectory,
    inertial_energy,
    integrate_inertial,
)
from oscillatools.accel.kuramoto_symplectic_inertial import integrate_symplectic_inertial
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force

_N = 6
_MASS = 1.3


def _problem(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(_N, _N))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=_N),
        "velocities": rng.standard_normal(_N) * 0.5,
        "omega": np.zeros(_N),
        "coupling": coupling,
    }


def _energy_drift_ratio(
    trajectory: InertialTrajectory,
    omega: NDArray[np.float64],
    potential: Any,
    reference_energy: float,
) -> float:
    """Ratio of the energy error in the final tenth of the run to that in the first tenth."""
    samples = np.arange(0, trajectory.times.size, 50)
    energies = np.array(
        [
            inertial_energy(
                trajectory.phases[i], trajectory.velocities[i], omega, potential, _MASS
            )
            for i in samples
        ]
    )
    window = max(1, samples.size // 10)
    early = float(np.max(np.abs(energies[:window] - reference_energy)))
    late = float(np.max(np.abs(energies[-window:] - reference_energy)))
    return late / max(early, 1e-12)


def test_undamped_energy_stays_bounded_while_rk4_drifts() -> None:
    problem = _problem()
    coupling = problem["coupling"]
    force = lambda theta: networked_kuramoto_force(theta, coupling)  # noqa: E731
    potential = lambda theta: kuramoto_interaction_energy(theta, coupling)  # noqa: E731
    dt, n_steps = 0.1, 12000
    reference_energy = inertial_energy(
        problem["phases"], problem["velocities"], problem["omega"], potential, _MASS
    )

    verlet = integrate_symplectic_inertial(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        force,
        _MASS,
        damping=0.0,
        dt=dt,
        n_steps=n_steps,
    )
    rk4 = integrate_inertial(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        force,
        _MASS,
        damping=0.0,
        dt=dt,
        n_steps=n_steps,
    )

    verlet_ratio = _energy_drift_ratio(verlet, problem["omega"], potential, reference_energy)
    rk4_ratio = _energy_drift_ratio(rk4, problem["omega"], potential, reference_energy)
    # the RK4 energy error drifts (grows); the symplectic Verlet error stays bounded
    assert rk4_ratio > 3.0
    assert verlet_ratio < 2.0


def test_short_time_agreement_is_second_order() -> None:
    problem = _problem(seed=1)
    coupling = problem["coupling"]
    force = lambda theta: networked_kuramoto_force(theta, coupling)  # noqa: E731
    verlet = integrate_symplectic_inertial(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        force,
        _MASS,
        damping=0.5,
        dt=0.01,
        n_steps=50,
    )
    rk4 = integrate_inertial(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        force,
        _MASS,
        damping=0.5,
        dt=0.01,
        n_steps=50,
    )
    assert isinstance(verlet, InertialTrajectory)
    assert verlet.phases.shape == (51, _N)
    assert verlet.terminal_phases == pytest.approx(rk4.terminal_phases, abs=1e-4)


def test_damped_energy_dissipates_monotonically() -> None:
    problem = _problem(seed=2)
    coupling = problem["coupling"]
    force = lambda theta: networked_kuramoto_force(theta, coupling)  # noqa: E731
    potential = lambda theta: kuramoto_interaction_energy(theta, coupling)  # noqa: E731
    trajectory = integrate_symplectic_inertial(
        problem["phases"],
        problem["velocities"],
        problem["omega"],
        force,
        _MASS,
        damping=0.8,
        dt=0.05,
        n_steps=2000,
    )
    energy = trajectory.energy(problem["omega"], potential)
    assert np.all(np.diff(energy) <= 1e-9)  # monotone non-increasing (dE/dt = -gamma |v|^2)
    assert energy[-1] < energy[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"omega": np.zeros((2, 2))}, "omega must be a non-empty"),
        ({"phases": np.zeros(3)}, "phases must have shape"),
        ({"velocities": np.zeros(3)}, "velocities must have shape"),
        ({"mass": 0.0}, "mass must be positive"),
        ({"damping": -1.0}, "damping must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    problem = _problem()
    coupling = problem["coupling"]
    force = lambda theta: networked_kuramoto_force(theta, coupling)  # noqa: E731
    call: dict[str, Any] = {
        "phases": problem["phases"],
        "velocities": problem["velocities"],
        "omega": problem["omega"],
        "mass": _MASS,
        "damping": 0.5,
        "dt": 0.01,
        "n_steps": 10,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        integrate_symplectic_inertial(
            call["phases"],
            call["velocities"],
            call["omega"],
            force,
            call["mass"],
            damping=call["damping"],
            dt=call["dt"],
            n_steps=call["n_steps"],
        )
