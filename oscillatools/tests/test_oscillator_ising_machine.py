# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Oscillator Ising Machine
"""Module-specific tests for :mod:`oscillator_ising_machine`.

The contracts: the machine field is exactly the negative gradient of the Lyapunov energy (so the
energy descends monotonically at fixed SHIL strength); the SHIL term binarises the phases to two
states; at binary phases the continuous coupling energy equals the discrete Ising Hamiltonian; with
``J = -A`` and a SHIL anneal the machine finds the true MAX-CUT of a small graph (checked against
brute force); and the input contract is enforced.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.oscillator_ising_machine import (
    OscillatorIsingTrajectory,
    cut_value,
    integrate_oscillator_ising_machine,
    ising_hamiltonian,
    ising_spins,
    oscillator_ising_energy,
    oscillator_ising_field,
)

# a small graph with a brute-force-checkable maximum cut
_ADJACENCY = np.array(
    [
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ],
    dtype=np.float64,
)
_N = 6


def _coupling(seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(_N, _N))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return coupling


def test_field_is_the_negative_energy_gradient() -> None:
    coupling = _coupling()
    phases = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, size=_N)
    shil = 0.7
    field = oscillator_ising_field(phases, coupling, shil)
    gradient = np.zeros(_N, dtype=np.float64)
    eps = 1e-6
    for index in range(_N):
        plus = phases.copy()
        minus = phases.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (
            oscillator_ising_energy(plus, coupling, shil)
            - oscillator_ising_energy(minus, coupling, shil)
        ) / (2.0 * eps)
    assert field == pytest.approx(-gradient, abs=1e-7)


def test_energy_descends_monotonically_at_fixed_shil() -> None:
    coupling = _coupling(2)
    phases = np.random.default_rng(3).uniform(0.0, 2.0 * np.pi, size=_N)
    shil = 0.7
    trajectory = integrate_oscillator_ising_machine(
        phases, coupling, 0.02, 1500, shil_strength=shil, anneal_fraction=0.0
    )
    assert isinstance(trajectory, OscillatorIsingTrajectory)
    energy = np.array(
        [oscillator_ising_energy(state, coupling, shil) for state in trajectory.phases]
    )
    assert np.all(np.diff(energy) <= 1e-9)
    # the SHIL term has binarised the converged phases to {0, pi}
    distance = np.minimum(
        np.abs(((trajectory.final_phases - 0.0 + np.pi) % (2.0 * np.pi)) - np.pi),
        np.abs(((trajectory.final_phases - np.pi + np.pi) % (2.0 * np.pi)) - np.pi),
    )
    assert np.max(distance) < 1e-2


def test_ising_hamiltonian_matches_continuous_energy_at_binary_phases() -> None:
    coupling = _coupling(4)
    spins = np.array([1, -1, 1, 1, -1, -1])
    phases = np.where(spins == 1, 0.0, np.pi)
    # at binary phases the SHIL term is at its minimum and the coupling energy is the Ising energy
    coupling_energy = oscillator_ising_energy(phases, coupling, 0.0)
    assert ising_hamiltonian(spins, coupling) == pytest.approx(coupling_energy)


def test_machine_finds_the_maximum_cut() -> None:
    brute_force = max(
        cut_value(np.array(configuration), _ADJACENCY)
        for configuration in itertools.product([-1, 1], repeat=_N)
    )
    best = 0.0
    for seed in range(20):
        phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=_N)
        trajectory = integrate_oscillator_ising_machine(
            phases, -_ADJACENCY, 0.05, 800, shil_strength=1.0, anneal_fraction=0.6
        )
        best = max(best, cut_value(trajectory.final_spins, _ADJACENCY))
    assert best == brute_force


def test_spin_and_cut_readout() -> None:
    phases = np.array([0.1, np.pi - 0.1, 0.0, np.pi + 0.2])
    spins = ising_spins(phases)
    assert spins.tolist() == [1, -1, 1, -1]
    adjacency = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float64
    )
    # this partition cuts every edge of the 4-cycle (4 undirected edges)
    assert cut_value(spins, adjacency) == pytest.approx(4.0)
    # H = -1/2 sigma^T A sigma; every adjacent pair is anti-aligned so sigma^T A sigma = -8
    assert ising_hamiltonian(spins, adjacency) == pytest.approx(4.0)


def test_trajectory_container() -> None:
    coupling = _coupling(5)
    phases = np.zeros(_N)
    trajectory = integrate_oscillator_ising_machine(
        phases, coupling, 0.05, 10, shil_strength=0.5, anneal_fraction=0.5
    )
    assert trajectory.phases.shape == (11, _N)
    assert trajectory.ising_energy_history.shape == (11,)
    assert trajectory.times[-1] == pytest.approx(10 * 0.05)
    assert trajectory.final_phases == pytest.approx(trajectory.phases[-1])
    assert set(trajectory.final_spins.tolist()) <= {-1, 1}


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("field", {"phases": np.zeros(1)}, "phases must be a one-dimensional"),
        ("field", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("field", {"coupling": np.triu(np.ones((_N, _N)), 1)}, "coupling must be symmetric"),
        ("field", {"phases": np.full(_N, np.nan)}, "must be finite"),
        ("field", {"shil_strength": -1.0}, "shil_strength must be non-negative"),
        ("hamiltonian", {"spins": np.ones(1, dtype=np.int_)}, "spins must be a one-dimensional"),
        ("hamiltonian", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("cut", {"spins": np.ones(1, dtype=np.int_)}, "spins must be a one-dimensional"),
        ("cut", {"adjacency": np.zeros((_N, _N + 1))}, "adjacency must have shape"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
        ("integrate", {"anneal_fraction": 1.5}, "anneal_fraction must lie in"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    coupling = _coupling()
    with pytest.raises(ValueError, match=message):
        if call == "field":
            args: dict[str, Any] = {
                "phases": np.zeros(_N),
                "coupling": coupling,
                "shil_strength": 0.5,
            }
            args.update(kwargs)
            oscillator_ising_field(args["phases"], args["coupling"], args["shil_strength"])
        elif call == "hamiltonian":
            args = {"spins": np.ones(_N, dtype=np.int_), "coupling": coupling}
            args.update(kwargs)
            ising_hamiltonian(args["spins"], args["coupling"])
        elif call == "cut":
            args = {"spins": np.ones(_N, dtype=np.int_), "adjacency": _ADJACENCY}
            args.update(kwargs)
            cut_value(args["spins"], args["adjacency"])
        else:
            args = {
                "phases": np.zeros(_N),
                "coupling": coupling,
                "dt": 0.05,
                "n_steps": 10,
                "shil_strength": 0.5,
                "anneal_fraction": 0.5,
            }
            args.update(kwargs)
            integrate_oscillator_ising_machine(
                args["phases"],
                args["coupling"],
                args["dt"],
                args["n_steps"],
                shil_strength=args["shil_strength"],
                anneal_fraction=args["anneal_fraction"],
            )
