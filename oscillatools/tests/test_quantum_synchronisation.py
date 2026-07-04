# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for quantum synchronisation of a van der Pol oscillator
"""Module-specific tests for :mod:`quantum_synchronisation`.

The contracts: the free quantum van der Pol oscillator relaxes onto a phase-symmetric limit cycle
(excited, but ``⟨a⟩ = 0`` and a flat phase distribution); an external drive phase-locks it (``|⟨a⟩|``
and the phase-synchronisation measure become finite), strongest on resonance and fading with detuning
(the quantum Arnold tongue); the Lindblad RK4 preserves trace and Hermiticity; the observables and the
normalised phase distribution are correct on known states; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oscillatools.accel.quantum_synchronisation import (
    QuantumVanDerPolTrajectory,
    coherent_amplitude,
    integrate_quantum_vanderpol,
    mean_photon_number,
    phase_distribution,
    phase_synchronisation,
    vacuum_state,
)

_DIM = 16
_DT = 0.01
_STEPS = 3000
_GAIN = 1.0
_LOSS = 0.4


def _steady(detuning: float, drive: float) -> np.ndarray[Any, Any]:
    trajectory = integrate_quantum_vanderpol(
        vacuum_state(_DIM),
        _DT,
        _STEPS,
        detuning=detuning,
        drive=drive,
        one_photon_gain=_GAIN,
        two_photon_loss=_LOSS,
    )
    return trajectory.final_state


def test_free_oscillator_is_a_phase_symmetric_limit_cycle() -> None:
    state = _steady(0.0, 0.0)
    # excited onto the limit cycle but with no phase preference (rotational symmetry)
    assert mean_photon_number(state) > 0.5
    assert abs(coherent_amplitude(state)) < 1e-6
    assert phase_synchronisation(state) < 1e-6


def test_drive_phase_locks_the_oscillator() -> None:
    free = _steady(0.0, 0.0)
    driven = _steady(0.0, 2.0)
    assert abs(coherent_amplitude(driven)) > 1.0
    assert phase_synchronisation(driven) > 0.5
    assert phase_synchronisation(driven) > phase_synchronisation(free)


def test_synchronisation_follows_the_arnold_tongue() -> None:
    syncs = [phase_synchronisation(_steady(detuning, 2.0)) for detuning in (0.0, 1.0, 3.0, 6.0)]
    # synchronisation is maximal on resonance and decreases monotonically with detuning
    assert np.all(np.diff(syncs) < 0.0)
    assert syncs[0] > syncs[-1]


def test_evolution_preserves_trace_and_hermiticity() -> None:
    trajectory = integrate_quantum_vanderpol(
        vacuum_state(_DIM),
        0.01,
        80,
        detuning=0.5,
        drive=1.0,
        one_photon_gain=_GAIN,
        two_photon_loss=_LOSS,
    )
    assert isinstance(trajectory, QuantumVanDerPolTrajectory)
    traces = np.array([np.trace(state).real for state in trajectory.density_matrices])
    assert np.max(np.abs(traces - 1.0)) < 1e-9
    hermiticity = max(
        np.max(np.abs(state - state.conj().T)) for state in trajectory.density_matrices
    )
    assert hermiticity < 1e-9


def test_observables_and_phase_distribution_on_known_states() -> None:
    vacuum = vacuum_state(4)
    assert vacuum[0, 0] == 1.0
    assert mean_photon_number(vacuum) == pytest.approx(0.0)
    # the number state |1><1|: one photon, no coherent amplitude, flat phase distribution
    number_state = np.zeros((4, 4), dtype=np.complex128)
    number_state[1, 1] = 1.0
    assert mean_photon_number(number_state) == pytest.approx(1.0)
    assert coherent_amplitude(number_state) == pytest.approx(0.0)
    assert phase_synchronisation(number_state) == pytest.approx(0.0, abs=1e-9)
    # a |0> + |1> superposition has a coherent amplitude of 1/2
    superposition = np.full((2, 2), 0.5, dtype=np.complex128)
    assert coherent_amplitude(superposition) == pytest.approx(0.5)
    angles, distribution = phase_distribution(superposition, 200)
    assert np.sum(distribution) * (angles[1] - angles[0]) == pytest.approx(1.0)
    assert np.min(distribution) >= 0.0


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("vacuum", {"fock_dimension": 1}, "fock_dimension must be at least two"),
        (
            "observable",
            {"density_matrix": np.zeros((3, 4), dtype=np.complex128)},
            "must be a square",
        ),
        (
            "observable",
            {"density_matrix": np.full((1, 1), 1.0, dtype=np.complex128)},
            "must be a square",
        ),
        ("observable", {"density_matrix": np.eye(3, dtype=np.complex128)}, "must have unit trace"),
        (
            "observable",
            {"density_matrix": np.full((3, 3), np.nan, dtype=np.complex128)},
            "must be finite",
        ),
        ("phasedist", {"n_angles": 0}, "n_angles must be positive"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
        ("integrate", {"detuning": np.inf}, "detuning and drive must be finite"),
        ("integrate", {"one_photon_gain": -1.0}, "one_photon_gain must be non-negative"),
        ("integrate", {"two_photon_loss": 0.0}, "two_photon_loss must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        if call == "vacuum":
            vacuum_state(kwargs["fock_dimension"])
        elif call == "observable":
            mean_photon_number(kwargs["density_matrix"])
        elif call == "phasedist":
            phase_synchronisation(vacuum_state(4), n_angles=kwargs["n_angles"])
        else:
            args: dict[str, Any] = {
                "initial_state": vacuum_state(4),
                "dt": 0.01,
                "n_steps": 10,
                "detuning": 0.0,
                "drive": 1.0,
                "one_photon_gain": _GAIN,
                "two_photon_loss": _LOSS,
            }
            args.update(kwargs)
            integrate_quantum_vanderpol(
                args["initial_state"],
                args["dt"],
                args["n_steps"],
                detuning=args["detuning"],
                drive=args["drive"],
                one_photon_gain=args["one_photon_gain"],
                two_photon_loss=args["two_photon_loss"],
            )


def test_non_hermitian_density_matrix_is_rejected() -> None:
    non_hermitian = np.zeros((3, 3), dtype=np.complex128)
    np.fill_diagonal(non_hermitian, 1.0 / 3.0)
    non_hermitian[0, 1] = 0.5
    with pytest.raises(ValueError, match="must be Hermitian"):
        coherent_amplitude(non_hermitian)
