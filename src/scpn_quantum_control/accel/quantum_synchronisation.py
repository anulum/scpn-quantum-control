# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum synchronisation of a van der Pol oscillator
r"""Quantum synchronisation of a van der Pol limit-cycle oscillator.

The quantum van der Pol oscillator is the canonical quantum limit-cycle system and the quantum
analogue of the Kuramoto unit (Lee & Sadeghpour, 2013; Walter, Nadlinger & Bruder, 2014). Its density
matrix evolves under a Lindblad master equation

.. math::

    \dot\rho = -i[H, \rho] + \gamma_1\,\mathcal D[a^\dagger]\rho + \gamma_2\,\mathcal D[a^2]\rho,
    \qquad \mathcal D[L]\rho = L\rho L^\dagger - \tfrac12\{L^\dagger L, \rho\},

with one-photon *gain* ``\gamma_1\mathcal D[a^\dagger]`` (negative damping that pumps the limit cycle)
and two-photon *loss* ``\gamma_2\mathcal D[a^2]`` (the nonlinear saturation). In the rotating frame of
an external drive the Hamiltonian is ``H = -\Delta\,a^\dagger a + \varepsilon\,(a + a^\dagger)`` with
detuning ``\Delta`` and drive strength ``\varepsilon``. The free oscillator sits on a phase-symmetric
limit cycle (excited but with ``\langle a\rangle = 0`` and a flat phase distribution); the drive breaks
that symmetry and *phase-locks* the oscillator — the phase distribution develops a peak and
``|\langle a\rangle|`` becomes finite, strongest on resonance and fading with detuning (the quantum
Arnold tongue). In the large-amplitude limit the model reduces to the classical van der Pol oscillator
and, after phase reduction, to the Adler / Kuramoto phase equation. The state is represented in a
truncated Fock basis and evolved by trace-preserving RK4. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QuantumVanDerPolTrajectory:
    """A quantum van der Pol density-matrix trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    density_matrices : numpy.ndarray
        The ``(n_steps + 1, D, D)`` complex density matrices in the Fock basis.
    """

    times: NDArray[np.float64]
    density_matrices: NDArray[np.complex128]

    @property
    def final_state(self) -> NDArray[np.complex128]:
        """The density matrix at the final step."""
        return np.ascontiguousarray(self.density_matrices[-1], dtype=np.complex128)


def _annihilation(dimension: int) -> NDArray[np.complex128]:
    """The truncated annihilation operator ``a`` (``a|n⟩ = √n|n-1⟩``)."""
    operator = np.zeros((dimension, dimension), dtype=np.complex128)
    roots = np.sqrt(np.arange(1, dimension, dtype=np.float64))
    operator[np.arange(dimension - 1), np.arange(1, dimension)] = roots
    return operator


def vacuum_state(fock_dimension: int) -> NDArray[np.complex128]:
    """The Fock vacuum density matrix ``|0⟩⟨0|`` of the given truncation."""
    if fock_dimension < 2:
        raise ValueError(f"fock_dimension must be at least two, got {fock_dimension}")
    state = np.zeros((fock_dimension, fock_dimension), dtype=np.complex128)
    state[0, 0] = 1.0
    return state


def coherent_amplitude(density_matrix: NDArray[np.complex128]) -> complex:
    r"""The coherent amplitude ``⟨a⟩ = Tr(ρ a)`` (the synchronisation order parameter)."""
    state = _validate_density_matrix(density_matrix)
    return complex(np.trace(state @ _annihilation(state.shape[0])))


def mean_photon_number(density_matrix: NDArray[np.complex128]) -> float:
    r"""The mean photon number ``⟨a^\dagger a⟩`` (the limit-cycle excitation)."""
    state = _validate_density_matrix(density_matrix)
    annihilation = _annihilation(state.shape[0])
    return float(np.real(np.trace(state @ annihilation.conj().T @ annihilation)))


def phase_distribution(
    density_matrix: NDArray[np.complex128], n_angles: int = 256
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The London phase distribution ``P(φ) = (2π)^{-1} Σ_{mn} ρ_{mn} e^{i(n-m)φ}``.

    Parameters
    ----------
    density_matrix : numpy.ndarray
        The ``(D, D)`` density matrix.
    n_angles : int
        The number of phase samples on ``[0, 2π)`` (``≥ 1``).

    Returns
    -------
    tuple of numpy.ndarray
        The phase grid and the (real, normalised) phase distribution.
    """
    state = _validate_density_matrix(density_matrix)
    if n_angles < 1:
        raise ValueError(f"n_angles must be positive, got {n_angles}")
    dimension = state.shape[0]
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float64)
    basis = np.exp(1j * np.outer(angles, np.arange(dimension)))
    distribution = np.real(np.einsum("km,mn,kn->k", basis.conj(), state, basis)) / (2.0 * np.pi)
    return angles, np.ascontiguousarray(distribution, dtype=np.float64)


def phase_synchronisation(density_matrix: NDArray[np.complex128], n_angles: int = 256) -> float:
    r"""The phase-synchronisation measure ``S = max_φ P(φ) - (2π)^{-1}`` (zero when phase-symmetric)."""
    _, distribution = phase_distribution(density_matrix, n_angles)
    return float(np.max(distribution) - 1.0 / (2.0 * np.pi))


def _validate_density_matrix(density_matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    state = np.ascontiguousarray(density_matrix, dtype=np.complex128)
    if state.ndim != 2 or state.shape[0] != state.shape[1] or state.shape[0] < 2:
        raise ValueError("density_matrix must be a square matrix of dimension at least two")
    if not np.all(np.isfinite(state)):
        raise ValueError("density_matrix must be finite")
    if not np.allclose(state, state.conj().T, atol=1e-8):
        raise ValueError("density_matrix must be Hermitian")
    if not np.isclose(np.trace(state).real, 1.0, atol=1e-6):
        raise ValueError("density_matrix must have unit trace")
    return state


def integrate_quantum_vanderpol(
    initial_state: NDArray[np.complex128],
    dt: float,
    n_steps: int,
    *,
    detuning: float,
    drive: float,
    one_photon_gain: float,
    two_photon_loss: float,
) -> QuantumVanDerPolTrajectory:
    r"""Integrate the driven quantum van der Pol master equation by trace-preserving RK4.

    Parameters
    ----------
    initial_state : numpy.ndarray
        The ``(D, D)`` initial density matrix (Hermitian, unit trace, ``D ≥ 2``).
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    detuning : float
        The drive detuning ``Δ`` in the rotating frame.
    drive : float
        The external drive strength ``ε``.
    one_photon_gain : float
        The one-photon gain rate ``γ_1`` (``≥ 0``).
    two_photon_loss : float
        The two-photon loss rate ``γ_2`` (``> 0``; the nonlinear saturation).

    Returns
    -------
    QuantumVanDerPolTrajectory
        The density-matrix trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    state = _validate_density_matrix(initial_state)
    dimension = state.shape[0]
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not (np.isfinite(detuning) and np.isfinite(drive)):
        raise ValueError("detuning and drive must be finite")
    if one_photon_gain < 0.0:
        raise ValueError(f"one_photon_gain must be non-negative, got {one_photon_gain}")
    if two_photon_loss <= 0.0:
        raise ValueError(f"two_photon_loss must be positive, got {two_photon_loss}")

    annihilation = _annihilation(dimension)
    creation = annihilation.conj().T
    number = creation @ annihilation
    hamiltonian = -detuning * number + drive * (annihilation + creation)
    gain_jump = np.sqrt(one_photon_gain) * creation
    loss_jump = np.sqrt(two_photon_loss) * (annihilation @ annihilation)
    jumps = (gain_jump, loss_jump)

    def generator(density: NDArray[np.complex128]) -> NDArray[np.complex128]:
        evolution = -1j * (hamiltonian @ density - density @ hamiltonian)
        for jump in jumps:
            adjoint = jump.conj().T
            product = adjoint @ jump
            evolution = (
                evolution
                + jump @ density @ adjoint
                - 0.5 * (product @ density + density @ product)
            )
        return evolution

    trajectory = np.empty((n_steps + 1, dimension, dimension), dtype=np.complex128)
    trajectory[0] = state
    current = state
    for step in range(n_steps):
        k1 = generator(current)
        k2 = generator(current + 0.5 * dt * k1)
        k3 = generator(current + 0.5 * dt * k2)
        k4 = generator(current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return QuantumVanDerPolTrajectory(times=times, density_matrices=trajectory)


__all__ = [
    "QuantumVanDerPolTrajectory",
    "coherent_amplitude",
    "integrate_quantum_vanderpol",
    "mean_photon_number",
    "phase_distribution",
    "phase_synchronisation",
    "vacuum_state",
]
