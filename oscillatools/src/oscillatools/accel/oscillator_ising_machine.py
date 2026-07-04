# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Oscillator Ising Machine (sub-harmonic injection locking)
r"""Oscillator Ising Machine: solving combinatorial optimisation with coupled phase oscillators.

A network of coupled oscillators with *sub-harmonic injection locking* (SHIL) settles into a phase
configuration that minimises an Ising Hamiltonian, so the analogue dynamics is an optimisation solver
(Wang & Roychowdhury, 2019). The phase model is a Kuramoto coupling plus a second-harmonic SHIL term,

.. math::

    \dot\phi_i = -\sum_j J_{ij}\sin(\phi_i - \phi_j) - K_s\sin(2\phi_i),

whose ``\sin(2\phi_i)`` term has minima at ``\phi_i \in \{0, \pi\}`` and so pins each oscillator to one
of two binary phases — the Ising spins ``\sigma_i = \operatorname{sign}(\cos\phi_i)``. The flow is exact
gradient descent ``\dot\phi = -\nabla E`` on the Lyapunov energy

.. math::

    E(\phi) = -\tfrac12\sum_{ij} J_{ij}\cos(\phi_i - \phi_j) - \tfrac{K_s}{2}\sum_i\cos(2\phi_i),

so ``E`` decreases monotonically and, at binary phases, its coupling part equals the Ising Hamiltonian
``-\tfrac12\,\sigma^{\mathsf T} J\,\sigma``. Setting ``J = -A`` for a graph adjacency ``A`` turns the
machine into a MAX-CUT solver. Ramping ``K_s`` from zero (a continuous-then-binarising anneal) lets the
oscillators explore before committing to spins, which escapes shallow local minima. Because the field
is smooth, the whole solver is differentiable — coupling and schedule can themselves be optimised. It
adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OscillatorIsingTrajectory:
    """An Oscillator Ising Machine trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    phases : numpy.ndarray
        The ``(n_steps + 1, N)`` oscillator phases.
    ising_energy_history : numpy.ndarray
        The ``(n_steps + 1,)`` Ising Hamiltonian of the binarised spins at each step (the
        solution-quality curve, well defined throughout the anneal).
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    ising_energy_history: NDArray[np.float64]

    @property
    def final_phases(self) -> NDArray[np.float64]:
        """The phases at the final step."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)

    @property
    def final_spins(self) -> NDArray[np.int_]:
        """The Ising spins ``σ = sign(cos φ)`` at the final step."""
        return ising_spins(self.final_phases)


def oscillator_ising_field(
    phases: NDArray[np.float64], coupling: NDArray[np.float64], shil_strength: float
) -> NDArray[np.float64]:
    r"""The Oscillator Ising Machine field ``-Σ_j J_{ij}\sin(φ_i-φ_j) - K_s\sin(2φ_i)``.

    Parameters
    ----------
    phases : numpy.ndarray
        The oscillator phases ``φ`` (length ``N ≥ 2``).
    coupling : numpy.ndarray
        The symmetric ``(N, N)`` Ising coupling ``J``.
    shil_strength : float
        The SHIL strength ``K_s`` (``≥ 0``).

    Returns
    -------
    numpy.ndarray
        The phase velocity (length ``N``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate_state(angle, matrix, shil_strength)
    difference = angle[:, None] - angle[None, :]
    return np.asarray(
        -np.sum(matrix * np.sin(difference), axis=1) - shil_strength * np.sin(2.0 * angle),
        dtype=np.float64,
    )


def oscillator_ising_energy(
    phases: NDArray[np.float64], coupling: NDArray[np.float64], shil_strength: float
) -> float:
    r"""The Lyapunov energy ``E = -½Σ_{ij} J_{ij}\cos(φ_i-φ_j) - ½K_s Σ_i\cos(2φ_i)``.

    The Oscillator Ising Machine field is exactly ``-∇E``, so ``E`` decreases monotonically along the
    flow at fixed ``K_s``.

    Parameters
    ----------
    phases, coupling, shil_strength
        As for :func:`oscillator_ising_field`.

    Returns
    -------
    float
        The energy.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate_state(angle, matrix, shil_strength)
    difference = angle[:, None] - angle[None, :]
    coupling_energy = -0.5 * float(np.sum(matrix * np.cos(difference)))
    shil_energy = -0.5 * shil_strength * float(np.sum(np.cos(2.0 * angle)))
    return coupling_energy + shil_energy


def ising_spins(phases: NDArray[np.float64]) -> NDArray[np.int_]:
    r"""Binarise phases to Ising spins ``σ_i = sign(cos φ_i)`` (``+1`` at ``φ≈0``, ``-1`` at ``φ≈π``)."""
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    return np.where(np.cos(angle) >= 0.0, 1, -1).astype(np.int_)


def ising_hamiltonian(spins: NDArray[np.int_], coupling: NDArray[np.float64]) -> float:
    r"""The Ising Hamiltonian ``-½\,σ^{\mathsf T} J\,σ`` of a spin configuration."""
    configuration = np.ascontiguousarray(spins, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if configuration.ndim != 1 or configuration.size < 2:
        raise ValueError("spins must be a one-dimensional array of length at least two")
    if matrix.shape != (configuration.size, configuration.size):
        raise ValueError(f"coupling must have shape ({configuration.size}, {configuration.size})")
    return -0.5 * float(configuration @ matrix @ configuration)


def cut_value(spins: NDArray[np.int_], adjacency: NDArray[np.float64]) -> float:
    r"""The graph cut ``¼ Σ_{ij} A_{ij}(1 - σ_iσ_j)`` of a spin partition."""
    configuration = np.ascontiguousarray(spins, dtype=np.float64)
    graph = np.ascontiguousarray(adjacency, dtype=np.float64)
    if configuration.ndim != 1 or configuration.size < 2:
        raise ValueError("spins must be a one-dimensional array of length at least two")
    if graph.shape != (configuration.size, configuration.size):
        raise ValueError(f"adjacency must have shape ({configuration.size}, {configuration.size})")
    return 0.25 * float(np.sum(graph * (1.0 - np.outer(configuration, configuration))))


def _validate_state(
    phases: NDArray[np.float64], coupling: NDArray[np.float64], shil_strength: float
) -> int:
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phases must be a one-dimensional array of length at least two")
    count = phases.size
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not np.allclose(coupling, coupling.T):
        raise ValueError("coupling must be symmetric")
    if not (np.all(np.isfinite(phases)) and np.all(np.isfinite(coupling))):
        raise ValueError("phases and coupling must be finite")
    if shil_strength < 0.0:
        raise ValueError(f"shil_strength must be non-negative, got {shil_strength}")
    return int(count)


def integrate_oscillator_ising_machine(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    shil_strength: float,
    anneal_fraction: float,
) -> OscillatorIsingTrajectory:
    r"""Integrate the Oscillator Ising Machine by RK4 with a SHIL anneal.

    The SHIL strength ramps linearly from zero to ``shil_strength`` over the first ``anneal_fraction``
    of the rollout (and is held thereafter); ``anneal_fraction = 0`` holds ``K_s`` constant from the
    start. The binarised-spin Ising Hamiltonian is recorded at every step.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``φ(0)`` (length ``N ≥ 2``).
    coupling : numpy.ndarray
        The symmetric ``(N, N)`` Ising coupling ``J`` (use ``J = -A`` to solve MAX-CUT on ``A``).
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``).
    shil_strength : float
        The final SHIL strength ``K_s`` (``≥ 0``).
    anneal_fraction : float
        The fraction of the rollout over which ``K_s`` ramps from zero (in ``[0, 1]``).

    Returns
    -------
    OscillatorIsingTrajectory
        The phase trajectory and the Ising-energy history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_state(angle, matrix, shil_strength)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not 0.0 <= anneal_fraction <= 1.0:
        raise ValueError(f"anneal_fraction must lie in [0, 1], got {anneal_fraction}")

    def schedule(step: int) -> float:
        if anneal_fraction == 0.0:
            return shil_strength
        return shil_strength * min(1.0, (step / n_steps) / anneal_fraction)

    def field(state: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
        difference = state[:, None] - state[None, :]
        return np.asarray(
            -np.sum(matrix * np.sin(difference), axis=1) - strength * np.sin(2.0 * state),
            dtype=np.float64,
        )

    trajectory = np.empty((n_steps + 1, count), dtype=np.float64)
    energy_history = np.empty(n_steps + 1, dtype=np.float64)
    trajectory[0] = angle
    energy_history[0] = ising_hamiltonian(ising_spins(angle), matrix)
    current = angle
    for step in range(n_steps):
        strength = schedule(step)
        k1 = field(current, strength)
        k2 = field(current + 0.5 * dt * k1, strength)
        k3 = field(current + 0.5 * dt * k2, strength)
        k4 = field(current + dt * k3, strength)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
        energy_history[step + 1] = ising_hamiltonian(ising_spins(current), matrix)
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return OscillatorIsingTrajectory(
        times=times, phases=trajectory, ising_energy_history=energy_history
    )


__all__ = [
    "OscillatorIsingTrajectory",
    "cut_value",
    "integrate_oscillator_ising_machine",
    "ising_hamiltonian",
    "ising_spins",
    "oscillator_ising_energy",
    "oscillator_ising_field",
]
