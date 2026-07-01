# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Swarmalators: oscillators that swarm and synchronise
r"""Swarmalators — oscillators whose phase and spatial motion are bidirectionally coupled.

A Kuramoto oscillator has only a phase. A *swarmalator* (O'Keeffe, Hong & Strogatz, Nat. Commun. 8,
1504, 2017) additionally carries a position in space, and the two couple both ways: oscillators in a
similar phase attract more strongly (phase shapes the swarm) and nearby oscillators synchronise more
strongly (space shapes the phase). This space–phase feedback is absent from every phase-only model
and produces states a phase model cannot — the *static phase wave* and *splintered / active phase
waves* in which the phase is organised in space rather than merely synchronised.

The canonical model (identical, non-self-propelled units with unit attraction and repulsion) is, for
positions ``x_i ∈ ℝ²`` and phases ``θ_i``,

.. math::

    \dot x_i = \frac1N\sum_{j\neq i}\Bigl[\frac{x_j-x_i}{|x_j-x_i|}\bigl(1 + J\cos(\theta_j-\theta_i)\bigr)
                - \frac{x_j-x_i}{|x_j-x_i|^2}\Bigr], \qquad
    \dot\theta_i = \frac KN\sum_{j\neq i}\frac{\sin(\theta_j-\theta_i)}{|x_j-x_i|},

where ``J`` sets how strongly phase similarity modulates spatial attraction and ``K`` sets the
spatial-distance-weighted phase coupling. The states are read off by the **rainbow order parameters**
``W_\pm = S_\pm e^{i\Psi_\pm} = \frac1N\sum_j e^{i(\phi_j \pm \theta_j)}`` (with ``\phi_j`` the polar
angle of ``x_j``): ``S_\pm \to 1`` in a static phase wave (phase locked to spatial angle), and the
ordinary phase coherence ``\to 1`` in static synchrony.

Self-interaction is excluded; the spatial repulsion keeps distinct units apart, so the integrand is
finite for any configuration of distinct positions. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SwarmalatorTrajectory:
    """A swarmalator trajectory in position–phase space.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    positions : numpy.ndarray
        The ``(n_steps + 1, N, 2)`` planar positions ``x(t)``.
    phases : numpy.ndarray
        The ``(n_steps + 1, N)`` phases ``θ(t)``.
    """

    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    phases: NDArray[np.float64]

    @property
    def terminal_positions(self) -> NDArray[np.float64]:
        """The final positions ``x(T)``."""
        return np.ascontiguousarray(self.positions[-1], dtype=np.float64)

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phases ``θ(T)``."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)


@dataclass(frozen=True)
class SwarmalatorOrderParameters:
    """The rainbow order parameters of a swarmalator configuration.

    Attributes
    ----------
    s_plus : float
        ``S_+ = |⟨e^{i(φ_j + θ_j)}⟩|`` — the correlation of spatial angle with ``+θ``.
    s_minus : float
        ``S_- = |⟨e^{i(φ_j − θ_j)}⟩|`` — the correlation of spatial angle with ``−θ``.
    phase_coherence : float
        The ordinary Kuramoto phase coherence ``|⟨e^{iθ_j}⟩|``.
    """

    s_plus: float
    s_minus: float
    phase_coherence: float


def _validate(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    coupling_phase: float,
    coupling_space: float,
) -> int:
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("positions must be an (N >= 2, 2) array")
    count = positions.shape[0]
    if phases.shape != (count,):
        raise ValueError(f"phases must have shape ({count},), got {phases.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases must be finite")
    if not np.isfinite(coupling_phase):
        raise ValueError(f"coupling_phase must be finite, got {coupling_phase}")
    if not np.isfinite(coupling_space):
        raise ValueError(f"coupling_space must be finite, got {coupling_space}")
    separation = positions[None, :, :] - positions[:, None, :]
    squared_distance = np.sum(separation**2, axis=2)
    off_diagonal = ~np.eye(count, dtype=bool)
    if np.any(squared_distance[off_diagonal] == 0.0):
        raise ValueError("positions must be pairwise distinct")
    return int(count)


def _field(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    coupling_phase: float,
    coupling_space: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(ẋ, θ̇)`` for the swarmalator ensemble (self-interaction excluded)."""
    count = positions.shape[0]
    separation = positions[None, :, :] - positions[:, None, :]
    distance = np.sqrt(np.sum(separation**2, axis=2))
    np.fill_diagonal(distance, np.inf)
    unit = separation / distance[..., None]
    phase_difference = phases[None, :] - phases[:, None]

    spatial_coefficient = (1.0 + coupling_phase * np.cos(phase_difference)) - 1.0 / distance
    position_velocity = np.sum(unit * spatial_coefficient[..., None], axis=1) / count
    phase_velocity = (coupling_space / count) * np.sum(np.sin(phase_difference) / distance, axis=1)
    return (
        np.asarray(position_velocity, dtype=np.float64),
        np.asarray(phase_velocity, dtype=np.float64),
    )


def swarmalator_field(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    coupling_phase: float,
    coupling_space: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The swarmalator vector field ``(ẋ, θ̇)``.

    Parameters
    ----------
    positions : numpy.ndarray
        The planar positions ``x`` (shape ``(N, 2)``, ``N ≥ 2``, distinct points).
    phases : numpy.ndarray
        The phases ``θ`` (length ``N``).
    coupling_phase : float
        The phase–space attraction coupling ``J``.
    coupling_space : float
        The space-weighted phase coupling ``K``.

    Returns
    -------
    tuple of numpy.ndarray
        The position velocities ``ẋ`` (shape ``(N, 2)``) and phase velocities ``θ̇`` (length ``N``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    location = np.ascontiguousarray(positions, dtype=np.float64)
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    _validate(location, angle, coupling_phase, coupling_space)
    return _field(location, angle, coupling_phase, coupling_space)


def integrate_swarmalators(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    coupling_phase: float,
    coupling_space: float,
    dt: float,
    n_steps: int,
) -> SwarmalatorTrajectory:
    r"""Integrate the swarmalator ensemble with classic RK4.

    Parameters
    ----------
    positions, phases, coupling_phase, coupling_space
        As for :func:`swarmalator_field`.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    SwarmalatorTrajectory
        The sampled position and phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    location = np.ascontiguousarray(positions, dtype=np.float64)
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    count = _validate(location, angle, coupling_phase, coupling_space)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    position_history = np.empty((n_steps + 1, count, 2), dtype=np.float64)
    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)
    position_history[0] = location
    phase_history[0] = angle
    for step in range(n_steps):
        p1, t1 = _field(location, angle, coupling_phase, coupling_space)
        p2, t2 = _field(
            location + 0.5 * dt * p1, angle + 0.5 * dt * t1, coupling_phase, coupling_space
        )
        p3, t3 = _field(
            location + 0.5 * dt * p2, angle + 0.5 * dt * t2, coupling_phase, coupling_space
        )
        p4, t4 = _field(location + dt * p3, angle + dt * t3, coupling_phase, coupling_space)
        location = location + (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4)
        angle = angle + (dt / 6.0) * (t1 + 2.0 * t2 + 2.0 * t3 + t4)
        position_history[step + 1] = location
        phase_history[step + 1] = angle
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return SwarmalatorTrajectory(times=times, positions=position_history, phases=phase_history)


def swarmalator_order_parameters(
    positions: NDArray[np.float64], phases: NDArray[np.float64]
) -> SwarmalatorOrderParameters:
    r"""The rainbow order parameters ``S_±`` and the phase coherence of a configuration.

    Parameters
    ----------
    positions : numpy.ndarray
        The planar positions ``x`` (shape ``(N, 2)``).
    phases : numpy.ndarray
        The phases ``θ`` (length ``N``).

    Returns
    -------
    SwarmalatorOrderParameters
        ``S_+``, ``S_-`` and the ordinary phase coherence.

    Raises
    ------
    ValueError
        If the inputs are mismatched.
    """
    location = np.ascontiguousarray(positions, dtype=np.float64)
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    if location.ndim != 2 or location.shape[1] != 2 or location.shape[0] < 1:
        raise ValueError("positions must be an (N >= 1, 2) array")
    if angle.shape != (location.shape[0],):
        raise ValueError(f"phases must have shape ({location.shape[0]},), got {angle.shape}")
    if not np.all(np.isfinite(location)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(angle)):
        raise ValueError("phases must be finite")
    spatial_angle = np.arctan2(location[:, 1], location[:, 0])
    return SwarmalatorOrderParameters(
        s_plus=float(np.abs(np.mean(np.exp(1j * (spatial_angle + angle))))),
        s_minus=float(np.abs(np.mean(np.exp(1j * (spatial_angle - angle))))),
        phase_coherence=float(np.abs(np.mean(np.exp(1j * angle)))),
    )


__all__ = [
    "SwarmalatorOrderParameters",
    "SwarmalatorTrajectory",
    "integrate_swarmalators",
    "swarmalator_field",
    "swarmalator_order_parameters",
]
