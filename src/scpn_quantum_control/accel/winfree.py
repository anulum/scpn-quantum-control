# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — The Winfree model and the unified Kuramoto–Winfree family
r"""The Winfree pulse-coupled model and the unified Kuramoto–Winfree symmetry-breaking family.

Kuramoto couples oscillators through the phase *difference* ``\sin(\theta_j-\theta_i)``, which is
invariant under a global phase shift. Winfree's earlier model couples them differently — each
oscillator emits a pulse ``P(\theta_j)`` and responds to the mean pulse through its own sensitivity
``R(\theta_i)`` — so the coupling is a *product* ``R(\theta_i)\,\overline{P(\theta)}`` that breaks
the rotational symmetry. With the canonical pulse ``P(\theta)=1+\cos\theta`` and sensitivity
``R(\theta)=-\sin\theta`` this is

.. math::

    \dot\theta_i = \omega_i - \varepsilon\,\sin\theta_i\,\bigl(1 + \langle\cos\theta\rangle\bigr).

A single symmetry-breaking interpolation parameter ``q \in [0, 1]`` joins the two models into one
family (Manoranjani et al., Eur. Phys. J. Plus, 2023): ``q = 0`` is Kuramoto, ``q = 1`` is Winfree,

.. math::

    \dot\theta_i = \omega_i + \varepsilon\Bigl[(1-q)\,\frac1N\sum_j\sin(\theta_j-\theta_i)
        - q\,\sin\theta_i\,\bigl(1 + \langle\cos\theta\rangle\bigr)\Bigr].

As ``q`` grows the rotational symmetry is progressively broken, moving the phase diagram from
Kuramoto partial synchronisation toward the Winfree regimes (oscillator death, partial death,
incoherence). The companion Jacobian — the coupling is all-to-all mean-field, so it is dense —
supports linear-stability and death-boundary analysis. The mean-field field evaluates in ``O(N)``;
it adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WinfreeTrajectory:
    """A Kuramoto–Winfree phase trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    phases : numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phases ``θ(T)``."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)


def _validate(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    interpolation: float,
) -> int:
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phases must be a one-dimensional array of length at least two")
    count = phases.size
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases must be finite")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    if not np.isfinite(coupling):
        raise ValueError(f"coupling must be finite, got {coupling}")
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError(f"interpolation must be in [0, 1], got {interpolation}")
    return count


def _field(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    interpolation: float,
) -> NDArray[np.float64]:
    """The unified Kuramoto–Winfree field (mean-field, ``O(N)``)."""
    mean_field = np.mean(np.exp(1j * phases))
    kuramoto_part = np.imag(mean_field * np.exp(-1j * phases))
    pulse = 1.0 + np.mean(np.cos(phases))
    winfree_part = -np.sin(phases) * pulse
    return np.asarray(
        omega + coupling * ((1.0 - interpolation) * kuramoto_part + interpolation * winfree_part),
        dtype=np.float64,
    )


def winfree_field(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    interpolation: float,
) -> NDArray[np.float64]:
    r"""The unified Kuramoto–Winfree vector field ``θ̇``.

    Parameters
    ----------
    phases : numpy.ndarray
        The phases ``θ`` (one-dimensional, length ``N ≥ 2``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : float
        The coupling strength ``ε``.
    interpolation : float
        The symmetry-breaking interpolation ``q ∈ [0, 1]``: ``0`` is Kuramoto, ``1`` is Winfree.

    Returns
    -------
    numpy.ndarray
        The phase velocities ``θ̇``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate(angle, frequencies, coupling, interpolation)
    return _field(angle, frequencies, coupling, interpolation)


def winfree_jacobian(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    interpolation: float,
) -> NDArray[np.float64]:
    r"""The ``(N, N)`` Jacobian ``∂θ̇/∂θ`` of the unified Kuramoto–Winfree field.

    Parameters
    ----------
    phases, omega, coupling, interpolation
        As for :func:`winfree_field` (the Jacobian is independent of ``ω``; it is validated for
        contract consistency).

    Returns
    -------
    numpy.ndarray
        The dense mean-field Jacobian.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    count = _validate(angle, frequencies, coupling, interpolation)

    cosine = np.cos(angle[None, :] - angle[:, None])
    kuramoto_jacobian = cosine / count
    np.fill_diagonal(kuramoto_jacobian, -(cosine.sum(axis=1) - 1.0) / count)

    pulse = 1.0 + np.mean(np.cos(angle))
    sine = np.sin(angle)
    winfree_jacobian_matrix = np.outer(sine, sine) / count
    np.fill_diagonal(
        winfree_jacobian_matrix, np.diag(winfree_jacobian_matrix) - np.cos(angle) * pulse
    )

    jacobian = coupling * (
        (1.0 - interpolation) * kuramoto_jacobian + interpolation * winfree_jacobian_matrix
    )
    return np.asarray(jacobian, dtype=np.float64)


def integrate_winfree(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    interpolation: float,
    dt: float,
    n_steps: int,
) -> WinfreeTrajectory:
    r"""Integrate the unified Kuramoto–Winfree model with classic RK4.

    Parameters
    ----------
    phases, omega, coupling, interpolation
        As for :func:`winfree_field`.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    WinfreeTrajectory
        The sampled phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    count = _validate(angle, frequencies, coupling, interpolation)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    trajectory = np.empty((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = angle
    current = angle
    for step in range(n_steps):
        k1 = _field(current, frequencies, coupling, interpolation)
        k2 = _field(current + 0.5 * dt * k1, frequencies, coupling, interpolation)
        k3 = _field(current + 0.5 * dt * k2, frequencies, coupling, interpolation)
        k4 = _field(current + dt * k3, frequencies, coupling, interpolation)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return WinfreeTrajectory(times=times, phases=trajectory)


__all__ = [
    "WinfreeTrajectory",
    "integrate_winfree",
    "winfree_field",
    "winfree_jacobian",
]
