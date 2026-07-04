# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Nonlocally coupled Stuart–Landau amplitude oscillators
r"""Nonlocally coupled Stuart–Landau amplitude oscillators and their death / chimera regimes.

The Kuramoto family is *phase only*: every oscillator sits on a fixed-radius limit cycle and only its
phase evolves. The Stuart–Landau oscillator instead carries a full complex amplitude ``z = r e^{iφ}``
— the normal form of a supercritical Hopf bifurcation — and so admits whole dynamical regimes a phase
model structurally cannot represent: amplitude variation, **amplitude death**, **oscillation death**
and **amplitude chimeras**, where the *amplitude* (not just the phase) becomes spatially structured.

This module integrates the nonlocally coupled ensemble on a ring with the symmetry-breaking coupling
of Zakharova/Kapeller/Schöll (PRL 112, 154101, 2014) — the coupling acts through the *real parts*
only, breaking the rotational ``S^1`` symmetry that is what makes oscillation/chimera death possible:

.. math::

    \dot z_j = (\lambda + i\omega - |z_j|^2)\,z_j
        + \frac{\sigma}{2P}\sum_{|k-j|\le P}\bigl(\operatorname{Re} z_k - \operatorname{Re} z_j\bigr).

Uncoupled (``\sigma = 0``) each oscillator relaxes to the limit cycle of radius ``\sqrt\lambda``; with
strong nonlocal coupling the ensemble collapses to an *inhomogeneous steady state* (oscillation
death) — oscillations cease while the amplitudes remain spatially patterned, the hallmark the
phase-only toolkit cannot reach. The companion Jacobian (in real ``[\operatorname{Re}; \operatorname{Im}]``
coordinates, since the symmetry-breaking coupling is not holomorphic) supports linear-stability and
death-boundary analysis. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StuartLandauTrajectory:
    """A Stuart–Landau ensemble trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    states : numpy.ndarray
        The ``(n_steps + 1, N)`` complex amplitudes ``z(t)``.
    """

    times: NDArray[np.float64]
    states: NDArray[np.complex128]

    @property
    def terminal_state(self) -> NDArray[np.complex128]:
        """The final complex amplitudes ``z(T)``."""
        return np.ascontiguousarray(self.states[-1], dtype=np.complex128)


def _ring_realpart_coupling(real: NDArray[np.float64], radius: int) -> NDArray[np.float64]:
    """Return ``Σ_{|k−j|≤P, k≠j}(Re z_k − Re z_j)`` for each ``j`` on a ring."""
    accumulated = np.zeros_like(real)
    for shift in range(1, radius + 1):
        accumulated += np.roll(real, -shift) + np.roll(real, shift)
    return accumulated - 2.0 * radius * real


def _validate(
    count: int,
    omega: float,
    coupling: float,
    radius: int,
    dt: float | None,
) -> None:
    if count < 2:
        raise ValueError(f"the ensemble needs at least two oscillators, got {count}")
    if radius < 1 or 2 * radius > count - 1:
        raise ValueError(f"radius must be in [1, {(count - 1) // 2}], got {radius}")
    if not np.isfinite(omega):
        raise ValueError(f"omega must be finite, got {omega}")
    if not np.isfinite(coupling):
        raise ValueError(f"coupling must be finite, got {coupling}")
    if dt is not None and dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")


def stuart_landau_field(
    state: NDArray[np.complex128],
    lam: float,
    omega: float,
    coupling: float,
    radius: int,
) -> NDArray[np.complex128]:
    r"""The nonlocal symmetry-breaking Stuart–Landau vector field ``ż``.

    Parameters
    ----------
    state : numpy.ndarray
        The complex amplitudes ``z`` (one-dimensional, length ``N ≥ 2``).
    lam : float
        The Hopf parameter ``λ`` (``> 0`` gives a limit cycle of radius ``√λ``).
    omega : float
        The oscillation frequency ``ω``.
    coupling : float
        The coupling strength ``σ``.
    radius : int
        The ring coupling radius ``P`` (``1 ≤ P ≤ (N−1)/2``).

    Returns
    -------
    numpy.ndarray
        The complex time derivative ``ż``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    amplitude = np.ascontiguousarray(state, dtype=np.complex128)
    _validate(amplitude.size, omega, coupling, radius, None)
    local = (lam + 1j * omega - np.abs(amplitude) ** 2) * amplitude
    interaction = (coupling / (2.0 * radius)) * _ring_realpart_coupling(np.real(amplitude), radius)
    return np.asarray(local + interaction, dtype=np.complex128)


def stuart_landau_jacobian(
    state: NDArray[np.complex128],
    lam: float,
    omega: float,
    coupling: float,
    radius: int,
) -> NDArray[np.float64]:
    r"""The ``(2N, 2N)`` real Jacobian in ``[Re z; Im z]`` coordinates.

    Parameters
    ----------
    state, lam, omega, coupling, radius
        As for :func:`stuart_landau_field`.

    Returns
    -------
    numpy.ndarray
        The Jacobian of ``[Re ż; Im ż]`` with respect to ``[Re z; Im z]``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    amplitude = np.ascontiguousarray(state, dtype=np.complex128)
    count = amplitude.size
    _validate(count, omega, coupling, radius, None)
    x = np.real(amplitude)
    y = np.imag(amplitude)
    jacobian = np.zeros((2 * count, 2 * count), dtype=np.float64)
    diagonal = np.arange(count)
    jacobian[diagonal, diagonal] = lam - (3.0 * x * x + y * y) - coupling
    jacobian[diagonal, count + diagonal] = -omega - 2.0 * x * y
    jacobian[count + diagonal, diagonal] = omega - 2.0 * x * y
    jacobian[count + diagonal, count + diagonal] = lam - (x * x + 3.0 * y * y)
    weight = coupling / (2.0 * radius)
    for shift in range(1, radius + 1):
        for node in range(count):
            jacobian[node, (node + shift) % count] += weight
            jacobian[node, (node - shift) % count] += weight
    return jacobian


def integrate_stuart_landau(
    state: NDArray[np.complex128],
    lam: float,
    omega: float,
    coupling: float,
    radius: int,
    dt: float,
    n_steps: int,
) -> StuartLandauTrajectory:
    r"""Integrate the Stuart–Landau ensemble with classic RK4.

    Parameters
    ----------
    state : numpy.ndarray
        The initial complex amplitudes ``z(0)`` (length ``N ≥ 2``).
    lam, omega, coupling, radius
        As for :func:`stuart_landau_field`.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    StuartLandauTrajectory
        The sampled complex trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    amplitude = np.ascontiguousarray(state, dtype=np.complex128)
    _validate(amplitude.size, omega, coupling, radius, dt)
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    def field(value: NDArray[np.complex128]) -> NDArray[np.complex128]:
        local = (lam + 1j * omega - np.abs(value) ** 2) * value
        interaction = (coupling / (2.0 * radius)) * _ring_realpart_coupling(np.real(value), radius)
        return np.asarray(local + interaction, dtype=np.complex128)

    trajectory = np.empty((n_steps + 1, amplitude.size), dtype=np.complex128)
    trajectory[0] = amplitude
    current = amplitude
    for step in range(n_steps):
        k1 = field(current)
        k2 = field(current + 0.5 * dt * k1)
        k3 = field(current + 0.5 * dt * k2)
        k4 = field(current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return StuartLandauTrajectory(times=times, states=trajectory)


def amplitudes(state: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the per-oscillator amplitudes ``|z_j|``."""
    return np.ascontiguousarray(np.abs(np.asarray(state, dtype=np.complex128)), dtype=np.float64)


def stuart_landau_order_parameter(state: NDArray[np.complex128]) -> complex:
    r"""Return the Kuramoto-style phase order parameter ``(1/N) Σ_j e^{i\arg z_j}``.

    A dead oscillator (``z_j = 0``) is taken to have phase ``0``.
    """
    phases = np.angle(np.asarray(state, dtype=np.complex128))
    return complex(np.mean(np.exp(1j * phases)))


def is_oscillation_death(
    state: NDArray[np.complex128],
    lam: float,
    omega: float,
    coupling: float,
    radius: int,
    *,
    velocity_tolerance: float = 1e-3,
) -> bool:
    r"""Whether ``state`` is an oscillation-death steady state (``‖ż‖ < velocity_tolerance``).

    Parameters
    ----------
    state, lam, omega, coupling, radius
        As for :func:`stuart_landau_field`.
    velocity_tolerance : float, optional
        The Euclidean velocity-norm threshold below which the state counts as a steady state
        (``> 0``); defaults to ``1e-3``.

    Returns
    -------
    bool
        ``True`` if the ensemble has collapsed to a steady state.

    Raises
    ------
    ValueError
        If ``velocity_tolerance`` is non-positive (other arguments are validated by the field).
    """
    if velocity_tolerance <= 0.0:
        raise ValueError(f"velocity_tolerance must be positive, got {velocity_tolerance}")
    derivative = stuart_landau_field(state, lam, omega, coupling, radius)
    speed = float(np.sqrt(np.sum(np.abs(derivative) ** 2)))
    return speed < velocity_tolerance


__all__ = [
    "StuartLandauTrajectory",
    "amplitudes",
    "integrate_stuart_landau",
    "is_oscillation_death",
    "stuart_landau_field",
    "stuart_landau_jacobian",
    "stuart_landau_order_parameter",
]
