# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Symplectic (structure-preserving) inertial Kuramoto integrator
r"""Symplectic structure-preserving integrator for the inertial (power-grid swing) Kuramoto model.

The inertial Kuramoto model ``m θ̈ + γ θ̇ = ω + F(θ)`` is, in the undamped limit ``γ = 0`` with a
potential force ``F = −∇U``, a Hamiltonian system with energy
``E = \tfrac{m}{2}\lVert v\rVert^2 + U(θ) − ω·θ``. A general-purpose integrator such as the RK4 in
:func:`~oscillatools.accel.kuramoto_inertial.integrate_inertial` is accurate per step but
its energy error *drifts secularly* — it grows with integration time — so long power-grid /
Hamiltonian runs slowly gain or lose energy. A **symplectic** integrator instead preserves the
phase-space structure, so its energy error stays *bounded* (it oscillates about a constant) for
exponentially long times, which is the property that matters for long-time swing-equation studies.

This module integrates the model by the velocity-Verlet (leapfrog) scheme, which is symplectic and
second-order for the undamped Hamiltonian flow, with the linear damping handled by a Strang
splitting: a half-step exponential velocity decay, a full velocity-Verlet kick–drift–kick of the
Hamiltonian part, and a second half-step decay. At ``γ = 0`` this is exactly velocity-Verlet (and
exactly symplectic); at ``γ > 0`` it is the structure-preserving extension whose energy dissipates
monotonically towards the phase-locked fixed point. It is second-order accurate, so over short
times it agrees with the RK4 integrator to ``O(dt^2)``; its advantage is the absence of long-time
energy drift, not per-step accuracy.

This is an analysis layer over the synchronisation dynamics — a splitting integrator composing the
polyglot networked force — and adds no compute kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .kuramoto_inertial import InertialTrajectory, PhaseForce, _validate_state


def integrate_symplectic_inertial(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: PhaseForce,
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
) -> InertialTrajectory:
    r"""Integrate the inertial Kuramoto model by a symplectic velocity-Verlet splitting.

    Advances ``θ̇ = v``, ``v̇ = (ω + F(θ) − γ v)/m`` by the damped velocity-Verlet step

    1. ``v ← v\,e^{-γ\,dt/(2m)}`` — half-step exponential damping,
    2. ``v ← v + \tfrac{dt}{2}\,a(θ)``, ``θ ← θ + dt\,v``, ``v ← v + \tfrac{dt}{2}\,a(θ)`` — the
       velocity-Verlet kick–drift–kick of the Hamiltonian part with ``a(θ) = (ω + F(θ))/m``,
    3. ``v ← v\,e^{-γ\,dt/(2m)}`` — second half-step damping.

    At ``γ = 0`` this is exactly the symplectic velocity-Verlet scheme.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (one-dimensional, length ``N``).
    velocities : numpy.ndarray
        The initial velocities ``v(0) = θ̇(0)`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies / power injections ``ω`` (length ``N``).
    force : callable
        The phase-coupling force ``F(θ)`` (see :data:`PhaseForce`).
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float, optional
        The damping ``γ`` (``≥ 0``); defaults to ``1``. ``0`` is the symplectic Hamiltonian limit.
    dt : float
        The Verlet time step (``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    InertialTrajectory
        The sampled phase and velocity trajectory with the ``mass`` and ``damping`` used.

    Raises
    ------
    ValueError
        If the state vectors are mismatched, or ``mass``/``damping``/``dt``/``n_steps`` are out of
        range.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    speed = np.ascontiguousarray(velocities, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate_state(theta, speed, frequencies)
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    count = theta.size
    decay = float(np.exp(-damping * dt / (2.0 * mass)))

    def acceleration(position: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((frequencies + force(position)) / mass, dtype=np.float64)

    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)
    velocity_history = np.empty((n_steps + 1, count), dtype=np.float64)
    phase_history[0] = theta
    velocity_history[0] = speed
    position = theta
    momentum = speed
    for step in range(n_steps):
        momentum = momentum * decay
        momentum = momentum + 0.5 * dt * acceleration(position)
        position = position + dt * momentum
        momentum = momentum + 0.5 * dt * acceleration(position)
        momentum = momentum * decay
        phase_history[step + 1] = position
        velocity_history[step + 1] = momentum
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return InertialTrajectory(times, phase_history, velocity_history, mass, damping)


__all__ = [
    "integrate_symplectic_inertial",
]
