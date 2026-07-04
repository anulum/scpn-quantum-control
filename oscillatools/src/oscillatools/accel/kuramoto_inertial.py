# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Inertial (second-order) Kuramoto dynamics, phase-space Jacobian and energy
r"""Inertial (second-order) Kuramoto model — the power-grid swing equation in phase space.

The second-order Kuramoto oscillator carries inertia: ``m θ̈ + γ θ̇ = ω + F(θ)``, the
swing equation of a synchronous machine on a power grid (``m`` the inertia, ``γ`` the
damping, ``ω`` the power injection / natural frequency, ``F`` any Kuramoto phase-coupling
force). Written as a first-order system on the ``(θ, θ̇)`` phase space with velocity
``v = θ̇`` the dynamics are

.. math::

    \dot{θ} = v, \qquad \dot{v} = \frac{1}{m}\bigl(ω + F(θ) - γ\,v\bigr),

so the phase-space vector field is a map :math:`\mathbb{R}^{2N} \to \mathbb{R}^{2N}` whose
Jacobian is the block matrix

.. math::

    \begin{pmatrix} 0 & I \\ J_F(θ)/m & -(γ/m)\,I \end{pmatrix},

with :math:`J_F = ∂F/∂θ` the force Jacobian of the chosen Kuramoto model. As the inertia
vanishes (:math:`m \to 0`) the velocity is slaved to ``(ω + F(θ))/γ`` and the model
reduces to the first-order Kuramoto flow ``θ̇ = (ω + F(θ))/γ`` (the overdamped limit).

When the force derives from a potential, :math:`F = -∇U` (the symmetric-coupling case where
``U`` is the Kuramoto interaction energy), the mechanical energy

.. math::

    E(θ, v) = \tfrac{m}{2}\lVert v \rVert^2 + U(θ) - ω·θ

is a Lyapunov function for the damped flow: :math:`\dot{E} = -γ\lVert v \rVert^2 \le 0`, so
the kinetic energy is dissipated and (for ``ω = 0``) the system relaxes monotonically to a
phase-locked fixed point. The undamped limit ``γ = 0`` conserves ``E`` (a Hamiltonian flow).

This is an analysis layer over the synchronisation dynamics: the phase-space flow is a
fixed-step RK4 composing any of the polyglot Kuramoto forces / Jacobians (the mean-field,
triadic or networked forces re-exported from :mod:`oscillatools.accel`) and the
interaction energy, so the module adds no compute kernel. The integrator is written as a
pure-functional RK4 over the concatenated state, composing analytic forces and Jacobians, so
gradients propagate through it where the underlying force is differentiable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: A phase-coupling force ``F(θ)`` already closed over its coupling parameters; it maps the
#: phase vector to a force vector of the same length. Bind a Kuramoto force to its coupling
#: to obtain one, e.g. ``lambda theta: mean_field_force(theta, K)`` or
#: ``lambda theta: networked_kuramoto_force(theta, coupling_matrix)``.
PhaseForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: The force Jacobian ``∂F/∂θ`` of a :data:`PhaseForce`, returning an ``(N, N)`` matrix; the
#: bound counterpart of e.g. :func:`~oscillatools.accel.kuramoto_mean_field.mean_field_jacobian`.
PhaseJacobian = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: A phase potential ``U(θ)`` whose negative gradient is the force, ``F = -∇U`` — the
#: symmetric-coupling :func:`~oscillatools.accel.kuramoto_energy.kuramoto_interaction_energy`
#: bound to its coupling. Used to form the mechanical-energy Lyapunov function.
PhasePotential = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True)
class InertialTrajectory:
    """A phase-space trajectory of the inertial Kuramoto model sampled at every RK4 step.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(T + 1,)`` sample times ``0, dt, …, T·dt``.
    phases : numpy.ndarray
        The ``(T + 1, N)`` phase trajectory ``θ(t)``.
    velocities : numpy.ndarray
        The ``(T + 1, N)`` velocity trajectory ``v(t) = θ̇(t)``.
    mass : float
        The inertia ``m`` used to generate the trajectory.
    damping : float
        The damping ``γ`` used to generate the trajectory.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    velocities: NDArray[np.float64]
    mass: float
    damping: float

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phase vector of the trajectory."""
        return np.asarray(self.phases[-1], dtype=np.float64)

    @property
    def terminal_velocities(self) -> NDArray[np.float64]:
        """The final velocity vector of the trajectory."""
        return np.asarray(self.velocities[-1], dtype=np.float64)

    def kinetic_energy(self) -> NDArray[np.float64]:
        """Return the kinetic energy ``(m/2)‖v‖²`` at each sample time (length ``T + 1``)."""
        return np.asarray(
            0.5 * self.mass * np.einsum("tj,tj->t", self.velocities, self.velocities),
            dtype=np.float64,
        )

    def energy(self, omega: NDArray[np.float64], potential: PhasePotential) -> NDArray[np.float64]:
        """Return the mechanical energy ``E = (m/2)‖v‖² + U(θ) − ω·θ`` at each sample time.

        ``potential`` must be the phase potential ``U`` of the trajectory's force
        (``F = −∇U``); see :func:`inertial_energy`.
        """
        frequencies = np.ascontiguousarray(omega, dtype=np.float64)
        return np.asarray(
            [
                inertial_energy(
                    self.phases[step], self.velocities[step], frequencies, potential, self.mass
                )
                for step in range(self.times.size)
            ],
            dtype=np.float64,
        )


def _validate_state(
    phases: NDArray[np.float64], velocities: NDArray[np.float64], omega: NDArray[np.float64]
) -> None:
    """Validate that ``phases``, ``velocities`` and ``omega`` are matching 1-D vectors."""
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    if phases.shape != omega.shape:
        raise ValueError(f"phases must have shape {omega.shape}, got {phases.shape}")
    if velocities.shape != omega.shape:
        raise ValueError(f"velocities must have shape {omega.shape}, got {velocities.shape}")


def inertial_vector_field(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: PhaseForce,
    mass: float,
    *,
    damping: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return the ``(θ, θ̇)`` phase-space vector field of the inertial Kuramoto model.

    For the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` the first-order phase-space flow is
    ``θ̇ = v`` and ``v̇ = (ω + F(θ) − γ v)/m``. The return is the concatenation
    ``[θ̇, v̇]`` of length ``2N``.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    velocities : numpy.ndarray
        The velocity vector ``v = θ̇`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies / power injections ``ω`` (length ``N``).
    force : callable
        The phase-coupling force ``F(θ)`` (see :data:`PhaseForce`).
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float, optional
        The damping ``γ`` (``≥ 0``); defaults to ``1`` to match ``m θ̈ + θ̇ = ω + F``.

    Returns
    -------
    numpy.ndarray
        The ``2N`` phase-space derivative ``[θ̇, v̇]``.

    Raises
    ------
    ValueError
        If the state vectors are mismatched or ``mass``/``damping`` are out of range.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    speed = np.ascontiguousarray(velocities, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate_state(theta, speed, frequencies)
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    acceleration = (frequencies + force(theta) - damping * speed) / mass
    return np.concatenate([speed, acceleration])


def inertial_jacobian(
    phases: NDArray[np.float64],
    force_jacobian: PhaseJacobian,
    mass: float,
    *,
    damping: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return the ``2N × 2N`` Jacobian of the inertial Kuramoto phase-space vector field.

    The block structure is

    .. math::

        \begin{pmatrix} 0 & I \\ J_F(θ)/m & -(γ/m)\,I \end{pmatrix},

    with :math:`J_F = ∂F/∂θ` evaluated at ``phases``. The velocity does not enter ``J_F``,
    so the Jacobian depends on the state only through the phases.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    force_jacobian : callable
        The force Jacobian ``∂F/∂θ`` returning an ``(N, N)`` matrix (see :data:`PhaseJacobian`).
    mass : float
        The inertia ``m`` (``> 0``).
    damping : float, optional
        The damping ``γ`` (``≥ 0``); defaults to ``1``.

    Returns
    -------
    numpy.ndarray
        The ``(2N, 2N)`` phase-space Jacobian.

    Raises
    ------
    ValueError
        If ``phases`` is not a non-empty 1-D vector, ``mass``/``damping`` are out of range,
        or ``force_jacobian`` does not return an ``(N, N)`` matrix.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    count = theta.size
    block = np.ascontiguousarray(force_jacobian(theta), dtype=np.float64)
    if block.shape != (count, count):
        raise ValueError(
            f"force_jacobian must return an ({count}, {count}) matrix, got shape {block.shape}"
        )
    identity = np.eye(count, dtype=np.float64)
    zeros = np.zeros((count, count), dtype=np.float64)
    top = np.hstack([zeros, identity])
    bottom = np.hstack([block / mass, -(damping / mass) * identity])
    return np.ascontiguousarray(np.vstack([top, bottom]), dtype=np.float64)


def integrate_inertial(
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
    r"""Integrate the inertial Kuramoto model by a fixed-step RK4 over the ``(θ, v)`` state.

    The concatenated state ``z = [θ, v]`` is advanced by the classical four-stage Runge–Kutta
    rule applied to :func:`inertial_vector_field`, sampling ``θ`` and ``v`` at every step. The
    update is written functionally (no in-place mutation of the running state), so gradients
    propagate through the integrator where ``force`` is differentiable.

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
        The damping ``γ`` (``≥ 0``); defaults to ``1``.
    dt : float
        The RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    InertialTrajectory
        The sampled phase and velocity trajectory with the ``mass`` and ``damping`` used.

    Raises
    ------
    ValueError
        If the state vectors are mismatched, or ``mass``/``damping``/``dt``/``n_steps`` are
        out of range.
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

    def rhs(state: NDArray[np.float64]) -> NDArray[np.float64]:
        acceleration = (frequencies + force(state[:count]) - damping * state[count:]) / mass
        return np.concatenate([state[count:], acceleration])

    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)
    velocity_history = np.empty((n_steps + 1, count), dtype=np.float64)
    phase_history[0] = theta
    velocity_history[0] = speed
    current = np.concatenate([theta, speed])
    for step in range(n_steps):
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * dt * k1)
        k3 = rhs(current + 0.5 * dt * k2)
        k4 = rhs(current + dt * k3)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        phase_history[step + 1] = current[:count]
        velocity_history[step + 1] = current[count:]
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return InertialTrajectory(times, phase_history, velocity_history, mass, damping)


def inertial_energy(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    potential: PhasePotential,
    mass: float,
) -> float:
    r"""Return the mechanical energy ``E = (m/2)‖v‖² + U(θ) − ω·θ`` of an inertial state.

    For a force deriving from a potential (``F = −∇U``, the symmetric-coupling case) this is a
    Lyapunov function of the damped flow: along a trajectory :math:`\dot{E} = -γ\lVert v
    \rVert^2 \le 0`, so the inertial Kuramoto model dissipates energy and relaxes (for
    ``ω = 0``) to a phase-locked fixed point. ``potential`` must be the interaction energy
    ``U`` bound to the same coupling as the force, e.g.
    ``lambda theta: kuramoto_interaction_energy(theta, K)``.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    velocities : numpy.ndarray
        The velocity vector ``v = θ̇`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    potential : callable
        The phase potential ``U(θ)`` with ``F = −∇U`` (see :data:`PhasePotential`).
    mass : float
        The inertia ``m`` (``> 0``).

    Returns
    -------
    float
        The mechanical energy of the state.

    Raises
    ------
    ValueError
        If the state vectors are mismatched or ``mass`` is not positive.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    speed = np.ascontiguousarray(velocities, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate_state(theta, speed, frequencies)
    if mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    kinetic = 0.5 * mass * float(np.dot(speed, speed))
    return kinetic + float(potential(theta)) - float(np.dot(frequencies, theta))


__all__ = [
    "InertialTrajectory",
    "PhaseForce",
    "PhaseJacobian",
    "PhasePotential",
    "inertial_energy",
    "inertial_jacobian",
    "inertial_vector_field",
    "integrate_inertial",
]
