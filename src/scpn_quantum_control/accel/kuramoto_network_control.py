# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Indirect-adjoint optimal control of the full oscillator network
r"""Indirect-adjoint (Pontryagin) optimal control of the full Kuramoto network.

Where :mod:`scpn_quantum_control.accel.kuramoto_collective_control` controls the *macroscopic*
Ott–Antonsen order parameter, this module controls the *full* network directly: each oscillator
receives a time-dependent control input ``u_i(t)`` and

.. math::

    \dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i) + u_i(t).

The control objective is the reference-trajectory-free desynchronisation functional — a running
penalty on the order-parameter magnitude plus a control-energy term,

.. math::

    J = \sum_k r(\theta_k)^2\,\mathrm dt + w \sum_k \lVert u_k\rVert^2\,\mathrm dt ,

so minimising ``J`` drives the population apart (``r \to 0``) at least control cost. This is the
indirect (Pontryagin) optimal-control formulation: the gradient of ``J`` with respect to the whole
control series is the discrete adjoint (costate) of the RK4 map, integrated backwards with the
running-cost injected at every state. It reuses the toolkit's shipped networked force and Jacobian
and the order-parameter observable + its gradient as the exact cost and dynamics substrate, so the
control machinery is a thin wrapper over already-tested kernels and matches finite differences to
machine precision. It adds no compute kernel.

The Riccati-feedback (SDRE) companion is intentionally a separate concern (a state-feedback law, not
a gradient-based open-loop optimiser) and is tracked separately; this module is the open-loop
adjoint optimal control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian
from .order_parameter_observables import order_parameter, order_parameter_gradient


@dataclass(frozen=True)
class ControlledNetworkTrajectory:
    """A controlled Kuramoto-network phase trajectory.

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


@dataclass(frozen=True)
class NetworkControlGradients:
    """The desynchronisation objective value and its gradients.

    Attributes
    ----------
    cost : float
        The objective ``Σ_k r(θ_k)² dt + w Σ_k ‖u_k‖² dt``.
    control_gradient : numpy.ndarray
        The ``(n_steps, N)`` gradient with respect to the control series; a steepest-descent step is
        ``u ← u − η · control_gradient``.
    initial_phase_gradient : numpy.ndarray
        The ``(N,)`` gradient with respect to the initial phases ``θ(0)``.
    """

    cost: float
    control_gradient: NDArray[np.float64]
    initial_phase_gradient: NDArray[np.float64]


def _field(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The controlled Kuramoto field ``ω + F(θ) + u``."""
    return np.asarray(
        omega + networked_kuramoto_force(phases, coupling) + control, dtype=np.float64
    )


def _step_with_jacobians(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance one RK4 step and return ``(next_phases, ∂next/∂phases, ∂next/∂control)``.

    The control enters additively, so ``∂field/∂control`` is the identity; ``∂field/∂phases`` is the
    shipped networked Jacobian.
    """
    count = phases.size
    identity = np.eye(count, dtype=np.float64)

    stage1 = _field(phases, control, omega, coupling)
    d1_phase = networked_kuramoto_jacobian(phases, coupling)
    d1_control = identity

    point2 = phases + 0.5 * dt * stage1
    stage2 = _field(point2, control, omega, coupling)
    jac2 = networked_kuramoto_jacobian(point2, coupling)
    d2_phase = jac2 @ (identity + 0.5 * dt * d1_phase)
    d2_control = jac2 @ (0.5 * dt * d1_control) + identity

    point3 = phases + 0.5 * dt * stage2
    stage3 = _field(point3, control, omega, coupling)
    jac3 = networked_kuramoto_jacobian(point3, coupling)
    d3_phase = jac3 @ (identity + 0.5 * dt * d2_phase)
    d3_control = jac3 @ (0.5 * dt * d2_control) + identity

    point4 = phases + dt * stage3
    stage4 = _field(point4, control, omega, coupling)
    jac4 = networked_kuramoto_jacobian(point4, coupling)
    d4_phase = jac4 @ (identity + dt * d3_phase)
    d4_control = jac4 @ (dt * d3_control) + identity

    next_phases = phases + (dt / 6.0) * (stage1 + 2.0 * stage2 + 2.0 * stage3 + stage4)
    d_phase = identity + (dt / 6.0) * (d1_phase + 2.0 * d2_phase + 2.0 * d3_phase + d4_phase)
    d_control = (dt / 6.0) * (d1_control + 2.0 * d2_control + 2.0 * d3_control + d4_control)
    return next_phases, d_phase, d_control


def _validate(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> None:
    count = phases.size
    if phases.ndim != 1 or count < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if control.ndim != 2 or control.shape[1] != count or control.shape[0] < 1:
        raise ValueError(f"control must have shape (n_steps, {count}), got {control.shape}")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")


def integrate_controlled_network(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> ControlledNetworkTrajectory:
    r"""Integrate the controlled Kuramoto network ``θ̇ = ω + F(θ) + u`` with RK4.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    control : numpy.ndarray
        The ``(n_steps, N)`` control series ``u_k`` applied on step ``k``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step (``> 0``).

    Returns
    -------
    ControlledNetworkTrajectory
        The ``(n_steps + 1, N)`` phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    state = np.ascontiguousarray(phases, dtype=np.float64)
    series = np.ascontiguousarray(control, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate(state, series, frequencies, matrix, dt)
    n_steps = series.shape[0]
    trajectory = np.empty((n_steps + 1, state.size), dtype=np.float64)
    trajectory[0] = state
    for index in range(n_steps):
        state = _step_with_jacobians(state, series[index], frequencies, matrix, dt)[0]
        trajectory[index + 1] = state
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return ControlledNetworkTrajectory(times=times, phases=trajectory)


def network_control_value_and_grad(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    *,
    control_weight: float,
) -> NetworkControlGradients:
    r"""Evaluate the desynchronisation control objective and its exact adjoint gradients.

    The objective ``J = Σ_k r(θ_k)² dt + w Σ_k ‖u_k‖² dt`` runs an order-parameter (desynchronisation)
    penalty over the trajectory plus control energy. The gradient with respect to the whole control
    series is the discrete adjoint (costate) of the RK4 map, integrated backwards with the running
    order-parameter cost injected at every state — matching finite differences to machine precision.

    Parameters
    ----------
    phases, control, omega, coupling, dt
        As for :func:`integrate_controlled_network`.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).

    Returns
    -------
    NetworkControlGradients
        The cost and the control / initial-phase gradients.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    state = np.ascontiguousarray(phases, dtype=np.float64)
    series = np.ascontiguousarray(control, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate(state, series, frequencies, matrix, dt)
    if control_weight < 0.0:
        raise ValueError(f"control_weight must be non-negative, got {control_weight}")
    n_steps, count = series.shape

    trajectory = np.empty((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = state
    phase_jacobians = np.empty((n_steps, count, count), dtype=np.float64)
    control_jacobians = np.empty((n_steps, count, count), dtype=np.float64)
    for index in range(n_steps):
        state, d_phase, d_control = _step_with_jacobians(
            state, series[index], frequencies, matrix, dt
        )
        trajectory[index + 1] = state
        phase_jacobians[index] = d_phase
        control_jacobians[index] = d_control

    radius = np.array([order_parameter(trajectory[k + 1]) for k in range(n_steps)])
    cost = float(np.sum(radius**2) * dt + control_weight * float(np.sum(series**2)) * dt)

    adjoint = np.zeros(count, dtype=np.float64)
    control_gradient = np.empty((n_steps, count), dtype=np.float64)
    for index in range(n_steps - 1, -1, -1):
        running = 2.0 * radius[index] * order_parameter_gradient(trajectory[index + 1]) * dt
        adjoint = adjoint + running
        control_gradient[index] = (
            control_jacobians[index].T @ adjoint + 2.0 * control_weight * series[index] * dt
        )
        adjoint = phase_jacobians[index].T @ adjoint

    return NetworkControlGradients(
        cost=cost,
        control_gradient=control_gradient,
        initial_phase_gradient=adjoint,
    )


def optimise_network_control(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    control_weight: float,
    learning_rate: float,
    n_iterations: int,
    initial_control: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Desynchronise the network by gradient descent on the control series.

    Runs ``n_iterations`` steepest-descent updates ``u ← u − η ∇_u J`` using
    :func:`network_control_value_and_grad`, returning the optimised control and the cost history.

    Parameters
    ----------
    phases, omega, coupling, dt
        As for :func:`integrate_controlled_network`.
    n_steps : int
        The control-horizon length (``≥ 1``).
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    learning_rate : float
        The gradient-descent step size ``η`` (``> 0``).
    n_iterations : int
        The number of descent iterations (``≥ 1``).
    initial_control : numpy.ndarray, optional
        The starting ``(n_steps, N)`` control; defaults to zeros (uncontrolled).

    Returns
    -------
    tuple of numpy.ndarray
        The optimised ``(n_steps, N)`` control and the ``(n_iterations + 1,)`` cost history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    state = np.ascontiguousarray(phases, dtype=np.float64)
    count = state.size
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    control: NDArray[np.float64]
    if initial_control is None:
        control = np.zeros((n_steps, count), dtype=np.float64)
    else:
        control = np.ascontiguousarray(initial_control, dtype=np.float64)
        if control.shape != (n_steps, count):
            raise ValueError(
                f"initial_control must have shape ({n_steps}, {count}), got {control.shape}"
            )

    history = np.empty(n_iterations + 1, dtype=np.float64)
    for iteration in range(n_iterations):
        gradients = network_control_value_and_grad(
            state, control, omega, coupling, dt, control_weight=control_weight
        )
        history[iteration] = gradients.cost
        control = control - learning_rate * gradients.control_gradient
    final = network_control_value_and_grad(
        state, control, omega, coupling, dt, control_weight=control_weight
    )
    history[n_iterations] = final.cost
    return np.ascontiguousarray(control, dtype=np.float64), history


__all__ = [
    "ControlledNetworkTrajectory",
    "NetworkControlGradients",
    "integrate_controlled_network",
    "network_control_value_and_grad",
    "optimise_network_control",
]
