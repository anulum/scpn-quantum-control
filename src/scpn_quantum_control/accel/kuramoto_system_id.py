# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Kuramoto system identification (learn K from trajectories)
"""Differentiable Kuramoto system identification — learn the coupling from observed trajectories.

The inverse of simulation: given a phase trajectory observed at a set of times, fit the coupling
matrix ``K`` that reproduces it. The trajectory-match loss
``L = Σ_m ‖θ(t_m) − θ_obs(t_m)‖²`` is differentiated through the integrator by accumulating one
adjoint per observation: the network is integrated once, then for each observation step the
reverse-mode adjoint runs on the trajectory prefix up to that step with the residual cotangent
``2(θ(t_m) − θ_obs(t_m))``, and the per-observation gradients are summed. ``learn_coupling`` then
descends this loss to recover ``K`` from data.

This is a control-application layer over the differentiable simulation (it reuses the integrators
and their adjoint on trajectory prefixes) and adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_euler import kuramoto_euler_trajectory, kuramoto_euler_vjp
from .diff_kuramoto_rk4 import kuramoto_rk4_trajectory, kuramoto_rk4_vjp

_GradResult = tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class SystemIdentificationResult:
    """Outcome of a learn-coupling run.

    Attributes
    ----------
    coupling : numpy.ndarray
        The learned ``(N, N)`` coupling matrix.
    cost_history : tuple of float
        The trajectory-match loss at the start of each accepted iteration (monotonically
        non-increasing).
    iterations : int
        The number of iterations performed.
    converged : bool
        ``True`` if the line search could no longer reduce the loss.
    """

    coupling: NDArray[np.float64]
    cost_history: tuple[float, ...]
    iterations: int
    converged: bool


def _validate(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    observed: NDArray[np.float64],
    observation_steps: NDArray[np.int64],
    n_steps: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    targets = np.ascontiguousarray(observed, dtype=np.float64)
    steps = np.ascontiguousarray(observation_steps, dtype=np.int64)
    count = phases.size
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if steps.ndim != 1:
        raise ValueError(f"observation_steps must be one-dimensional, got shape {steps.shape}")
    if targets.shape != (steps.size, count):
        raise ValueError(f"observed must have shape ({steps.size}, {count}), got {targets.shape}")
    if steps.size == 0 or np.any(steps < 0) or np.any(steps > n_steps):
        raise ValueError(f"observation_steps must be non-empty within [0, {n_steps}]")
    return phases, frequencies, matrix, targets, steps


def trajectory_match_value_and_grad(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    observed: NDArray[np.float64],
    observation_steps: NDArray[np.int64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> _GradResult:
    r"""Trajectory-match loss ``Σ_m ‖θ(t_m) − θ_obs(t_m)‖²`` and its gradient.

    Integrates the network once and backpropagates the sum-of-squared residuals at the observed
    steps to the initial phases, the natural frequencies and the coupling matrix.

    Parameters
    ----------
    theta0, omega, coupling : numpy.ndarray
        The ``(N,)`` initial phases, ``(N,)`` frequencies and ``(N, N)`` coupling.
    observed : numpy.ndarray
        The ``(M, N)`` observed phases at the observation steps.
    observation_steps : numpy.ndarray
        The ``(M,)`` integer step indices in ``[0, n_steps]`` at which ``observed`` is recorded.
    dt : float
        The integrator step size.
    n_steps : int
        The number of integration steps.
    integrator : str, optional
        ``"rk4"`` (default) or ``"euler"``.

    Returns
    -------
    tuple
        ``(loss, grad_theta0, grad_omega, grad_coupling)``.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or ``integrator`` is invalid.
    """
    phases, frequencies, matrix, targets, steps = _validate(
        theta0, omega, coupling, observed, observation_steps, n_steps
    )
    count = phases.size
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(phases, frequencies, matrix, dt, n_steps)
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(phases, frequencies, matrix, dt, n_steps)
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    loss = 0.0
    grad_theta0 = np.zeros(count, dtype=np.float64)
    grad_omega = np.zeros(count, dtype=np.float64)
    grad_coupling = np.zeros((count, count), dtype=np.float64)
    for index, step in enumerate(steps):
        residual = trajectory[step] - targets[index]
        loss += float(residual @ residual)
        cotangent = np.ascontiguousarray(2.0 * residual, dtype=np.float64)
        prefix = np.ascontiguousarray(trajectory[: int(step) + 1], dtype=np.float64)
        if integrator == "rk4":
            grad_t0, grad_w, grad_k = kuramoto_rk4_vjp(prefix, frequencies, matrix, dt, cotangent)
        else:
            grad_t0, grad_w, grad_k = kuramoto_euler_vjp(prefix, matrix, dt, cotangent)
        grad_theta0 += grad_t0
        grad_omega += grad_w
        grad_coupling += grad_k
    return loss, grad_theta0, grad_omega, grad_coupling


def trajectory_match_value(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    observed: NDArray[np.float64],
    observation_steps: NDArray[np.int64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> float:
    """Trajectory-match loss ``Σ_m ‖θ(t_m) − θ_obs(t_m)‖²`` without its gradient.

    The value-only companion of :func:`trajectory_match_value_and_grad` (used for the learning
    line search). Parameters and the ``integrator`` choice are as there.
    """
    phases, frequencies, matrix, targets, steps = _validate(
        theta0, omega, coupling, observed, observation_steps, n_steps
    )
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(phases, frequencies, matrix, dt, n_steps)
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(phases, frequencies, matrix, dt, n_steps)
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    loss = 0.0
    for index, step in enumerate(steps):
        residual = trajectory[step] - targets[index]
        loss += float(residual @ residual)
    return loss


def learn_coupling(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    observed: NDArray[np.float64],
    observation_steps: NDArray[np.int64],
    coupling_init: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
    max_iterations: int = 200,
    learning_rate: float = 1.0,
    symmetric: bool = True,
    tolerance: float = 1e-12,
    min_step: float = 1e-10,
) -> SystemIdentificationResult:
    r"""Learn the coupling matrix that reproduces an observed trajectory.

    Minimises the trajectory-match loss over ``K`` by gradient descent with a backtracking line
    search (monotone). If ``symmetric`` the coupling is symmetrised with a zero diagonal each
    step (the gradient is symmetrised too), restricting the search to undirected networks.

    Parameters
    ----------
    theta0, omega : numpy.ndarray
        The ``(N,)`` initial phases and natural frequencies (known).
    observed : numpy.ndarray
        The ``(M, N)`` observed phases at the observation steps.
    observation_steps : numpy.ndarray
        The ``(M,)`` integer step indices in ``[0, n_steps]``.
    coupling_init : numpy.ndarray
        The ``(N, N)`` initial coupling guess.
    dt : float
        The integrator step size.
    n_steps : int
        The number of integration steps.
    integrator : str, optional
        ``"rk4"`` (default) or ``"euler"``.
    max_iterations : int, optional
        The iteration budget.
    learning_rate : float, optional
        The initial line-search step.
    symmetric : bool, optional
        Constrain the learned coupling to be symmetric with a zero diagonal.
    tolerance : float, optional
        The minimum loss decrease for a step to be accepted.
    min_step : float, optional
        The smallest line-search step before convergence is declared.

    Returns
    -------
    SystemIdentificationResult
        The learned coupling, the monotone loss history, the iteration count and the convergence
        flag.

    Raises
    ------
    ValueError
        If ``max_iterations`` is not positive or the inputs are inconsistent.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")

    def constrain(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        if not symmetric:
            return matrix
        constrained = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(constrained, 0.0)
        return np.ascontiguousarray(constrained, dtype=np.float64)

    def loss_only(matrix: NDArray[np.float64]) -> float:
        return trajectory_match_value(
            theta0, omega, matrix, observed, observation_steps, dt, n_steps, integrator=integrator
        )

    current = constrain(np.ascontiguousarray(coupling_init, dtype=np.float64))
    history: list[float] = []
    converged = False
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        loss, _, _, grad_coupling = trajectory_match_value_and_grad(
            theta0, omega, current, observed, observation_steps, dt, n_steps, integrator=integrator
        )
        history.append(loss)
        descent = constrain(grad_coupling) if symmetric else grad_coupling
        step = learning_rate
        improved = False
        while step >= min_step:
            candidate = constrain(current - step * descent)
            if loss_only(candidate) < loss - tolerance:
                current = candidate
                improved = True
                break
            step *= 0.5
        if not improved:
            converged = True
            break
    return SystemIdentificationResult(
        coupling=current,
        cost_history=tuple(history),
        iterations=iterations,
        converged=converged,
    )


__all__ = [
    "SystemIdentificationResult",
    "learn_coupling",
    "trajectory_match_value",
    "trajectory_match_value_and_grad",
]
