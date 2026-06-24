# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Kuramoto pinning control
"""Differentiable Kuramoto pinning control by sparse pinning-gain design.

Pinning control drives the whole network to synchronisation by acting on only a subset of
oscillators: each pinned node ``j`` feels an extra force ``g_j sin(s − θ_j)`` toward a target
phase ``s``. This is expressed *exactly* as networked Kuramoto dynamics on the network augmented
by one fixed pacemaker node held at ``s`` (zero frequency, no incoming coupling), with the
pinning gains forming its coupling column — so the existing differentiable networked integrators
and adjoint apply unchanged, and the pinning-gain gradient ``∂L/∂g_j`` is the corresponding
column of ``∂L/∂K``.

:func:`design_pinning` optimises the pinning gains by projected gradient descent with an
ℓ1 sparsity penalty and a backtracking line search; the penalty drives most gains to zero so the
surviving non-zero gains select a small pin-set that still pulls the ensemble into coherence.
This is a control-application layer over the differentiable simulation and adds no compute
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_euler import kuramoto_euler_trajectory, kuramoto_euler_vjp
from .diff_kuramoto_rk4 import kuramoto_rk4_trajectory, kuramoto_rk4_vjp
from .order_parameter_observables import order_parameter, order_parameter_gradient


@dataclass(frozen=True)
class PinningDesignResult:
    """Outcome of a pinning-control design run.

    Attributes
    ----------
    gains : numpy.ndarray
        The optimised ``(N,)`` non-negative pinning gains (sparse).
    pin_set : numpy.ndarray
        The integer indices of the pinned nodes (gains above the pin threshold), sorted.
    cost_history : tuple of float
        The coherence cost at the start of each accepted iteration (monotonically
        non-increasing in the ℓ1-penalised objective).
    iterations : int
        The number of iterations performed.
    converged : bool
        ``True`` if the line search could no longer reduce the penalised objective.
    """

    gains: NDArray[np.float64]
    pin_set: NDArray[np.int64]
    cost_history: tuple[float, ...]
    iterations: int
    converged: bool


def _augment(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    gains: NDArray[np.float64],
    target: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the pacemaker-augmented ``(N+1)`` system for pinning gains ``g`` and target ``s``."""
    count = theta0.size
    theta_aug = np.concatenate([theta0, [float(target)]]).astype(np.float64)
    omega_aug = np.concatenate([omega, [0.0]]).astype(np.float64)
    coupling_aug = np.zeros((count + 1, count + 1), dtype=np.float64)
    coupling_aug[:count, :count] = coupling
    coupling_aug[:count, count] = gains
    return theta_aug, omega_aug, coupling_aug


def _validate(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    gains: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    pinning = np.ascontiguousarray(gains, dtype=np.float64)
    count = phases.size
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if pinning.shape != (count,):
        raise ValueError(f"gains must have shape ({count},), got {pinning.shape}")
    return phases, frequencies, matrix, pinning


def pinning_coherence_value_and_grad(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    gains: NDArray[np.float64],
    target: float,
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> tuple[float, NDArray[np.float64]]:
    r"""Coherence cost ``1 − r(θ_T)²`` of the pinned network and its gradient in the gains.

    Integrates the pacemaker-augmented dynamics, evaluates the order-parameter cost on the
    original ``N`` oscillators (excluding the pacemaker) and backpropagates to the pinning gains.

    Parameters
    ----------
    theta0, omega, coupling : numpy.ndarray
        The ``(N,)`` initial phases, ``(N,)`` frequencies and ``(N, N)`` coupling of the network.
    gains : numpy.ndarray
        The ``(N,)`` non-negative pinning gains.
    target : float
        The pacemaker / pin target phase ``s``.
    dt : float
        The integrator step size.
    n_steps : int
        The number of integration steps.
    integrator : str, optional
        ``"rk4"`` (default) or ``"euler"``.

    Returns
    -------
    tuple[float, numpy.ndarray]
        ``(cost, grad_gains)`` with the gain gradient of shape ``(N,)``.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or ``integrator`` is invalid.
    """
    phases, frequencies, matrix, pinning = _validate(theta0, omega, coupling, gains)
    count = phases.size
    theta_aug, omega_aug, coupling_aug = _augment(phases, frequencies, matrix, pinning, target)
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(theta_aug, omega_aug, coupling_aug, dt, n_steps)
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(theta_aug, omega_aug, coupling_aug, dt, n_steps)
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    final = trajectory[-1][:count]
    radius = float(order_parameter(final))
    cost = 1.0 - radius * radius
    cotangent = np.zeros(count + 1, dtype=np.float64)
    cotangent[:count] = -2.0 * radius * order_parameter_gradient(final)
    if integrator == "rk4":
        _, _, grad_coupling = kuramoto_rk4_vjp(trajectory, omega_aug, coupling_aug, dt, cotangent)
    else:
        _, _, grad_coupling = kuramoto_euler_vjp(trajectory, coupling_aug, dt, cotangent)
    grad_gains = np.ascontiguousarray(grad_coupling[:count, count], dtype=np.float64)
    return cost, grad_gains


def pinning_coherence_value(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    gains: NDArray[np.float64],
    target: float,
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> float:
    """Coherence cost ``1 − r(θ_T)²`` of the pinned network without its gradient.

    The value-only companion of :func:`pinning_coherence_value_and_grad` (used for the design
    line search). Parameters and the ``integrator`` choice are as there.
    """
    phases, frequencies, matrix, pinning = _validate(theta0, omega, coupling, gains)
    count = phases.size
    theta_aug, omega_aug, coupling_aug = _augment(phases, frequencies, matrix, pinning, target)
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(theta_aug, omega_aug, coupling_aug, dt, n_steps)
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(theta_aug, omega_aug, coupling_aug, dt, n_steps)
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    radius = float(order_parameter(trajectory[-1][:count]))
    return 1.0 - radius * radius


def design_pinning(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    target: float,
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
    max_iterations: int = 100,
    learning_rate: float = 1.0,
    sparsity_weight: float = 0.0,
    max_gain: float = np.inf,
    pin_threshold: float = 1e-3,
    tolerance: float = 1e-9,
    min_step: float = 1e-8,
) -> PinningDesignResult:
    r"""Design sparse pinning gains that drive the network to synchronisation.

    Minimises ``1 − r(θ_T)² + λ Σ_j g_j`` over non-negative gains ``g ≤ g_max`` by projected
    gradient descent with a backtracking line search (monotone in the penalised objective). The
    ℓ1 weight ``λ`` (``sparsity_weight``) trades coherence against the number of pins; the
    surviving gains above ``pin_threshold`` form the pin-set.

    Parameters
    ----------
    theta0, omega, coupling : numpy.ndarray
        The ``(N,)`` initial phases, ``(N,)`` frequencies and ``(N, N)`` coupling.
    target : float
        The pin target phase.
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
    sparsity_weight : float, optional
        The ℓ1 penalty weight ``λ`` on the gains (``0`` for no sparsity pressure).
    max_gain : float, optional
        The upper clip on each gain.
    pin_threshold : float, optional
        Gains above this enter the reported pin-set.
    tolerance : float, optional
        The minimum penalised-objective decrease for a step to be accepted.
    min_step : float, optional
        The smallest line-search step before convergence is declared.

    Returns
    -------
    PinningDesignResult
        The optimised gains, the selected pin-set, the cost history and the convergence flag.

    Raises
    ------
    ValueError
        If ``max_iterations`` is not positive or the inputs are inconsistent.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    phases, frequencies, matrix, gains = _validate(theta0, omega, coupling, np.zeros(theta0.size))

    def penalised(current_gains: NDArray[np.float64]) -> float:
        cost = pinning_coherence_value(
            phases, frequencies, matrix, current_gains, target, dt, n_steps, integrator=integrator
        )
        return cost + sparsity_weight * float(current_gains.sum())

    current = np.clip(gains, 0.0, max_gain)
    history: list[float] = []
    converged = False
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        cost, grad_gains = pinning_coherence_value_and_grad(
            phases, frequencies, matrix, current, target, dt, n_steps, integrator=integrator
        )
        history.append(cost)
        penalised_now = cost + sparsity_weight * float(current.sum())
        descent = grad_gains + sparsity_weight
        step = learning_rate
        improved = False
        while step >= min_step:
            candidate = np.clip(current - step * descent, 0.0, max_gain)
            if penalised(candidate) < penalised_now - tolerance:
                current = candidate
                improved = True
                break
            step *= 0.5
        if not improved:
            converged = True
            break
    pin_set = np.ascontiguousarray(np.flatnonzero(current > pin_threshold), dtype=np.int64)
    return PinningDesignResult(
        gains=np.ascontiguousarray(current, dtype=np.float64),
        pin_set=pin_set,
        cost_history=tuple(history),
        iterations=iterations,
        converged=converged,
    )


__all__ = [
    "PinningDesignResult",
    "design_pinning",
    "pinning_coherence_value",
    "pinning_coherence_value_and_grad",
]
