# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Kuramoto control objectives and value-and-grad
"""Differentiable Kuramoto control objectives and their value-and-grad.

This is the control layer of the Kuramoto specialisation: it composes the differentiable
networked-Kuramoto integrators
(:mod:`~oscillatools.accel.diff_kuramoto_euler`,
:mod:`~oscillatools.accel.diff_kuramoto_rk4`) with terminal objectives to return the
gradient of a control cost with respect to the initial phases, the natural frequencies and the
coupling matrix — the quantities optimal coupling design, pinning control and learning ``K``
optimise. A *terminal objective* maps the final integrated phase ``θ_T`` to a scalar cost and
its cotangent ``∂cost/∂θ_T``; :func:`terminal_objective_value_and_grad` integrates the dynamics,
evaluates the objective and backpropagates the cotangent through the chosen integrator.

The objectives reuse the dispatched (Rust → Julia → Python) order-parameter and
interaction-energy observables for their cotangents, so the whole value-and-grad runs through
the accelerated tiers; this module itself is the orchestration layer and adds no compute kernel.

Catalogue:

- :func:`coherence_objective` — drive to synchronisation, ``L = 1 − r(θ_T)²``.
- :func:`phase_target_objective` — track a target phase pattern,
  ``L = ⟨1 − cos(θ_T − θ*)⟩``.
- :func:`interaction_energy_objective` — minimise the interaction energy ``E(θ_T, K)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_euler import kuramoto_euler_trajectory, kuramoto_euler_vjp
from .diff_kuramoto_rk4 import kuramoto_rk4_trajectory, kuramoto_rk4_vjp
from .kuramoto_energy import kuramoto_interaction_energy, kuramoto_interaction_energy_gradient
from .order_parameter_observables import order_parameter, order_parameter_gradient

# A terminal objective maps the final phase θ_T to (cost, ∂cost/∂θ_T).
TerminalObjective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]

_GradResult = tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def coherence_objective(theta_final: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    r"""Synchronisation cost ``L = 1 − r(θ_T)²`` and its cotangent.

    Minimised when the ensemble is fully phase-locked (``r = 1``). The cotangent is
    :math:`\partial L/\partial \theta_T = -2 r\, \partial r/\partial \theta_T`, using the
    dispatched order parameter and its gradient.

    Parameters
    ----------
    theta_final : numpy.ndarray
        One-dimensional array of ``N`` final phases.

    Returns
    -------
    tuple[float, numpy.ndarray]
        ``(cost, cotangent)`` with the cotangent of shape ``(N,)``.
    """
    radius = float(order_parameter(theta_final))
    gradient = order_parameter_gradient(theta_final)
    return 1.0 - radius * radius, np.ascontiguousarray(-2.0 * radius * gradient, dtype=np.float64)


def phase_target_objective(target: NDArray[np.float64]) -> TerminalObjective:
    r"""Build a phase-tracking objective ``L = ⟨1 − cos(θ_T − θ*)⟩`` for a target pattern.

    Minimised when every final phase matches the target ``θ*`` (up to a global shift the cosine
    cost ignores per oscillator). The cotangent is
    :math:`\partial L/\partial \theta_{T,j} = \sin(\theta_{T,j} - \theta^*_j)/N`.

    Parameters
    ----------
    target : numpy.ndarray
        One-dimensional array of ``N`` target phases in radians.

    Returns
    -------
    TerminalObjective
        A callable mapping ``θ_T`` to ``(cost, cotangent)``.
    """
    reference = np.ascontiguousarray(target, dtype=np.float64)

    def objective(theta_final: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        if theta_final.shape != reference.shape:
            raise ValueError(f"target must have shape {theta_final.shape}, got {reference.shape}")
        difference = theta_final - reference
        count = theta_final.size
        cost = float(np.mean(1.0 - np.cos(difference)))
        cotangent = np.ascontiguousarray(np.sin(difference) / count, dtype=np.float64)
        return cost, cotangent

    return objective


def interaction_energy_objective(coupling: NDArray[np.float64]) -> TerminalObjective:
    r"""Build an interaction-energy objective ``L = E(θ_T, K)`` for a coupling matrix.

    The Kuramoto interaction energy ``E = −½ Σ_jk K_jk cos(θ_j − θ_k)``; minimised at the
    phase-locked configuration of ``K``. The cotangent is the dispatched energy gradient
    ``∂E/∂θ_T``. The ``coupling`` here defines the *energy landscape* and need not equal the
    coupling driving the dynamics.

    Parameters
    ----------
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix defining the energy.

    Returns
    -------
    TerminalObjective
        A callable mapping ``θ_T`` to ``(cost, cotangent)``.
    """
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)

    def objective(theta_final: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        cost = float(kuramoto_interaction_energy(theta_final, matrix))
        cotangent = np.ascontiguousarray(
            kuramoto_interaction_energy_gradient(theta_final, matrix), dtype=np.float64
        )
        return cost, cotangent

    return objective


def terminal_objective_value_and_grad(
    objective: TerminalObjective,
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> _GradResult:
    r"""Integrate, evaluate a terminal objective and backpropagate its gradient.

    Runs the chosen differentiable integrator forward, evaluates ``objective`` on the final
    phase, and backpropagates the objective cotangent to return the cost and its gradients with
    respect to the initial phases, the natural frequencies and the coupling matrix.

    Parameters
    ----------
    objective : TerminalObjective
        A callable mapping the final phase ``θ_T`` to ``(cost, ∂cost/∂θ_T)`` — e.g. from
        :func:`coherence_objective`, :func:`phase_target_objective` or
        :func:`interaction_energy_objective`.
    theta0 : numpy.ndarray
        One-dimensional ``(N,)`` initial phases.
    omega : numpy.ndarray
        One-dimensional ``(N,)`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix driving the dynamics.
    dt : float
        The integrator step size.
    n_steps : int
        The number of integration steps.
    integrator : str, optional
        ``"rk4"`` (default, fourth-order) or ``"euler"`` (first-order).

    Returns
    -------
    tuple
        ``(cost, grad_theta0, grad_omega, grad_coupling)`` with gradient shapes ``(N,)``,
        ``(N,)`` and ``(N, N)``.

    Raises
    ------
    ValueError
        If ``integrator`` is not ``"rk4"`` or ``"euler"``.
    """
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        cost, cotangent = objective(trajectory[-1])
        grad_theta0, grad_omega, grad_coupling = kuramoto_rk4_vjp(
            trajectory, omega, coupling, dt, cotangent
        )
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
        cost, cotangent = objective(trajectory[-1])
        grad_theta0, grad_omega, grad_coupling = kuramoto_euler_vjp(
            trajectory, coupling, dt, cotangent
        )
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    return cost, grad_theta0, grad_omega, grad_coupling


def terminal_objective_value(
    objective: TerminalObjective,
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> float:
    """Integrate and evaluate a terminal objective without its gradient.

    The value-only companion of :func:`terminal_objective_value_and_grad` — runs the forward
    integrator and returns only ``objective``'s cost on the final phase, skipping the adjoint.
    Useful for line searches and objective sweeps. Parameters and ``integrator`` choices are as
    in :func:`terminal_objective_value_and_grad`.

    Returns
    -------
    float
        The objective cost at the integrated final phase.

    Raises
    ------
    ValueError
        If ``integrator`` is not ``"rk4"`` or ``"euler"``.
    """
    if integrator == "rk4":
        trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
    elif integrator == "euler":
        trajectory = kuramoto_euler_trajectory(theta0, omega, coupling, dt, n_steps)
    else:
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")
    cost, _ = objective(trajectory[-1])
    return cost


def synchronisation_value_and_grad(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
) -> _GradResult:
    """Cost ``1 − r(θ_T)²`` and its gradient — the canonical drive-to-synchronisation objective.

    Convenience wrapper of :func:`terminal_objective_value_and_grad` with
    :func:`coherence_objective`. See those for the parameters and return value.
    """
    return terminal_objective_value_and_grad(
        coherence_objective, theta0, omega, coupling, dt, n_steps, integrator=integrator
    )


__all__ = [
    "TerminalObjective",
    "coherence_objective",
    "interaction_energy_objective",
    "phase_target_objective",
    "synchronisation_value_and_grad",
    "terminal_objective_value",
    "terminal_objective_value_and_grad",
]
