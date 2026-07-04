# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Kuramoto optimal coupling design
"""Differentiable Kuramoto optimal coupling design by projected gradient descent.

Optimal coupling design: find the coupling matrix ``K`` that minimises a Kuramoto control
objective (e.g. drive-to-synchronisation) by gradient descent through the differentiable
integrator. Each iteration evaluates the objective and its gradient with respect to ``K`` (via
:func:`~oscillatools.accel.kuramoto_control.terminal_objective_value_and_grad`) and takes
a step along the negative gradient, projected back onto a structural constraint set (symmetry,
non-negativity, zero diagonal, …). A backtracking line search shrinks the step until the
objective strictly decreases, so the cost history is monotonically non-increasing and the
iteration converges to a local optimum of the design problem.

This is the control-application layer on top of the differentiable simulation: it composes the
accelerated integrators and value-and-grad and adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_control import (
    TerminalObjective,
    coherence_objective,
    terminal_objective_value,
    terminal_objective_value_and_grad,
)

# A coupling projection maps a candidate K onto the admissible constraint set.
CouplingProjection = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class CouplingDesignResult:
    """Outcome of an optimal-coupling-design run.

    Attributes
    ----------
    coupling : numpy.ndarray
        The optimised ``(N, N)`` coupling matrix.
    cost_history : tuple of float
        The objective cost at the start of each accepted iteration (monotonically
        non-increasing).
    iterations : int
        The number of iterations performed.
    converged : bool
        ``True`` if the line search could no longer reduce the cost (a local optimum was
        reached) before the iteration budget ran out.
    """

    coupling: NDArray[np.float64]
    cost_history: tuple[float, ...]
    iterations: int
    converged: bool


def symmetric_nonnegative_projection(coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a coupling matrix onto symmetric, non-negative, zero-diagonal matrices.

    The standard admissible set for an undirected, excitatory Kuramoto network: symmetrise,
    clip negative weights to zero and remove self-coupling.

    Parameters
    ----------
    coupling : numpy.ndarray
        A two-dimensional ``(N, N)`` candidate coupling matrix.

    Returns
    -------
    numpy.ndarray
        The projected ``(N, N)`` matrix.
    """
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    matrix = np.clip(matrix, 0.0, None)
    np.fill_diagonal(matrix, 0.0)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def optimise_coupling(
    objective: TerminalObjective,
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling_init: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
    max_iterations: int = 100,
    learning_rate: float = 1.0,
    projection: CouplingProjection | None = None,
    tolerance: float = 1e-9,
    min_step: float = 1e-8,
) -> CouplingDesignResult:
    """Design a coupling matrix minimising a control objective by projected gradient descent.

    Iterates ``K ← Π(K − η ∇_K cost)`` with a backtracking line search on the step ``η`` so each
    accepted iteration strictly lowers the objective, where ``Π`` is the (optional) constraint
    projection. The gradient flows through the differentiable integrator.

    Parameters
    ----------
    objective : TerminalObjective
        A terminal objective (e.g. :func:`~oscillatools.accel.kuramoto_control.coherence_objective`).
    theta0 : numpy.ndarray
        One-dimensional ``(N,)`` initial phases.
    omega : numpy.ndarray
        One-dimensional ``(N,)`` natural frequencies.
    coupling_init : numpy.ndarray
        Two-dimensional ``(N, N)`` initial coupling matrix; projected before the first step.
    dt : float
        The integrator step size.
    n_steps : int
        The number of integration steps.
    integrator : str, optional
        ``"rk4"`` (default) or ``"euler"``.
    max_iterations : int, optional
        The iteration budget.
    learning_rate : float, optional
        The initial step size for the line search.
    projection : CouplingProjection or None, optional
        A projection onto the admissible coupling set, applied to the initial matrix and after
        every step. ``None`` leaves the coupling unconstrained.
    tolerance : float, optional
        The minimum cost decrease for a step to be accepted.
    min_step : float, optional
        The smallest line-search step before the iteration is declared converged.

    Returns
    -------
    CouplingDesignResult
        The optimised coupling, the monotone cost history, the iteration count and whether a
        local optimum was reached.

    Raises
    ------
    ValueError
        If ``max_iterations`` is not positive, or ``integrator`` is invalid (from the
        value-and-grad call).
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    project = projection if projection is not None else (lambda matrix: matrix)
    current = project(np.ascontiguousarray(coupling_init, dtype=np.float64))
    history: list[float] = []
    converged = False
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        cost, _, _, grad_coupling = terminal_objective_value_and_grad(
            objective, theta0, omega, current, dt, n_steps, integrator=integrator
        )
        history.append(cost)
        step = learning_rate
        improved = False
        while step >= min_step:
            candidate = project(current - step * grad_coupling)
            candidate_cost = terminal_objective_value(
                objective, theta0, omega, candidate, dt, n_steps, integrator=integrator
            )
            if candidate_cost < cost - tolerance:
                current = candidate
                improved = True
                break
            step *= 0.5
        if not improved:
            converged = True
            break
    return CouplingDesignResult(
        coupling=current,
        cost_history=tuple(history),
        iterations=iterations,
        converged=converged,
    )


def design_synchronising_coupling(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling_init: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    integrator: str = "rk4",
    max_iterations: int = 100,
    learning_rate: float = 1.0,
    projection: CouplingProjection | None = symmetric_nonnegative_projection,
    tolerance: float = 1e-9,
    min_step: float = 1e-8,
) -> CouplingDesignResult:
    """Design a coupling matrix that drives the ensemble to synchronisation.

    Convenience wrapper of :func:`optimise_coupling` with the coherence objective
    ``1 − r(θ_T)²`` and, by default, the symmetric non-negative projection. See
    :func:`optimise_coupling` for the parameters and return value.
    """
    return optimise_coupling(
        coherence_objective,
        theta0,
        omega,
        coupling_init,
        dt,
        n_steps,
        integrator=integrator,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        projection=projection,
        tolerance=tolerance,
        min_step=min_step,
    )


__all__ = [
    "CouplingDesignResult",
    "CouplingProjection",
    "design_synchronising_coupling",
    "optimise_coupling",
    "symmetric_nonnegative_projection",
]
