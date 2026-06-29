# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Pseudo-arclength continuation past fold points
r"""Pseudo-arclength continuation — tracing solution branches past fold points.

Natural-parameter continuation follows a branch of equilibria ``G(x, λ) = 0`` by stepping the
parameter ``λ`` and solving for the state ``x``; it *fails at a fold* (a saddle-node / limit
point), where the branch turns around in ``λ`` and the state Jacobian ``∂G/∂x`` becomes singular —
the parameter cannot be advanced and the solver stalls. The explosive-synchronisation continuation
already in the toolkit (:mod:`scpn_quantum_control.accel.kuramoto_explosive_continuation`) is a
natural-parameter sweep and inherits this limitation.

Pseudo-arclength continuation (Keller 1977 — the method underlying AUTO and MatCont) instead
parametrises the branch by *arclength*: it predicts along the unit tangent and corrects with Newton
on the *bordered* system

.. math::

    G(x, λ) = 0, \qquad \dot x_0^\mathsf{T}(x - x_0) + \dot λ_0 (λ - λ_0) - ds = 0,

whose ``(n+1) × (n+1)`` Jacobian ``[[G_x, G_λ], [\dot x_0^\mathsf{T}, \dot λ_0]]`` stays non-singular
*through* the fold even though ``G_x`` alone is singular there. The branch is therefore traced
smoothly around limit points, picking up the second (e.g. unstable) solution branch that
natural-parameter continuation cannot reach. The canonical Kuramoto fold is the single-oscillator
locking saddle-node ``G(φ, K) = K\sin φ - Δω`` at ``K = Δω`` (``φ = π/2``), where the stable and
unstable locked phases merge.

This is a generic numerical-continuation core over a user-supplied residual and its Jacobians; it
adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: A residual ``G(x, λ)`` returning an ``(n,)`` vector for the state ``x`` and parameter ``λ``.
Residual = Callable[[NDArray[np.float64], float], NDArray[np.float64]]

#: The state Jacobian ``∂G/∂x`` returning an ``(n, n)`` matrix.
StateJacobian = Callable[[NDArray[np.float64], float], NDArray[np.float64]]

#: The parameter derivative ``∂G/∂λ`` returning an ``(n,)`` vector.
ParameterDerivative = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


@dataclass(frozen=True)
class PseudoArclengthBranch:
    """A solution branch ``G(x, λ) = 0`` traced by pseudo-arclength continuation.

    Attributes
    ----------
    states : numpy.ndarray
        The ``(n_points, n)`` state vectors along the branch.
    parameters : numpy.ndarray
        The ``(n_points,)`` continuation-parameter values.
    arclengths : numpy.ndarray
        The cumulative arclength at each point (``0, ds, 2 ds, …``).
    """

    states: NDArray[np.float64]
    parameters: NDArray[np.float64]
    arclengths: NDArray[np.float64]

    @property
    def n_points(self) -> int:
        """The number of points on the branch."""
        return int(self.parameters.size)


def _bordered_solve(
    state_jacobian: NDArray[np.float64],
    parameter_derivative: NDArray[np.float64],
    tangent_state: NDArray[np.float64],
    tangent_parameter: float,
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the bordered ``(n+1)`` system ``[[G_x, G_λ], [ẋᵀ, λ̇]] z = rhs``."""
    count = tangent_state.size
    matrix = np.zeros((count + 1, count + 1), dtype=np.float64)
    matrix[:count, :count] = state_jacobian
    matrix[:count, count] = parameter_derivative
    matrix[count, :count] = tangent_state
    matrix[count, count] = tangent_parameter
    return np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)


def pseudo_arclength_continuation(
    residual: Residual,
    state_jacobian: StateJacobian,
    parameter_derivative: ParameterDerivative,
    initial_state: NDArray[np.float64],
    initial_parameter: float,
    *,
    step: float,
    n_steps: int,
    parameter_direction: float = 1.0,
    newton_tolerance: float = 1e-10,
    max_newton_iterations: int = 50,
) -> PseudoArclengthBranch:
    r"""Trace a branch of ``G(x, λ) = 0`` by pseudo-arclength continuation past fold points.

    Parameters
    ----------
    residual : callable
        The residual ``G(x, λ) → (n,)`` (see :data:`Residual`).
    state_jacobian : callable
        The state Jacobian ``∂G/∂x → (n, n)`` (see :data:`StateJacobian`).
    parameter_derivative : callable
        The parameter derivative ``∂G/∂λ → (n,)`` (see :data:`ParameterDerivative`).
    initial_state : numpy.ndarray
        A starting point ``x_0`` on the branch (``G(x_0, λ_0) ≈ 0``).
    initial_parameter : float
        The starting parameter ``λ_0``.
    step : float
        The arclength step ``ds`` (``> 0``).
    n_steps : int
        The number of continuation steps (``≥ 1``); the branch has ``n_steps + 1`` points.
    parameter_direction : float, optional
        The sign of the initial parameter sweep (``+1`` increases ``λ`` first); defaults to ``+1``.
    newton_tolerance : float, optional
        The corrector convergence tolerance on ``‖G‖ + |N|``; defaults to ``1e-10``.
    max_newton_iterations : int, optional
        The maximum corrector iterations per step; defaults to ``50``.

    Returns
    -------
    PseudoArclengthBranch
        The traced branch (states, parameters and arclengths).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    RuntimeError
        If the Newton corrector fails to converge at a step.
    """
    state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if state.ndim != 1 or state.size < 1:
        raise ValueError("initial_state must be a non-empty one-dimensional array")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if parameter_direction not in (1.0, -1.0):
        raise ValueError(f"parameter_direction must be +1 or -1, got {parameter_direction}")
    count = state.size
    parameter = float(initial_parameter)

    # Initial tangent: solve G_x ẋ = −G_λ with λ̇ = 1, normalise and orient by the requested sweep.
    tangent_state: NDArray[np.float64] = np.asarray(
        np.linalg.solve(state_jacobian(state, parameter), -parameter_derivative(state, parameter)),
        dtype=np.float64,
    )
    tangent_parameter = 1.0
    norm = float(np.sqrt(tangent_state @ tangent_state + tangent_parameter**2))
    tangent_state = np.asarray(tangent_state / norm, dtype=np.float64)
    tangent_parameter = tangent_parameter / norm
    if np.sign(tangent_parameter) != np.sign(parameter_direction):
        tangent_state = np.asarray(-tangent_state, dtype=np.float64)
        tangent_parameter = -tangent_parameter

    states = np.empty((n_steps + 1, count), dtype=np.float64)
    parameters = np.empty(n_steps + 1, dtype=np.float64)
    states[0] = state
    parameters[0] = parameter

    for index in range(n_steps):
        anchor_state = state.copy()
        anchor_parameter = parameter
        candidate_state = anchor_state + step * tangent_state
        candidate_parameter = anchor_parameter + step * tangent_parameter

        converged = False
        for _ in range(max_newton_iterations):
            g = residual(candidate_state, candidate_parameter)
            constraint = float(
                tangent_state @ (candidate_state - anchor_state)
                + tangent_parameter * (candidate_parameter - anchor_parameter)
                - step
            )
            if float(np.sqrt(g @ g + constraint**2)) < newton_tolerance:
                converged = True
                break
            update = _bordered_solve(
                state_jacobian(candidate_state, candidate_parameter),
                parameter_derivative(candidate_state, candidate_parameter),
                tangent_state,
                tangent_parameter,
                -np.concatenate([g, [constraint]]),
            )
            candidate_state = candidate_state + update[:count]
            candidate_parameter = candidate_parameter + float(update[count])
        if not converged:
            raise RuntimeError(f"pseudo-arclength corrector failed to converge at step {index}")

        state = candidate_state
        parameter = candidate_parameter
        states[index + 1] = state
        parameters[index + 1] = parameter

        # New tangent: the bordered null vector, continued in the same arclength direction.
        new_tangent = _bordered_solve(
            state_jacobian(state, parameter),
            parameter_derivative(state, parameter),
            tangent_state,
            tangent_parameter,
            np.concatenate([np.zeros(count), [1.0]]),
        )
        norm = float(np.sqrt(new_tangent @ new_tangent))
        tangent_state = np.asarray(new_tangent[:count] / norm, dtype=np.float64)
        tangent_parameter = float(new_tangent[count]) / norm

    arclengths = step * np.arange(n_steps + 1, dtype=np.float64)
    return PseudoArclengthBranch(states=states, parameters=parameters, arclengths=arclengths)


__all__ = [
    "ParameterDerivative",
    "PseudoArclengthBranch",
    "Residual",
    "StateJacobian",
    "pseudo_arclength_continuation",
]
