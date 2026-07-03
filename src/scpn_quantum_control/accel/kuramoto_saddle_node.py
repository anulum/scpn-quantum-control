# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Saddle-node (fold) location for the Kuramoto phase-locked state
r"""Locate the saddle-node bifurcation of the Kuramoto phase-locked state by Newton.

As the coupling strength ``K`` of a networked Kuramoto system is lowered, the
phase-locked state (every oscillator sharing one collective frequency ``Ω``)
disappears at a saddle-node fold: a stable and an unstable locked branch collide
and the locked solution ceases to exist. This module finds that fold **directly**
— by Newton's method on the fold defining system — rather than by tracing the
branch with a parameter sweep or continuation and watching for the collapse.

The defining system is the standard Moore–Spence extended system

.. math::

    G(\theta, \Omega, v, K) =
    \begin{bmatrix} F(\theta; K) - \Omega\,\mathbf{1} \\ J_R(\theta; K)\,v \\ \langle c, v\rangle - 1 \end{bmatrix} = 0,

where :math:`F_i = \omega_i + \sum_j K_{ij}\sin(\theta_j-\theta_i)` is the
production networked force, the first phase is pinned (:math:`\theta_0=0`) to
remove the uniform-phase-shift Goldstone gauge and ``Ω`` is carried as an
unknown, :math:`J_R` is the reduced Jacobian of the locked-state residual
(non-singular on a regular branch, singular exactly at the fold), ``v`` is its
null vector, and :math:`\langle c, v\rangle = 1` fixes the null-vector scale.
Newton on ``G`` converges quadratically to the fold and pins ``K`` to machine
precision — for two oscillators it recovers the analytic fold
:math:`K_{\text{fold}} = |\omega_0-\omega_1|/2` to ``~1e-14``.

The Newton step needs exact derivatives: the reduced Jacobian (from
:func:`~scpn_quantum_control.accel.networked_kuramoto.networked_kuramoto_jacobian`),
the parameter derivative :math:`\partial F/\partial K`, and the directional
derivative of the Jacobian along the null vector,
:math:`(\partial J/\partial\theta_m)\,v` — a Hessian-vector product of the force.
That third-order object is supplied analytically here; because the Kuramoto
Jacobian closes under differentiation, the Hessian-vector product has the same
row-sum-zero structure as the Jacobian itself. A finite-difference toolkit would
have to approximate it, losing the machine-precision fold; the exact derivative
is what makes the direct locator possible. The heavy per-step force and Jacobian
evaluations dispatch through the accelerated networked tiers; the Newton
orchestration is a pure-Python sequential loop (no accelerated tier), matching
the treatment of the adaptive integrators and the pseudo-arclength continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian


@dataclass(frozen=True)
class SaddleNodePoint:
    """The located saddle-node fold of a Kuramoto phase-locked branch.

    Parameters
    ----------
    critical_coupling:
        The coupling strength ``K`` at the fold — the smallest strength for which
        the phase-locked state exists on this branch.
    phases:
        The folding locked phases ``θ`` (length ``N``, gauge ``θ[0] = 0``).
    collective_frequency:
        The common frequency ``Ω`` the locked state rotates at.
    null_vector:
        The reduced-Jacobian null vector ``v`` at the fold (the marginal
        direction along which the stable and unstable branches merge).
    residual_norm:
        The infinity norm of the fold defining system at the solution.
    iterations:
        Newton iterations taken by the refinement.
    converged:
        Whether the refinement reached ``newton_tolerance``.
    """

    critical_coupling: float
    phases: NDArray[np.float64]
    collective_frequency: float
    null_vector: NDArray[np.float64]
    residual_norm: float
    iterations: int
    converged: bool


def _reduced_jacobian(jacobian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assemble the ``(N, N)`` locked-state Jacobian ``J_R = [J[:, 1:] | -1]``.

    The columns are the free-phase derivatives (phase ``0`` is the gauge) and a
    final ``-1`` column for the collective-frequency unknown ``Ω``.
    """
    count = jacobian.shape[0]
    reduced = np.empty((count, count), dtype=np.float64)
    reduced[:, : count - 1] = jacobian[:, 1:]
    reduced[:, count - 1] = -1.0
    return reduced


def _jacobian_direction_derivative(
    theta: NDArray[np.float64], coupling: NDArray[np.float64], direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Return ``P`` with ``P[i, m] = Σ_j (∂J_ij/∂θ_m) direction_j``.

    The analytic Hessian-vector product of the networked force. Differentiating
    :math:`J_{ij} = K_{ij}\cos(\theta_j-\theta_i)` gives an object with the same
    row-sum-zero structure as ``J``: the off-diagonal is
    :math:`s_{im}(d_i - d_m)` with :math:`s_{im} = K_{im}\sin(\theta_m-\theta_i)`,
    and the diagonal is minus the row sum.
    """
    difference = theta[None, :] - theta[:, None]
    sin_term = coupling * np.sin(difference)  # s[i, m] = K_im sin(theta_m - theta_i)
    off = sin_term * (direction[:, None] - direction[None, :])
    result = off.copy()
    np.fill_diagonal(result, -off.sum(axis=1))
    return result


def _locked_residual(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    collective_frequency: float,
) -> NDArray[np.float64]:
    """Return ``ω + F_coupling(θ; K) - Ω`` — zero exactly on a phase-locked state.

    ``networked_kuramoto_force`` returns only the coupling term ``Σ_j K_ij
    sin(θ_j-θ_i)``; the natural frequencies are added here (the integrators add
    them separately as ``θ̇ = ω + F``).
    """
    return omega + networked_kuramoto_force(theta, coupling) - collective_frequency


def _solve_locked_state(
    omega: NDArray[np.float64],
    structure: NDArray[np.float64],
    strength: float,
    seed: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> NDArray[np.float64] | None:
    """Newton for the locked state from ``seed = (θ_1..θ_{N-1}, Ω)``; ``None`` if lost.

    Returns the converged ``(free phases, Ω)`` vector, or ``None`` when the branch
    has folded away (the Newton step diverges or the Jacobian is singular).
    """
    count = omega.shape[0]
    coupling = strength * structure
    unknown = seed.astype(np.float64).copy()
    for _ in range(max_iterations):
        theta = np.concatenate([[0.0], unknown[: count - 1]])
        residual = _locked_residual(theta, omega, coupling, float(unknown[count - 1]))
        if np.max(np.abs(residual)) < tolerance:
            return unknown
        reduced = _reduced_jacobian(networked_kuramoto_jacobian(theta, coupling))
        try:
            unknown = unknown - np.linalg.solve(reduced, residual)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(unknown[: count - 1])) > 1.0e3:
            return None
    return None


def fold_defining_residual(
    unknowns: NDArray[np.float64],
    omega: NDArray[np.float64],
    structure: NDArray[np.float64],
    normalisation: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Evaluate the Moore–Spence fold system ``G`` (length ``2N+1``).

    Parameters
    ----------
    unknowns:
        The packed vector ``(θ_1..θ_{N-1}, Ω, v_0..v_{N-1}, K)`` of length
        ``2N+1``.
    omega, structure:
        Natural frequencies and the coupling pattern (scaled by ``K``).
    normalisation:
        The fixed vector ``c`` of the scaling condition ``⟨c, v⟩ = 1``.

    Returns
    -------
    numpy.ndarray
        ``[F - Ω\mathbf 1;  J_R v;  ⟨c, v⟩ - 1]``.
    """
    count = omega.shape[0]
    theta = np.concatenate([[0.0], unknowns[: count - 1]])
    collective_frequency = float(unknowns[count - 1])
    null_vector = unknowns[count : 2 * count]
    strength = float(unknowns[-1])
    coupling = strength * structure
    reduced = _reduced_jacobian(networked_kuramoto_jacobian(theta, coupling))
    residual = np.empty(2 * count + 1, dtype=np.float64)
    residual[:count] = _locked_residual(theta, omega, coupling, collective_frequency)
    residual[count : 2 * count] = reduced @ null_vector
    residual[2 * count] = float(normalisation @ null_vector) - 1.0
    return residual


def fold_defining_jacobian(
    unknowns: NDArray[np.float64],
    omega: NDArray[np.float64],
    structure: NDArray[np.float64],
    normalisation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the exact ``(2N+1, 2N+1)`` Jacobian of :func:`fold_defining_residual`.

    The block structure is the bordered fold system: the reduced Jacobian and its
    parameter derivative for the residual block; the Hessian-vector product, the
    reduced Jacobian, and the parameter derivative of ``J_R v`` for the null-vector
    block; and the fixed normalisation row.
    """
    count = omega.shape[0]
    theta = np.concatenate([[0.0], unknowns[: count - 1]])
    null_vector = unknowns[count : 2 * count]
    strength = float(unknowns[-1])
    coupling = strength * structure
    jacobian = networked_kuramoto_jacobian(theta, coupling)
    reduced = _reduced_jacobian(jacobian)

    difference = theta[None, :] - theta[:, None]
    force_parameter_derivative = (structure * np.sin(difference)).sum(axis=1)  # dF/dK
    # J_R v uses J[:, 1:] contracted with v[:N-1]; pad so the gauge column drops out.
    padded_null = np.concatenate([[0.0], null_vector[: count - 1]])
    hessian_vector = _jacobian_direction_derivative(theta, coupling, padded_null)
    reduced_parameter_derivative = (jacobian[:, 1:] / strength) @ null_vector[: count - 1]

    matrix = np.zeros((2 * count + 1, 2 * count + 1), dtype=np.float64)
    matrix[:count, : count - 1] = jacobian[:, 1:]  # dR/dθ_free
    matrix[:count, count - 1] = -1.0  # dR/dΩ
    matrix[:count, -1] = force_parameter_derivative  # dR/dK
    matrix[count : 2 * count, : count - 1] = hessian_vector[:, 1:]  # d(J_R v)/dθ_free
    matrix[count : 2 * count, count : 2 * count] = reduced  # d(J_R v)/dv
    matrix[count : 2 * count, -1] = reduced_parameter_derivative  # d(J_R v)/dK
    matrix[2 * count, count : 2 * count] = normalisation  # d(⟨c,v⟩-1)/dv
    return matrix


def _newton_fold(
    seed: NDArray[np.float64],
    omega: NDArray[np.float64],
    structure: NDArray[np.float64],
    normalisation: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int, float, bool]:
    """Newton on the fold defining system; return ``(solution, iters, residual, converged)``."""
    unknowns = seed.astype(np.float64).copy()
    for iteration in range(max_iterations):
        residual = fold_defining_residual(unknowns, omega, structure, normalisation)
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm < tolerance:
            return unknowns, iteration, residual_norm, True
        jacobian = fold_defining_jacobian(unknowns, omega, structure, normalisation)
        unknowns = unknowns - np.linalg.solve(jacobian, residual)
    final = fold_defining_residual(unknowns, omega, structure, normalisation)
    return unknowns, max_iterations, float(np.max(np.abs(final))), False


def _validate(omega: NDArray[np.float64], structure: NDArray[np.float64]) -> int:
    """Validate the problem shapes and return the oscillator count."""
    if omega.ndim != 1:
        raise ValueError("omega must be one-dimensional")
    count = int(omega.shape[0])
    if count < 2:
        raise ValueError(f"need at least two oscillators, got {count}")
    if structure.shape != (count, count):
        raise ValueError(f"structure must be {(count, count)}, got {structure.shape}")
    return count


def locate_saddle_node(
    omega: NDArray[np.float64],
    structure: NDArray[np.float64],
    *,
    initial_coupling: float,
    initial_phases: NDArray[np.float64] | None = None,
    continuation_step: float = 0.02,
    newton_tolerance: float = 1.0e-11,
    max_newton_iterations: int = 60,
    max_continuation_steps: int = 100_000,
) -> SaddleNodePoint:
    r"""Locate the saddle-node fold of the Kuramoto phase-locked branch.

    Solves the locked state at ``initial_coupling`` (which must lie above the
    fold, on the locked branch), continues it down in coupling until the branch
    folds away to bracket the fold, seeds the null vector from the reduced
    Jacobian at the last locked state, and refines to the fold by Newton on the
    Moore–Spence defining system.

    Parameters
    ----------
    omega:
        Natural frequencies, shape ``(N,)``, ``N >= 2``.
    structure:
        The coupling pattern ``A_ij``, shape ``(N, N)``; the coupling matrix at
        strength ``K`` is ``K * structure``. A zero diagonal is assumed.
    initial_coupling:
        The coupling strength to start from; must be above the fold so a locked
        state exists there.
    initial_phases:
        Optional seed for the locked phases at ``initial_coupling`` (length
        ``N``); defaults to the aligned state ``0``.
    continuation_step:
        Coupling decrement per natural-parameter continuation step.
    newton_tolerance:
        Convergence tolerance (infinity norm) for the fold Newton.
    max_newton_iterations:
        Maximum fold-Newton iterations.
    max_continuation_steps:
        Safety cap on the bracketing continuation.

    Returns
    -------
    SaddleNodePoint
        The located fold.

    Raises
    ------
    ValueError
        If the shapes are invalid, or no locked state exists at
        ``initial_coupling``.
    RuntimeError
        If the continuation reaches zero coupling without bracketing a fold.
    """
    count = _validate(omega, structure)
    if initial_coupling <= 0.0:
        raise ValueError(f"initial_coupling must be positive, got {initial_coupling}")
    if continuation_step <= 0.0:
        raise ValueError(f"continuation_step must be positive, got {continuation_step}")

    if initial_phases is None:
        free = np.zeros(count - 1, dtype=np.float64)
    else:
        phases = np.asarray(initial_phases, dtype=np.float64)
        if phases.shape != (count,):
            raise ValueError(f"initial_phases must be {(count,)}, got {phases.shape}")
        free = phases[1:] - phases[0]
    seed = np.concatenate([free, [float(np.mean(omega))]])

    locked = _solve_locked_state(
        omega,
        structure,
        initial_coupling,
        seed,
        tolerance=newton_tolerance,
        max_iterations=max_newton_iterations,
    )
    if locked is None:
        raise ValueError(
            f"no phase-locked state at initial_coupling={initial_coupling}; "
            "start from a larger coupling"
        )

    strength = float(initial_coupling)
    for _ in range(max_continuation_steps):
        next_strength = strength - continuation_step
        if next_strength <= 0.0:
            break
        stepped = _solve_locked_state(
            omega,
            structure,
            next_strength,
            locked,
            tolerance=newton_tolerance,
            max_iterations=max_newton_iterations,
        )
        if stepped is None:
            theta = np.concatenate([[0.0], locked[: count - 1]])
            reduced = _reduced_jacobian(networked_kuramoto_jacobian(theta, strength * structure))
            null_vector = np.linalg.svd(reduced)[2][-1]
            normalisation = null_vector.copy()
            fold_seed = np.concatenate(
                [locked[: count - 1], [locked[count - 1]], null_vector, [strength]]
            )
            solution, iterations, residual_norm, converged = _newton_fold(
                fold_seed,
                omega,
                structure,
                normalisation,
                tolerance=newton_tolerance,
                max_iterations=max_newton_iterations,
            )
            return SaddleNodePoint(
                critical_coupling=float(solution[-1]),
                phases=np.concatenate([[0.0], solution[: count - 1]]),
                collective_frequency=float(solution[count - 1]),
                null_vector=np.asarray(solution[count : 2 * count], dtype=np.float64),
                residual_norm=residual_norm,
                iterations=iterations,
                converged=converged,
            )
        strength, locked = next_strength, stepped

    raise RuntimeError(
        "continuation reached zero coupling without bracketing a fold; "
        "the locked branch may persist for all positive coupling"
    )


__all__ = [
    "SaddleNodePoint",
    "fold_defining_jacobian",
    "fold_defining_residual",
    "locate_saddle_node",
]
