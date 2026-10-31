# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — implicit root and KKT sensitivities at a verified solution
"""Implicit root and KKT sensitivities, issued only at a verified solution.

The implicit function theorem gives ``dx*/dp = -J_x^-1 J_p`` at a root of
``F(x, p) = 0``, and its KKT analogue at a constrained optimum. Both formulas
are exact only where the solution has actually been reached, the linear system
is well conditioned and, with constraints, the active set stays fixed under a
small perturbation. This module checks those conditions before the linear solve
and otherwise raises :class:`ImplicitDerivativeRefusal` with a typed reason.

It never differentiates the iterations that produced the solution. That
derivative of an unrolled optimiser is a different object: it depends on the
iteration count and the starting point, and only approaches the implicit one as
the iterations converge.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ImplicitRefusalReason = Literal[
    "incomplete_solve", "singular", "active_set_switch", "not_a_strict_minimum"
]
IMPLICIT_DERIVATIVE_KIND: Final[str] = "implicit_at_verified_solution"
"""Kind of every result here: the implicit derivative at a checked solution, not an unrolled one."""

ResidualFunction = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
HessianVectorProduct = Callable[[NDArray[np.float64]], ArrayLike]


class ImplicitDerivativeRefusal(ValueError):
    """A solution at which no implicit derivative is issued, with a typed reason.

    ``incomplete_solve``: the residual or KKT conditions are not met.
    ``singular``: the Jacobian or KKT matrix is singular or ill-conditioned,
    or the active constraint gradients are linearly dependent.
    ``active_set_switch``: a constraint is weakly active, so an arbitrarily
    small perturbation can change the active set and the one-sided derivatives
    differ. ``not_a_strict_minimum``: the reduced Hessian is not positive
    definite, so the minimiser is not unique or not a minimum.
    """

    def __init__(self, reason: ImplicitRefusalReason, message: str) -> None:
        """Store the typed reason beside the message.

        Parameters
        ----------
        reason
            One of the :data:`ImplicitRefusalReason` values.
        message
            Human-readable explanation.

        """
        super().__init__(message)
        self.reason: ImplicitRefusalReason = reason


@dataclass(frozen=True)
class RootSensitivity:
    """Implicit derivative of a verified root of ``F(x, p) = 0``.

    Attributes
    ----------
    solution
        The root ``x*`` (length ``n``).
    sensitivity
        ``dx*/dp``, shape ``(n, m)``.
    residual_norm
        ``max |F(x*, p)|``, at most the admitted tolerance.
    condition_number
        2-norm condition number of ``dF/dx`` at the root.
    method, kind
        Provenance labels.

    """

    solution: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    residual_norm: float
    condition_number: float
    method: str = "implicit_root_sensitivity"
    kind: str = IMPLICIT_DERIVATIVE_KIND


@dataclass(frozen=True)
class KktSensitivity:
    """Implicit derivative of a verified constrained optimum with a fixed active set.

    The problem is ``min f(x, p)`` subject to ``g(x, p) <= 0`` and
    ``h(x, p) = 0`` with Lagrangian ``f + lambda^T g + mu^T h``.

    Attributes
    ----------
    solution
        The optimum ``x*`` (length ``n``).
    sensitivity
        ``dx*/dp``, shape ``(n, m)``.
    multipliers
        Inequality multipliers ``lambda`` (length ``k``).
    multiplier_sensitivity
        ``dlambda/dp``, shape ``(k, m)``; zero rows for inactive constraints.
    equality_multipliers
        Equality multipliers ``mu`` (length ``q``).
    equality_multiplier_sensitivity
        ``dmu/dp``, shape ``(q, m)``.
    active
        Whether each inequality is active.
    constraint_values
        ``g(x*, p)`` (length ``k``).
    constraint_sensitivity
        Total derivative ``dg/dp = G_x dx*/dp + G_p``, shape ``(k, m)``.
    kkt_residual
        Largest violation of stationarity, feasibility, dual feasibility and
        complementarity.
    condition_number
        2-norm condition number of the KKT matrix of the active set.
    reduced_hessian_min_eigenvalue
        Smallest eigenvalue of the Lagrangian Hessian on the null space of the
        active constraint gradients, or ``inf`` when that space is ``{0}``.
    method, kind
        Provenance labels.

    """

    solution: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    multipliers: NDArray[np.float64]
    multiplier_sensitivity: NDArray[np.float64]
    equality_multipliers: NDArray[np.float64]
    equality_multiplier_sensitivity: NDArray[np.float64]
    active: tuple[bool, ...]
    constraint_values: NDArray[np.float64]
    constraint_sensitivity: NDArray[np.float64]
    kkt_residual: float
    condition_number: float
    reduced_hessian_min_eigenvalue: float
    method: str = "implicit_kkt_sensitivity"
    kind: str = IMPLICIT_DERIVATIVE_KIND

    def step_limit(self, direction: ArrayLike) -> float:
        """Return how far ``p`` may move along ``direction`` before the active set changes.

        To first order an inactive constraint reaches zero at
        ``-g_i / (dg_i/dp . d)`` and an active multiplier reaches zero at
        ``lambda_i / -(dlambda_i/dp . d)``; the smallest positive step is
        returned, or ``inf`` when neither happens along ``d``. The prediction
        is exact for a quadratic objective with constraints affine in
        ``(x, p)`` and first order otherwise; beyond it the derivative here no
        longer describes the solution.

        Parameters
        ----------
        direction
            Parameter direction ``d`` (length ``m``), finite and nonzero.

        Returns
        -------
        float
            The step ``t >= 0`` along ``d`` at which the active set first changes.

        Raises
        ------
        ValueError
            If ``direction`` has the wrong length, is not finite or is zero.

        """
        step = np.asarray(direction, dtype=np.float64)
        if step.shape != (self.sensitivity.shape[1],) or not np.all(np.isfinite(step)):
            raise ValueError(
                f"direction must be a finite vector of length {self.sensitivity.shape[1]}"
            )
        if not np.any(step):
            raise ValueError("direction must be nonzero")
        limit = math.inf
        for index, is_active in enumerate(self.active):
            if is_active:
                rate = float(self.multiplier_sensitivity[index] @ step)
                if rate < 0.0:
                    limit = min(limit, float(self.multipliers[index]) / -rate)
            else:
                rate = float(self.constraint_sensitivity[index] @ step)
                if rate > 0.0:
                    limit = min(limit, -float(self.constraint_values[index]) / rate)
        return limit


def _vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return array


def _matrix(name: str, value: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {shape} matrix, got shape {array.shape}")
    return array


def _tolerance(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def implicit_root_sensitivity(
    residual: ResidualFunction,
    state_jacobian: ResidualFunction,
    parameter_jacobian: ResidualFunction,
    solution: ArrayLike,
    parameters: ArrayLike,
    *,
    residual_tolerance: float = 1.0e-10,
    rcond: float = 1.0e-12,
) -> RootSensitivity:
    """Differentiate a root of ``F(x, p) = 0`` after verifying it.

    Parameters
    ----------
    residual
        ``F(x, p)`` returning a vector of length ``n``.
    state_jacobian
        ``dF/dx`` returning an ``(n, n)`` matrix.
    parameter_jacobian
        ``dF/dp`` returning an ``(n, m)`` matrix.
    solution
        The claimed root ``x*`` (length ``n``).
    parameters
        ``p`` (length ``m``).
    residual_tolerance
        Largest admitted ``max |F(x*, p)|``, absolute.
    rcond
        Reciprocal of the largest admitted condition number of ``dF/dx``.

    Returns
    -------
    RootSensitivity
        ``dx*/dp = -(dF/dx)^-1 dF/dp`` with the residual and conditioning it was issued at.

    Raises
    ------
    ImplicitDerivativeRefusal
        ``incomplete_solve`` when the residual exceeds the tolerance or is not
        finite; ``singular`` when ``dF/dx`` is singular or ill-conditioned.
    ValueError
        If the inputs or returned shapes are malformed.

    """
    point = _vector("solution", solution)
    values = _vector("parameters", parameters)
    tolerance = _tolerance("residual_tolerance", residual_tolerance)
    threshold = 1.0 / _tolerance("rcond", rcond)
    n, m = point.size, values.size
    if n == 0 or m == 0:
        raise ValueError("solution and parameters must be non-empty")
    remainder = np.asarray(residual(point, values), dtype=np.float64)
    if remainder.shape != (n,):
        raise ValueError(f"residual must return shape ({n},), got {remainder.shape}")
    norm = float(np.max(np.abs(remainder)))
    if not math.isfinite(norm) or norm > tolerance:
        raise ImplicitDerivativeRefusal(
            "incomplete_solve",
            f"the residual {norm:.3e} exceeds {tolerance:.3e}; the point is not a verified root",
        )
    jacobian = _matrix("state_jacobian", state_jacobian(point, values), (n, n))
    cross = _matrix("parameter_jacobian", parameter_jacobian(point, values), (n, m))
    condition = float(np.linalg.cond(jacobian))
    if not math.isfinite(condition) or condition > threshold:
        raise ImplicitDerivativeRefusal(
            "singular",
            f"dF/dx has condition number {condition:.3e} (limit {threshold:.3e}); "
            "the root is not locally a smooth function of p",
        )
    return RootSensitivity(
        solution=point,
        sensitivity=np.asarray(-np.linalg.solve(jacobian, cross), dtype=np.float64),
        residual_norm=norm,
        condition_number=condition,
    )


def _hessian(value: ArrayLike | HessianVectorProduct, n: int) -> NDArray[np.float64]:
    """Return the Lagrangian Hessian, materialising a Hessian-vector product on unit vectors."""
    if callable(value):
        columns = [np.asarray(value(unit), dtype=np.float64) for unit in np.eye(n)]
        if any(column.shape != (n,) for column in columns):
            raise ValueError(f"the Hessian-vector product must return vectors of length {n}")
        matrix = np.column_stack(columns) if columns else np.zeros((0, 0))
    else:
        matrix = np.asarray(value, dtype=np.float64)
    matrix = _matrix("lagrangian_hessian", matrix, (n, n))
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 1.0)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-10 * scale):
        raise ValueError("lagrangian_hessian must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _constraints(
    name: str,
    values: ArrayLike | None,
    jacobian: ArrayLike | None,
    parameter_jacobian: ArrayLike | None,
    multipliers: ArrayLike | None,
    n: int,
    m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate one constraint block: values, Jacobians and multipliers, all given or none."""
    if values is None and jacobian is None and parameter_jacobian is None and multipliers is None:
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros((0, n)), np.zeros((0, m)), empty
    if values is None or jacobian is None or parameter_jacobian is None or multipliers is None:
        raise ValueError(f"{name} values, Jacobians and multipliers must be given together")
    level = _vector(f"{name} values", values)
    count = level.size
    weights = _vector(f"{name} multipliers", multipliers)
    if weights.size != count:
        raise ValueError(f"{name} multipliers must have length {count}")
    return (
        level,
        _matrix(f"{name} jacobian", jacobian, (count, n)),
        _matrix(f"{name} parameter jacobian", parameter_jacobian, (count, m)),
        weights,
    )


def implicit_kkt_sensitivity(
    solution: ArrayLike,
    objective_gradient: ArrayLike,
    lagrangian_hessian: ArrayLike | HessianVectorProduct,
    cross_derivative: ArrayLike,
    *,
    constraint_values: ArrayLike | None = None,
    constraint_jacobian: ArrayLike | None = None,
    constraint_parameter_jacobian: ArrayLike | None = None,
    multipliers: ArrayLike | None = None,
    equality_values: ArrayLike | None = None,
    equality_jacobian: ArrayLike | None = None,
    equality_parameter_jacobian: ArrayLike | None = None,
    equality_multipliers: ArrayLike | None = None,
    kkt_tolerance: float = 1.0e-9,
    active_tolerance: float = 1.0e-7,
    rcond: float = 1.0e-12,
) -> KktSensitivity:
    """Differentiate a constrained optimum after verifying it and its active set.

    The problem is ``min f(x, p)`` subject to ``g(x, p) <= 0`` and
    ``h(x, p) = 0``, with Lagrangian ``L = f + lambda^T g + mu^T h``. The
    point is first checked against the KKT conditions. Activity is decided
    without regard to how a constraint is scaled: constraint ``i`` is active
    when its distance to the bound ``-g_i / |dg_i/dx|`` is at most
    ``active_tolerance``, and its multiplier counts when its force
    ``lambda_i |dg_i/dx|`` exceeds ``active_tolerance`` times the largest of
    ``max |df/dx|`` and every force. Every active constraint must have a
    multiplier that counts, and no inactive one may; the active constraint
    gradients must be linearly independent and the Lagrangian Hessian
    positive definite on their null space. Only then is the KKT system of the
    fixed active set solved for ``dx*/dp`` and the multiplier derivatives.

    Parameters
    ----------
    solution
        ``x*`` (length ``n``).
    objective_gradient
        ``df/dx`` at ``x*`` (length ``n``).
    lagrangian_hessian
        ``d2L/dx2`` at ``x*``: an ``(n, n)`` symmetric matrix, or a
        Hessian-vector product ``v -> (d2L/dx2) v`` that is applied to the
        ``n`` unit vectors and must give a symmetric matrix.
    cross_derivative
        ``d2L/dx dp`` at ``x*``, shape ``(n, m)``.
    constraint_values, constraint_jacobian, constraint_parameter_jacobian, multipliers
        ``g`` (length ``k``), ``dg/dx`` ``(k, n)``, ``dg/dp`` ``(k, m)`` and
        ``lambda`` (length ``k``); all four or none.
    equality_values, equality_jacobian, equality_parameter_jacobian, equality_multipliers
        ``h`` (length ``q``), ``dh/dx`` ``(q, n)``, ``dh/dp`` ``(q, m)`` and
        ``mu`` (length ``q``); all four or none.
    kkt_tolerance
        Largest admitted violation of stationarity, primal and dual
        feasibility and complementarity, absolute.
    active_tolerance
        Largest distance to a bound, in units of ``x``, at which a constraint
        is active, and the relative floor below which a multiplier's force
        does not count (see above).
    rcond
        Reciprocal of the largest admitted condition number of the KKT
        matrix; also the relative threshold for independent active gradients
        and for a positive reduced Hessian, whose smallest eigenvalue must
        exceed ``rcond`` times the largest ``|eig|`` of the Lagrangian Hessian.

    Returns
    -------
    KktSensitivity
        ``dx*/dp``, the multiplier derivatives and the evidence they were issued at.

    Raises
    ------
    ImplicitDerivativeRefusal
        ``incomplete_solve`` when a KKT condition is violated;
        ``active_set_switch`` when a constraint is weakly active or an
        inactive one carries a multiplier; ``singular`` when the active
        gradients are dependent or the KKT matrix is ill-conditioned;
        ``not_a_strict_minimum`` when the reduced Hessian is not positive
        definite.
    ValueError
        If shapes or values are malformed.

    """
    point = _vector("solution", solution)
    n = point.size
    if n == 0:
        raise ValueError("solution must be non-empty")
    gradient = _vector("objective_gradient", objective_gradient)
    if gradient.size != n:
        raise ValueError(f"objective_gradient must have length {n}")
    cross = np.asarray(cross_derivative, dtype=np.float64)
    if cross.ndim == 1:
        cross = cross.reshape((-1, 1))
    if cross.ndim != 2 or cross.shape[0] != n or cross.shape[1] == 0:
        raise ValueError(f"cross_derivative must have shape ({n}, m) with m >= 1")
    m = cross.shape[1]
    cross = _matrix("cross_derivative", cross, (n, m))
    hessian = _hessian(lagrangian_hessian, n)
    tolerance = _tolerance("kkt_tolerance", kkt_tolerance)
    margin = _tolerance("active_tolerance", active_tolerance)
    relative = _tolerance("rcond", rcond)
    g, g_x, g_p, lam = _constraints(
        "constraint",
        constraint_values,
        constraint_jacobian,
        constraint_parameter_jacobian,
        multipliers,
        n,
        m,
    )
    h, h_x, h_p, mu = _constraints(
        "equality",
        equality_values,
        equality_jacobian,
        equality_parameter_jacobian,
        equality_multipliers,
        n,
        m,
    )
    violations = [
        float(np.max(np.abs(gradient + g_x.T @ lam + h_x.T @ mu))),
        float(np.max(g, initial=0.0)),
        float(np.max(np.abs(h), initial=0.0)),
        float(np.max(-lam, initial=0.0)),
        float(np.max(np.abs(lam * g), initial=0.0)),
    ]
    residual = max(violations)
    if residual > tolerance:
        raise ImplicitDerivativeRefusal(
            "incomplete_solve",
            f"the KKT residual {residual:.3e} exceeds {tolerance:.3e} "
            "(stationarity, primal, equality, dual, complementarity: "
            f"{', '.join(f'{value:.2e}' for value in violations)}); the point is not a verified optimum",
        )
    # Scale-free activity: rescaling a constraint by s scales g by s and lambda by 1/s, so the
    # distance -g/|dg/dx| (in x) and the force lambda |dg/dx| (in gradient units) are invariant.
    norms = np.linalg.norm(g_x, axis=1)
    norms = np.where(norms > 0.0, norms, 1.0)
    distance = -g / norms
    force = lam * norms
    force_scale = max(
        float(np.max(np.abs(gradient), initial=0.0)), float(np.max(force, initial=0.0))
    )
    active = distance <= margin
    strong = force > margin * force_scale
    weak = np.flatnonzero(active & ~strong)
    loaded = np.flatnonzero(~active & strong)
    if weak.size or loaded.size:
        raise ImplicitDerivativeRefusal(
            "active_set_switch",
            f"constraints {weak.tolist()} are weakly active and {loaded.tolist()} are inactive with "
            "a multiplier; a small perturbation can change the active set, so the derivative is "
            "one-sided",
        )
    rows = np.vstack([g_x[active], h_x])
    parameter_rows = np.vstack([g_p[active], h_p])
    count = rows.shape[0]
    if count:
        singular_values = np.linalg.svd(rows, compute_uv=False)
        if count > n or singular_values[-1] <= relative * singular_values[0]:
            raise ImplicitDerivativeRefusal(
                "singular",
                "the active constraint gradients are linearly dependent; the multipliers are not "
                "unique",
            )
        basis = np.linalg.svd(rows)[2][count:].T
    else:
        basis = np.eye(n)
    if basis.shape[1]:
        reduced = np.linalg.eigvalsh(basis.T @ hessian @ basis)
        scale = float(np.max(np.abs(np.linalg.eigvalsh(hessian))))
        lowest = float(reduced[0])
        if lowest <= relative * scale:
            raise ImplicitDerivativeRefusal(
                "not_a_strict_minimum",
                f"the reduced Hessian has smallest eigenvalue {lowest:.3e}; the minimiser is not "
                "unique or not a minimum",
            )
    else:
        lowest = math.inf
    system = np.block([[hessian, rows.T], [rows, np.zeros((count, count))]])
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > 1.0 / relative:
        raise ImplicitDerivativeRefusal(
            "singular",
            f"the KKT matrix has condition number {condition:.3e} (limit {1.0 / relative:.3e})",
        )
    solved = np.linalg.solve(system, -np.vstack([cross, parameter_rows]))
    sensitivity = np.asarray(solved[:n], dtype=np.float64)
    multiplier_sensitivity = np.zeros((g.size, m), dtype=np.float64)
    active_count = int(np.count_nonzero(active))
    multiplier_sensitivity[active] = solved[n : n + active_count]
    return KktSensitivity(
        solution=point,
        sensitivity=sensitivity,
        multipliers=lam,
        multiplier_sensitivity=multiplier_sensitivity,
        equality_multipliers=mu,
        equality_multiplier_sensitivity=np.asarray(solved[n + active_count :], dtype=np.float64),
        active=tuple(bool(flag) for flag in active),
        constraint_values=g,
        constraint_sensitivity=np.asarray(g_x @ sensitivity + g_p, dtype=np.float64),
        kkt_residual=residual,
        condition_number=condition,
        reduced_hessian_min_eigenvalue=lowest,
    )


__all__ = [
    "IMPLICIT_DERIVATIVE_KIND",
    "ImplicitDerivativeRefusal",
    "ImplicitRefusalReason",
    "KktSensitivity",
    "RootSensitivity",
    "implicit_kkt_sensitivity",
    "implicit_root_sensitivity",
]
