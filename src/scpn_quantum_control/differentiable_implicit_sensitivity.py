# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable implicit sensitivity module
# scpn-quantum-control -- implicit sensitivity solvers
"""Implicit stationary, fixed-point and explicit-solver sensitivities.

The solver part separates two different objects. The derivative of the
discretised solver is the exact derivative of one fixed-step map at one mesh.
The continuous sensitivity is the limit as the step goes to zero. Equal discrete
derivatives from two implementations never establish the continuous one; only
mesh-refinement evidence does. A failed solve or a crossed event yields a typed
refusal, never a gradient.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import (
    Parameter,
    _as_real_numeric_array,
    _as_real_scalar,
)
from .differentiable_result_contracts import (
    FixedPointSensitivityResult,
    ImplicitSensitivityResult,
)
from .differentiable_transform_helpers import _normalise_parameters


def implicit_stationary_sensitivity(
    hessian: ArrayLike,
    cross_derivative: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> ImplicitSensitivityResult:
    """Return sensitivities for an implicit stationary optimum.

    Parameters
    ----------
    hessian
        Symmetric positive-definite Hessian ``H`` of the stationarity
        equations with respect to trainable parameters.
    cross_derivative
        Cross derivative matrix ``B`` with one row per parameter and one
        column per hyperparameter. One-dimensional inputs are treated as a
        single hyperparameter column.
    parameters
        Optional parameter metadata. Frozen parameters receive zero
        sensitivity and are excluded from the trainable linear solve.
    hyperparameter_names
        Optional names for the cross-derivative columns. Defaults to
        ``alpha0``, ``alpha1``, and so on.
    damping
        Non-negative diagonal damping applied to the active Hessian before the
        solve.
    rcond
        Positive reciprocal-condition threshold used to reject ill-conditioned
        trainable Hessian blocks.

    Returns
    -------
    ImplicitSensitivityResult
        Validated ``dx*/dalpha = -H^-1 B`` sensitivity metadata.

    """
    hessian_arr = _as_real_numeric_array("implicit hessian", hessian)
    cross = _as_real_numeric_array("implicit cross_derivative", cross_derivative)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        raise ValueError("implicit hessian must be a square matrix")
    if cross.ndim == 1:
        cross = cross.reshape((-1, 1))
    if cross.ndim != 2 or cross.shape[0] != hessian_arr.shape[0]:
        raise ValueError("implicit cross_derivative row count must match hessian dimension")
    if not np.all(np.isfinite(hessian_arr)) or not np.all(np.isfinite(cross)):
        raise ValueError("implicit operands must contain only finite values")
    if not np.allclose(hessian_arr, hessian_arr.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("implicit hessian must be symmetric")
    damping_value = _as_real_scalar("implicit damping", damping)
    if damping_value < 0.0:
        raise ValueError("implicit damping must be finite and non-negative")
    rcond_value = _as_real_scalar("implicit rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("implicit rcond must be finite and positive")
    parameter_values = np.zeros(hessian_arr.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(cross.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != cross.shape[1]:
        raise ValueError("hyperparameter_names length must match cross_derivative columns")
    trainable = np.asarray([parameter.trainable for parameter in parameter_meta], dtype=bool)
    sensitivity = np.zeros_like(cross)
    condition_number = 1.0
    if np.any(trainable):
        active_hessian = hessian_arr[np.ix_(trainable, trainable)].copy()
        if damping_value > 0.0:
            active_hessian += damping_value * np.eye(active_hessian.shape[0], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(active_hessian)
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))
        if min_eigenvalue <= 0.0:
            raise ValueError("implicit hessian must be positive definite on trainable parameters")
        condition_number = max_eigenvalue / min_eigenvalue
        if condition_number > 1.0 / rcond_value:
            raise ValueError("implicit hessian is ill-conditioned")
        sensitivity[trainable, :] = -np.linalg.solve(active_hessian, cross[trainable, :])
    return ImplicitSensitivityResult(
        sensitivity=sensitivity,
        hessian=hessian_arr,
        cross_derivative=cross,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_stationary_sensitivity",
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
        hyperparameter_names=hyper_names,
    )


def implicit_fixed_point_sensitivity(
    state_jacobian: ArrayLike,
    parameter_jacobian: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    hyperparameter_names: Sequence[str] | None = None,
    damping: float = 0.0,
    rcond: float = 1.0e-12,
) -> FixedPointSensitivityResult:
    """Return sensitivities for a fixed point ``x* = T(x*, alpha)``.

    Parameters
    ----------
    state_jacobian
        Square Jacobian ``dT/dx`` evaluated at the fixed point.
    parameter_jacobian
        Jacobian ``dT/dalpha`` with one row per state entry and one column per
        hyperparameter. One-dimensional inputs are treated as a single
        hyperparameter column.
    parameters
        Optional state-parameter metadata. Frozen entries receive zero
        sensitivity and are excluded from the active linear solve.
    hyperparameter_names
        Optional names for parameter-Jacobian columns. Defaults to ``alpha0``,
        ``alpha1``, and so on.
    damping
        Non-negative diagonal damping added to ``I - dT/dx``.
    rcond
        Positive reciprocal-condition threshold used to reject ill-conditioned
        active systems.

    Returns
    -------
    FixedPointSensitivityResult
        Validated ``(I - dT/dx)^-1 dT/dalpha`` sensitivity metadata.

    """
    state = _as_real_numeric_array("fixed-point state_jacobian", state_jacobian)
    parameter = _as_real_numeric_array("fixed-point parameter_jacobian", parameter_jacobian)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("fixed-point state_jacobian must be a square matrix")
    if parameter.ndim == 1:
        parameter = parameter.reshape((-1, 1))
    if parameter.ndim != 2 or parameter.shape[0] != state.shape[0]:
        raise ValueError("fixed-point parameter_jacobian row count must match state dimension")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(parameter)):
        raise ValueError("fixed-point operands must contain only finite values")
    damping_value = _as_real_scalar("fixed-point damping", damping)
    if damping_value < 0.0:
        raise ValueError("fixed-point damping must be finite and non-negative")
    rcond_value = _as_real_scalar("fixed-point rcond", rcond)
    if rcond_value <= 0.0:
        raise ValueError("fixed-point rcond must be finite and positive")
    parameter_values = np.zeros(state.shape[0], dtype=np.float64)
    parameter_meta = _normalise_parameters(parameter_values, parameters)
    hyper_names = (
        tuple(f"alpha{index}" for index in range(parameter.shape[1]))
        if hyperparameter_names is None
        else tuple(hyperparameter_names)
    )
    if len(hyper_names) != parameter.shape[1]:
        raise ValueError(
            "hyperparameter_names length must match fixed-point parameter_jacobian columns"
        )
    system_matrix = np.eye(state.shape[0], dtype=np.float64) - state
    if damping_value > 0.0:
        system_matrix = system_matrix + damping_value * np.eye(state.shape[0], dtype=np.float64)
    trainable = np.asarray(
        [parameter_info.trainable for parameter_info in parameter_meta], dtype=bool
    )
    sensitivity = np.zeros_like(parameter)
    condition_number = 1.0
    if np.any(trainable):
        active_system = system_matrix[np.ix_(trainable, trainable)]
        condition_number = float(np.linalg.cond(active_system))
        if not np.isfinite(condition_number) or condition_number > 1.0 / rcond_value:
            raise ValueError("fixed-point system is ill-conditioned")
        sensitivity[trainable, :] = np.linalg.solve(active_system, parameter[trainable, :])
    return FixedPointSensitivityResult(
        sensitivity=sensitivity,
        state_jacobian=state,
        parameter_jacobian=parameter,
        system_matrix=system_matrix,
        damping=damping_value,
        condition_number=condition_number,
        method="implicit_fixed_point_sensitivity",
        parameter_names=tuple(parameter_info.name for parameter_info in parameter_meta),
        trainable=tuple(parameter_info.trainable for parameter_info in parameter_meta),
        hyperparameter_names=hyper_names,
    )


# --- Explicit fixed-step solver sensitivity -------------------------------------------

StateFunction = Callable[[float, NDArray[np.float64], NDArray[np.float64]], ArrayLike]
SolverRefusalReason = Literal["failed_solve", "event_crossing", "not_refined"]

RK4_ORDER: Final[int] = 4
"""Global order of the classical fixed-step Runge-Kutta method."""
DELAYED_METHOD_OF_STEPS_ORDERS: Final[tuple[int, ...]] = (2, 4)
"""Global orders the delayed method-of-steps integrator shows under step halving.

Lagged values between grid points are interpolated, which limits the order to
two in general; where that interpolation is exact (a linear history without
coupling feedback) the RK4 order four shows through.
"""
SOLVER_DERIVATIVE_SUPPORT: Final[dict[str, str]] = {
    "fixed_step_rk4": (
        "supported: rk4_forward_sensitivity is the exact derivative of the discretised map; "
        "refine_solver_sensitivity is the only route to the continuous sensitivity"
    ),
    "adaptive_accept_reject": (
        "qualified only by tolerance refinement: kuramoto_dopri_vjp differentiates the "
        "realised step sequence with every accept/reject decision held fixed, so it is the "
        "derivative of that discrete map, not of the flow or of the step controller; "
        "qualify_adaptive_dopri_gradient issues a continuous claim only from tolerance "
        "refinement evidence"
    ),
    "event_time": (
        "refused: a crossed event raises SolverDerivativeRefusal; a continuous claim is also "
        "refused when the event comes within two one-step changes of zero on the finest mesh. "
        "The event is sampled only at mesh points: a feature of g narrower than one step can "
        "cross zero and return unseen, so the continuous claim assumes g is resolved by the "
        "finest mesh"
    ),
    "delayed_history": (
        "qualified only by step refinement: delayed_phase_sensitivity is the exact derivative "
        "of the discrete method-of-steps map; qualify_delayed_sensitivity issues a continuous "
        "claim for the frequency and coupling channels only from step-halving evidence at "
        "a consistent observed order of two or four"
    ),
    "failed_solve": (
        "refused: non-finite or diverging states or sensitivities raise SolverDerivativeRefusal"
    ),
    "open_system_objectives": (
        "discrete only: phase.open_system_objectives reports central finite differences of the "
        "time-discretised Lindblad solve at the case dt and of a seeded MCWF ensemble; its "
        "claim boundary excludes adjoint, continuous-time and unbiased stochastic gradients"
    ),
}
"""What each solver control decision means for the derivative."""


class SolverDerivativeRefusal(ValueError):
    """A solve that cannot yield a qualified derivative, with a typed reason."""

    def __init__(self, reason: SolverRefusalReason, message: str) -> None:
        """Store the typed reason beside the message.

        Parameters
        ----------
        reason
            ``"failed_solve"``, ``"event_crossing"`` or ``"not_refined"``.
        message
            Human-readable explanation.

        """
        super().__init__(message)
        self.reason: SolverRefusalReason = reason


@dataclass(frozen=True)
class DiscreteSolverSensitivity:
    """Exact derivative of the fixed-step RK4 map at one mesh.

    This is the derivative of the discretised solver, not the continuous
    sensitivity; :func:`refine_solver_sensitivity` is the only route to the
    latter. ``event_margin`` is the smallest ``|g|`` over the mesh points
    divided by the largest one-step change of ``g``, or ``None`` without an
    event.
    """

    final_state: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    steps: int
    final_time: float
    method: str = "fixed_step_rk4_discrete_derivative"
    kind: str = "discrete"
    event_margin: float | None = None


@dataclass(frozen=True)
class ContinuousSolverSensitivity:
    """Continuous sensitivity estimated only from mesh-refinement evidence.

    ``observed_orders`` holds one order per consecutive pair of mesh
    differences; every one of them was within the order tolerance.
    ``tolerance`` is absolute, on every sensitivity entry.
    """

    sensitivity: NDArray[np.float64]
    meshes: tuple[int, ...]
    discrete_sensitivities: tuple[NDArray[np.float64], ...]
    observed_order: float
    error_estimate: float
    tolerance: float
    method: str = "richardson_refined_rk4_sensitivity"
    kind: str = "continuous_refined"
    observed_orders: tuple[float, ...] = ()


@dataclass(frozen=True)
class QualifiedAdaptiveGradient:
    """Fixed-grid Dormand-Prince adjoint gradients qualified by tolerance refinement.

    Attributes
    ----------
    initial_phases, omega, coupling
        Gradients at the tightest tolerance, ``dL/dtheta_0``, ``dL/domega`` and
        ``dL/dK``.
    rtols
        Relative tolerances, loosest first.
    differences
        Largest absolute change of the gradient between consecutive tolerances.
    error_estimate
        The last difference, an estimate of the error of the second-tightest
        gradient and therefore conservative for the tightest.
    tolerance
        Largest admitted error estimate, absolute.

    """

    initial_phases: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]
    rtols: tuple[float, ...]
    differences: tuple[float, ...]
    error_estimate: float
    tolerance: float
    method: str = "tolerance_refined_dopri_fixed_grid_adjoint"
    kind: str = "continuous_refined"


def _state_vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return array


def _positive_count(name: str, value: object, minimum: int) -> int:
    """Return an integer (NumPy integers included) of at least ``minimum``."""
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _evaluated(
    name: str,
    function: StateFunction,
    time: float,
    state: NDArray[np.float64],
    theta: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    with np.errstate(all="ignore"):
        value = np.asarray(function(time, state, theta), dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{name} must return shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise SolverDerivativeRefusal(
            "failed_solve", f"{name} became non-finite at t={time:.6g}; no derivative is issued"
        )
    return value


def _event_value(
    event: Callable[[float, NDArray[np.float64], NDArray[np.float64]], float],
    time: float,
    state: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """Evaluate the event function, refusing a non-finite value."""
    value = float(event(time, state, theta))
    if not math.isfinite(value):
        raise SolverDerivativeRefusal(
            "failed_solve", f"the event function is non-finite at t={time:.6g}"
        )
    return value


def rk4_forward_sensitivity(
    rhs: StateFunction,
    state_jacobian: StateFunction,
    parameter_jacobian: StateFunction,
    initial_state: ArrayLike,
    theta: ArrayLike,
    final_time: float,
    steps: int,
    *,
    initial_sensitivity: ArrayLike | None = None,
    event: Callable[[float, NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    divergence_bound: float = 1.0e12,
) -> DiscreteSolverSensitivity:
    """Differentiate the fixed-step RK4 solution of ``y' = f(t, y, theta)`` exactly.

    Parameters
    ----------
    rhs
        Right-hand side ``f(t, y, theta)`` returning a state-shaped vector.
    state_jacobian
        ``df/dy`` returning an ``(n, n)`` matrix.
    parameter_jacobian
        ``df/dtheta`` returning an ``(n, p)`` matrix.
    initial_state
        ``y(0)``, a finite vector of length ``n``.
    theta
        Parameters, a finite vector of length ``p``.
    final_time
        Positive finite integration end.
    steps
        Number of equal steps, at least one (NumPy integers are accepted).
    initial_sensitivity
        ``dy(0)/dtheta``; zero when omitted.
    event
        Optional ``g(t, y, theta)``. Reaching or changing sign between any two
        mesh states is refused; the event-time derivative is not supported. A
        non-finite value is a failed solve.
    divergence_bound
        States whose largest magnitude exceeds this bound count as a failed solve.

    Returns
    -------
    DiscreteSolverSensitivity
        Final state and ``dy_N/dtheta`` of the discrete map, propagated through
        every Runge-Kutta stage, and the event margin when an event is given.

    Raises
    ------
    SolverDerivativeRefusal
        ``failed_solve`` for non-finite or diverging states, a non-finite
        sensitivity or event value; ``event_crossing`` when the event reaches
        zero or changes sign.
    ValueError
        If the inputs or returned shapes are malformed.

    """
    state = _state_vector("initial_state", initial_state)
    parameters = _state_vector("theta", theta)
    if isinstance(final_time, bool) or not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive")
    steps = _positive_count("steps", steps, 1)
    if not math.isfinite(divergence_bound) or divergence_bound <= 0.0:
        raise ValueError("divergence_bound must be finite and positive")
    n, p = state.size, parameters.size
    sensitivity = (
        np.zeros((n, p), dtype=np.float64)
        if initial_sensitivity is None
        else np.asarray(initial_sensitivity, dtype=np.float64)
    )
    if sensitivity.shape != (n, p) or not np.all(np.isfinite(sensitivity)):
        raise ValueError(f"initial_sensitivity must be a finite ({n}, {p}) matrix")
    h = final_time / steps
    event_side = 0.0
    event_values: list[float] = []
    if event is not None:
        start = _event_value(event, 0.0, state, parameters)
        event_side = math.copysign(1.0, start)
        event_values.append(start)
        if start == 0.0:
            raise SolverDerivativeRefusal(
                "event_crossing", "the solve starts on the event surface; no derivative is issued"
            )

    def stage(
        time: float, y: NDArray[np.float64], s: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        k = _evaluated("rhs", rhs, time, y, parameters, (n,))
        jy = _evaluated("state_jacobian", state_jacobian, time, y, parameters, (n, n))
        jp = _evaluated("parameter_jacobian", parameter_jacobian, time, y, parameters, (n, p))
        return k, jy @ s + jp

    for index in range(steps):
        t = index * h
        k1, d1 = stage(t, state, sensitivity)
        k2, d2 = stage(t + h / 2, state + h / 2 * k1, sensitivity + h / 2 * d1)
        k3, d3 = stage(t + h / 2, state + h / 2 * k2, sensitivity + h / 2 * d2)
        k4, d4 = stage(t + h, state + h * k3, sensitivity + h * d3)
        with np.errstate(all="ignore"):
            state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            sensitivity = sensitivity + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        if not np.all(np.isfinite(state)) or float(np.max(np.abs(state))) > divergence_bound:
            raise SolverDerivativeRefusal(
                "failed_solve",
                f"the solution diverged by t={t + h:.6g}; no derivative is issued",
            )
        if not np.all(np.isfinite(sensitivity)):
            raise SolverDerivativeRefusal(
                "failed_solve",
                f"the sensitivity became non-finite by t={t + h:.6g}; no derivative is issued",
            )
        if event is not None:
            value = _event_value(event, t + h, state, parameters)
            event_values.append(value)
            if value * event_side <= 0.0:
                # Reaching or passing the surface, even exactly at a mesh point, is a crossing.
                raise SolverDerivativeRefusal(
                    "event_crossing",
                    f"the event reached or changed sign between t={t:.6g} and t={t + h:.6g}; "
                    "the event-time derivative is not supported",
                )
    margin: float | None = None
    if event is not None:
        changes = np.abs(np.diff(np.asarray(event_values, dtype=np.float64)))
        largest = float(np.max(changes))
        smallest = float(np.min(np.abs(event_values)))
        margin = math.inf if largest == 0.0 else smallest / largest
    return DiscreteSolverSensitivity(
        final_state=state,
        sensitivity=sensitivity,
        steps=steps,
        final_time=final_time,
        event_margin=margin,
    )


def _richardson(
    meshes: tuple[int, ...],
    discrete: tuple[NDArray[np.float64], ...],
    orders_allowed: tuple[int, ...],
    tolerance: float,
    order_tolerance: float,
) -> tuple[NDArray[np.float64], tuple[float, ...], float]:
    """Extrapolate the finest pair once every consecutive order is near one allowed order.

    The allowed order nearest to the mean observed order is used, and every
    observed order must lie within ``order_tolerance`` of it.

    Returns
    -------
    tuple
        Refined value, the observed orders and the error estimate.

    Raises
    ------
    SolverDerivativeRefusal
        ``not_refined`` when any observed order is off, a difference vanishes
        before the finest one, or the error estimate exceeds ``tolerance``.

    """
    differences = [float(np.max(np.abs(finer - coarser))) for coarser, finer in pairwise(discrete)]
    size = max(float(np.max(np.abs(item))) for item in discrete)
    exact = 64.0 * float(np.finfo(np.float64).eps) * size
    if differences[-1] <= exact:
        # Converged to rounding; the order cannot be measured and is not needed.
        return discrete[-1], (float(max(orders_allowed)),), differences[-1]
    orders: list[float] = []
    for coarse, fine in pairwise(differences):
        orders.append(0.0 if coarse <= exact else math.log2(coarse / fine))
    mean = sum(orders) / len(orders)
    order = min(orders_allowed, key=lambda allowed: abs(allowed - mean))
    factor = 2**order - 1
    error_estimate = differences[-1] / factor
    converged = all(abs(observed - order) <= order_tolerance for observed in orders)
    if not converged or error_estimate > tolerance:
        raise SolverDerivativeRefusal(
            "not_refined",
            f"meshes {meshes} show orders {tuple(round(value, 3) for value in orders)} "
            f"and error estimate {error_estimate:.3e} (tolerance {tolerance:.3e}); "
            "no continuous claim is issued",
        )
    refined = discrete[-1] + (discrete[-1] - discrete[-2]) / factor
    return refined, tuple(orders), error_estimate


def _refinement_settings(
    base_steps: object, refinements: object, tolerance: float, order_tolerance: float
) -> tuple[int, int]:
    base = _positive_count("base_steps", base_steps, 1)
    levels = _positive_count("refinements", refinements, 4)
    for name, value in (("tolerance", tolerance), ("order_tolerance", order_tolerance)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return base, levels


def refine_solver_sensitivity(
    rhs: StateFunction,
    state_jacobian: StateFunction,
    parameter_jacobian: StateFunction,
    initial_state: ArrayLike,
    theta: ArrayLike,
    final_time: float,
    *,
    base_steps: int = 16,
    refinements: int = 4,
    tolerance: float = 1.0e-8,
    order_tolerance: float = 0.5,
    initial_sensitivity: ArrayLike | None = None,
    event: Callable[[float, NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    divergence_bound: float = 1.0e12,
) -> ContinuousSolverSensitivity:
    """Estimate the continuous sensitivity only from mesh-refinement evidence.

    Parameters
    ----------
    rhs, state_jacobian, parameter_jacobian, initial_state, theta, final_time
        As for :func:`rk4_forward_sensitivity`.
    base_steps
        Coarsest mesh; each refinement doubles it.
    refinements
        Number of meshes, at least four, so that two consecutive observed
        orders can be compared; one ratio alone admits pre-asymptotic meshes.
    tolerance
        Largest admitted Richardson error estimate, absolute on every entry.
    order_tolerance
        Largest admitted distance of every observed order from four.
    initial_sensitivity, event, divergence_bound
        As for :func:`rk4_forward_sensitivity`; every mesh obeys them.

    Returns
    -------
    ContinuousSolverSensitivity
        Richardson-extrapolated sensitivity from the two finest meshes, with the
        observed orders and error estimate that justify it.

    Raises
    ------
    SolverDerivativeRefusal
        ``not_refined`` when any consecutive pair of meshes does not show
        fourth-order convergence or the error estimate exceeds ``tolerance``;
        ``event_crossing`` when the event comes within two one-step changes of
        zero on the finest mesh, since the flow could then cross between mesh
        points; any refusal of a single mesh. The event is sampled only at
        mesh points, so a feature of ``g`` narrower than one finest step can
        cross zero unseen; the continuous claim assumes the finest mesh
        resolves ``g``.
    ValueError
        If the refinement settings are malformed.

    """
    base, levels = _refinement_settings(base_steps, refinements, tolerance, order_tolerance)
    meshes = tuple(base * 2**level for level in range(levels))
    results = tuple(
        rk4_forward_sensitivity(
            rhs,
            state_jacobian,
            parameter_jacobian,
            initial_state,
            theta,
            final_time,
            steps,
            initial_sensitivity=initial_sensitivity,
            event=event,
            divergence_bound=divergence_bound,
        )
        for steps in meshes
    )
    finest_margin = results[-1].event_margin
    if finest_margin is not None and finest_margin <= 2.0:
        raise SolverDerivativeRefusal(
            "event_crossing",
            f"the event comes within {finest_margin:.3g} one-step changes of zero on the finest "
            "mesh; a crossing between mesh points cannot be excluded",
        )
    discrete = tuple(result.sensitivity for result in results)
    refined, orders, error_estimate = _richardson(
        meshes, discrete, (RK4_ORDER,), tolerance, order_tolerance
    )
    return ContinuousSolverSensitivity(
        sensitivity=refined,
        meshes=meshes,
        discrete_sensitivities=discrete,
        observed_order=orders[-1],
        error_estimate=error_estimate,
        tolerance=tolerance,
        observed_orders=orders,
    )


def qualify_adaptive_dopri_gradient(
    initial_phases: ArrayLike,
    omega: ArrayLike,
    coupling: ArrayLike,
    cotangent: ArrayLike,
    *,
    final_time: float,
    rtols: Sequence[float] = (1.0e-6, 1.0e-8, 1.0e-10),
    atol_scale: float = 1.0e-3,
    tolerance: float = 1.0e-6,
) -> QualifiedAdaptiveGradient:
    """Qualify the fixed-grid Dormand-Prince adjoint against tolerance refinement.

    The adjoint of :func:`oscillatools.accel.diff_kuramoto_dopri.kuramoto_dopri_vjp`
    differentiates the realised step sequence with every accept/reject
    decision held fixed. Its distance to the continuous sensitivity shrinks
    with the tolerance but is not estimated by the adjoint itself. This
    function runs the forward and adjoint at every tolerance, and issues the
    tightest gradient only when every consecutive change shrinks at least by
    the square root of the tolerance ratio and the last change is within
    ``tolerance``.

    Parameters
    ----------
    initial_phases, omega, coupling
        Initial phases ``theta_0`` (length ``N``), frequencies and ``(N, N)``
        coupling of the Kuramoto flow.
    cotangent
        ``dL/dtheta_T`` at the final phases, length ``N``.
    final_time
        Positive finite integration end.
    rtols
        At least three strictly decreasing relative tolerances in ``(0, 1)``.
    atol_scale
        Absolute tolerance as a multiple of each relative tolerance.
    tolerance
        Largest admitted error estimate, absolute on every gradient entry.

    Returns
    -------
    QualifiedAdaptiveGradient
        Gradients at the tightest tolerance with the refinement evidence.

    Raises
    ------
    SolverDerivativeRefusal
        ``failed_solve`` when a gradient is non-finite; ``not_refined`` when
        the changes do not shrink or the estimate exceeds ``tolerance``.
    ValueError
        If the tolerances or inputs are malformed.

    """
    from oscillatools.accel.diff_kuramoto_dopri import (
        kuramoto_dopri_trajectory,
        kuramoto_dopri_vjp,
    )

    levels = tuple(float(value) for value in rtols)
    if len(levels) < 3 or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in levels
    ):
        raise ValueError("rtols must hold at least three finite values in (0, 1)")
    if any(tighter >= looser for looser, tighter in pairwise(levels)):
        raise ValueError("rtols must be strictly decreasing")
    for name, value in (("atol_scale", atol_scale), ("tolerance", tolerance)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(final_time, bool) or not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive")
    theta0 = _state_vector("initial_phases", initial_phases)
    frequencies = _state_vector("omega", omega)
    matrix = np.asarray(coupling, dtype=np.float64)
    seed = _state_vector("cotangent", cotangent)
    gradients: list[tuple[NDArray[np.float64], ...]] = []
    for rtol in levels:
        trajectory = kuramoto_dopri_trajectory(
            theta0, frequencies, matrix, t_end=final_time, rtol=rtol, atol=rtol * atol_scale
        )
        parts = kuramoto_dopri_vjp(trajectory.phases, trajectory.steps, frequencies, matrix, seed)
        if not all(np.all(np.isfinite(part)) for part in parts):
            raise SolverDerivativeRefusal(
                "failed_solve", f"the adjoint gradient is non-finite at rtol={rtol:.3g}"
            )
        gradients.append(parts)
    flat = [np.concatenate([part.ravel() for part in parts]) for parts in gradients]
    differences = tuple(
        float(np.max(np.abs(finer - coarser))) for coarser, finer in pairwise(flat)
    )
    shrinking = all(
        finer <= coarser / math.sqrt(looser / tighter) or finer == 0.0
        for (coarser, finer), (looser, tighter) in zip(
            pairwise(differences), pairwise(levels[1:]), strict=True
        )
    )
    error_estimate = differences[-1]
    if not shrinking or error_estimate > tolerance:
        raise SolverDerivativeRefusal(
            "not_refined",
            f"tolerances {levels} give gradient changes {tuple(f'{d:.3e}' for d in differences)} "
            f"(tolerance {tolerance:.3e}); no continuous claim is issued",
        )
    initial, frequency_gradient, coupling_gradient = gradients[-1]
    return QualifiedAdaptiveGradient(
        initial_phases=initial,
        omega=frequency_gradient,
        coupling=coupling_gradient,
        rtols=levels,
        differences=differences,
        error_estimate=error_estimate,
        tolerance=tolerance,
    )


def qualify_delayed_sensitivity(
    history: Callable[[NDArray[np.float64]], ArrayLike],
    omega: ArrayLike,
    coupling: ArrayLike,
    *,
    delay: float,
    final_time: float,
    base_steps_per_delay: int = 4,
    refinements: int = 4,
    tolerance: float = 1.0e-6,
    order_tolerance: float = 0.5,
) -> ContinuousSolverSensitivity:
    """Qualify the delayed method-of-steps sensitivity against step halving.

    :func:`oscillatools.accel.diff_kuramoto_delayed.delayed_phase_sensitivity`
    is the exact derivative of the discrete delayed map at one step. The
    history channels change size with the step, so only the frequency and
    coupling channels, ``dtheta_T/d(omega, K)``, are refined. The integrator
    converges at second order in general and at fourth order where its
    interpolation of lagged values is exact. The continuous claim is issued
    only when every consecutive observed order is within ``order_tolerance``
    of the same order in :data:`DELAYED_METHOD_OF_STEPS_ORDERS`, which is then
    used for the Richardson step.

    Parameters
    ----------
    history
        Initial history: maps an array of times in ``[-delay, 0]`` to an
        ``(len(times), N)`` array of phases. It is sampled on each mesh.
    omega, coupling
        Frequencies (length ``N``) and ``(N, N)`` coupling.
    delay
        Positive coupling delay.
    final_time
        Positive end time; an integer multiple of ``delay / base_steps_per_delay``.
    base_steps_per_delay
        Steps per delay on the coarsest mesh; each refinement doubles it.
    refinements
        Number of meshes, at least four.
    tolerance
        Largest admitted error estimate, absolute on every entry.
    order_tolerance
        Largest admitted distance of every observed order from the order used.

    Returns
    -------
    ContinuousSolverSensitivity
        Richardson-refined ``dtheta_T/d(omega, K)``, columns ordered as
        ``[omega (N), K row-major (N**2)]``.

    Raises
    ------
    SolverDerivativeRefusal
        ``not_refined`` as for :func:`refine_solver_sensitivity`, at order two;
        ``failed_solve`` when a sensitivity is non-finite.
    ValueError
        If the settings are malformed or ``final_time`` is off the coarsest mesh.

    """
    from oscillatools.accel.diff_kuramoto_delayed import delayed_phase_sensitivity

    base, levels = _refinement_settings(
        base_steps_per_delay, refinements, tolerance, order_tolerance
    )
    for name, value in (("delay", delay), ("final_time", final_time)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    coarse_steps = final_time * base / delay
    if abs(coarse_steps - round(coarse_steps)) > 1e-9 * max(1.0, coarse_steps):
        raise ValueError("final_time must be an integer multiple of delay / base_steps_per_delay")
    frequencies = _state_vector("omega", omega)
    matrix = np.asarray(coupling, dtype=np.float64)
    count = frequencies.size
    meshes = tuple(base * 2**level for level in range(levels))
    discrete: list[NDArray[np.float64]] = []
    for per_delay in meshes:
        dt = delay / per_delay
        times = np.linspace(-delay, 0.0, per_delay + 1)
        sampled = np.asarray(history(times), dtype=np.float64)
        if sampled.shape != (per_delay + 1, count) or not np.all(np.isfinite(sampled)):
            raise ValueError(f"history must return a finite ({per_delay + 1}, {count}) array")
        _, sensitivity = delayed_phase_sensitivity(
            sampled,
            frequencies,
            matrix,
            delay=delay,
            dt=dt,
            n_steps=int(round(final_time / dt)),
        )
        channels = np.asarray(sensitivity[:, (per_delay + 1) * count :], dtype=np.float64)
        if not np.all(np.isfinite(channels)):
            raise SolverDerivativeRefusal(
                "failed_solve", f"the delayed sensitivity is non-finite at dt={dt:.3g}"
            )
        discrete.append(channels)
    refined, orders, error_estimate = _richardson(
        meshes, tuple(discrete), DELAYED_METHOD_OF_STEPS_ORDERS, tolerance, order_tolerance
    )
    return ContinuousSolverSensitivity(
        sensitivity=refined,
        meshes=meshes,
        discrete_sensitivities=tuple(discrete),
        observed_order=orders[-1],
        error_estimate=error_estimate,
        tolerance=tolerance,
        method="richardson_refined_delayed_method_of_steps_sensitivity",
        observed_orders=orders,
    )
