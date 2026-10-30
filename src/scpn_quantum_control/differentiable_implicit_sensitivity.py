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
SOLVER_DERIVATIVE_SUPPORT: Final[dict[str, str]] = {
    "fixed_step_rk4": "supported: exact derivative of the discretised map",
    "adaptive_accept_reject": "excluded: no adaptive step control is differentiated",
    "event_time": "refused: a crossed event raises SolverDerivativeRefusal",
    "delayed_history": "excluded: right-hand sides depend on the current state only",
    "failed_solve": "refused: non-finite or diverging states raise SolverDerivativeRefusal",
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
    latter.
    """

    final_state: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    steps: int
    final_time: float
    method: str = "fixed_step_rk4_discrete_derivative"
    kind: str = "discrete"


@dataclass(frozen=True)
class ContinuousSolverSensitivity:
    """Continuous sensitivity estimated only from mesh-refinement evidence."""

    sensitivity: NDArray[np.float64]
    meshes: tuple[int, ...]
    discrete_sensitivities: tuple[NDArray[np.float64], ...]
    observed_order: float
    error_estimate: float
    tolerance: float
    method: str = "richardson_refined_rk4_sensitivity"
    kind: str = "continuous_refined"


def _state_vector(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return array


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
        Number of equal steps, at least one.
    initial_sensitivity
        ``dy(0)/dtheta``; zero when omitted.
    event
        Optional ``g(t, y, theta)``. A sign change of ``g`` between any two
        accepted states is refused; the event-time derivative is not supported.
    divergence_bound
        States whose largest magnitude exceeds this bound count as a failed solve.

    Returns
    -------
    DiscreteSolverSensitivity
        Final state and ``dy_N/dtheta`` of the discrete map, propagated through
        every Runge-Kutta stage.

    Raises
    ------
    SolverDerivativeRefusal
        ``failed_solve`` for non-finite or diverging values, ``event_crossing``
        when the event changes sign.
    ValueError
        If the inputs or returned shapes are malformed.

    """
    state = _state_vector("initial_state", initial_state)
    parameters = _state_vector("theta", theta)
    if isinstance(final_time, bool) or not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
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
    if event is not None:
        start = float(event(0.0, state, parameters))
        event_side = math.copysign(1.0, start)
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
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sensitivity = sensitivity + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        if not np.all(np.isfinite(state)) or float(np.max(np.abs(state))) > divergence_bound:
            raise SolverDerivativeRefusal(
                "failed_solve",
                f"the solution diverged by t={t + h:.6g}; no derivative is issued",
            )
        if event is not None and float(event(t + h, state, parameters)) * event_side <= 0.0:
            # Reaching or passing the surface, even exactly at a mesh point, is a crossing.
            raise SolverDerivativeRefusal(
                "event_crossing",
                f"the event reached or changed sign between t={t:.6g} and t={t + h:.6g}; "
                "the event-time derivative is not supported",
            )
    return DiscreteSolverSensitivity(
        final_state=state, sensitivity=sensitivity, steps=steps, final_time=final_time
    )


def refine_solver_sensitivity(
    rhs: StateFunction,
    state_jacobian: StateFunction,
    parameter_jacobian: StateFunction,
    initial_state: ArrayLike,
    theta: ArrayLike,
    final_time: float,
    *,
    base_steps: int = 16,
    refinements: int = 3,
    tolerance: float = 1.0e-8,
    order_tolerance: float = 0.5,
    initial_sensitivity: ArrayLike | None = None,
    event: Callable[[float, NDArray[np.float64], NDArray[np.float64]], float] | None = None,
) -> ContinuousSolverSensitivity:
    """Estimate the continuous sensitivity only from mesh-refinement evidence.

    Parameters
    ----------
    rhs, state_jacobian, parameter_jacobian, initial_state, theta, final_time
        As for :func:`rk4_forward_sensitivity`.
    base_steps
        Coarsest mesh; each refinement doubles it.
    refinements
        Number of meshes, at least three so the observed order can be measured.
    tolerance
        Largest admitted Richardson error estimate.
    order_tolerance
        Largest admitted distance of the observed order from four.
    initial_sensitivity, event
        As for :func:`rk4_forward_sensitivity`; every mesh obeys them.

    Returns
    -------
    ContinuousSolverSensitivity
        Richardson-extrapolated sensitivity from the two finest meshes, with the
        observed order and error estimate that justify it.

    Raises
    ------
    SolverDerivativeRefusal
        ``not_refined`` when the meshes do not show fourth-order convergence or
        the error estimate exceeds ``tolerance``; any refusal of a single mesh.
    ValueError
        If the refinement settings are malformed.

    """
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if isinstance(refinements, bool) or not isinstance(refinements, int) or refinements < 3:
        raise ValueError("refinements must be an integer >= 3")
    for name, value in (("tolerance", tolerance), ("order_tolerance", order_tolerance)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    meshes = tuple(base_steps * 2**level for level in range(refinements))
    discrete = tuple(
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
        ).sensitivity
        for steps in meshes
    )
    coarse = float(np.max(np.abs(discrete[-2] - discrete[-3])))
    fine = float(np.max(np.abs(discrete[-1] - discrete[-2])))
    scale = max(1.0, float(np.max(np.abs(discrete[-1]))))
    if fine <= 1.0e-14 * scale:
        observed_order = float(RK4_ORDER)
    elif coarse <= 1.0e-14 * scale:
        observed_order = 0.0
    else:
        observed_order = math.log2(coarse / fine)
    error_estimate = fine / (2**RK4_ORDER - 1)
    refined = discrete[-1] + (discrete[-1] - discrete[-2]) / (2**RK4_ORDER - 1)
    converged = abs(observed_order - RK4_ORDER) <= order_tolerance or fine <= 1.0e-14 * scale
    if not converged or error_estimate > tolerance:
        raise SolverDerivativeRefusal(
            "not_refined",
            f"meshes {meshes} show order {observed_order:.3g} and error estimate "
            f"{error_estimate:.3e} (tolerance {tolerance:.3e}); no continuous claim is issued",
        )
    return ContinuousSolverSensitivity(
        sensitivity=refined,
        meshes=meshes,
        discrete_sensitivities=discrete,
        observed_order=observed_order,
        error_estimate=error_estimate,
        tolerance=tolerance,
    )
