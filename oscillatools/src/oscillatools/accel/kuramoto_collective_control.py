# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Reduced-order optimal control of the collective phase
r"""Reduced-order optimal control of the macroscopic collective phase via the Ott–Antonsen reduction.

Steering a synchronised Kuramoto population by acting on every oscillator is high-dimensional, but
the Ott–Antonsen ansatz collapses a Lorentzian population to a *single* complex order parameter
``z = r e^{iψ}`` obeying ``ż = (iω₀ − Δ + K/2) z − (K/2)|z|² z``. Optimal control on this two-real-
dimensional macroscopic model — rather than on the full network — is the 2025 state of the art for
steering collective synchronisation (Chaos 35, 063125, 2025): one designs a control that drives the
*collective* state, then applies it to the ensemble.

The control here is an external harmonic forcing ``F(t) ∈ ℂ`` applied in the rotating frame, which
enters the reduced flow through the forced Ott–Antonsen term ``+½(F − F̄ z²)`` (Childs & Strogatz,
Chaos 18, 043128, 2008):

.. math::

    \dot z = (i ω_0 - Δ + K/2)\,z - (K/2)\,|z|^2 z + \tfrac12\bigl(F - \bar F z^2\bigr).

Because the flow is only two-real-dimensional, the gradient of a terminal control objective with
respect to the *whole forcing time series* is obtained exactly by the discrete adjoint (reverse-mode)
of the RK4 map — one backward sweep over the stored per-step Jacobians — giving machine-precision
agreement with finite differences. This is the reduced-order, control-grade companion to the
full-network adjoint control: it reuses the toolkit's shipped Ott–Antonsen reduction as the exact
control substrate and is cheap enough to optimise long horizons. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ForcedCollectiveTrajectory:
    """A forced Ott–Antonsen collective-phase trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    order_parameter : numpy.ndarray
        The ``(n_steps + 1,)`` complex order parameter ``z(t) = r e^{iψ}``.
    """

    times: NDArray[np.float64]
    order_parameter: NDArray[np.complex128]

    @property
    def terminal_order_parameter(self) -> complex:
        """The final order parameter ``z(T)``."""
        return complex(self.order_parameter[-1])


@dataclass(frozen=True)
class CollectiveControlGradients:
    """The control objective value and its gradients.

    Attributes
    ----------
    cost : float
        The objective ``|z(T) − z_target|² + w Σ_k |F_k|² dt``.
    forcing_gradient : numpy.ndarray
        The ``(n_steps,)`` gradient with respect to the complex forcing series; the real and
        imaginary parts are the derivatives with respect to the real and imaginary forcing
        components, so a steepest-descent step is ``F ← F − η · forcing_gradient``.
    initial_state_gradient : complex
        The gradient with respect to the initial order parameter ``z(0)``.
    """

    cost: float
    forcing_gradient: NDArray[np.complex128]
    initial_state_gradient: complex


def _field(
    state: NDArray[np.float64],
    forcing: NDArray[np.float64],
    coupling: float,
    half_width: float,
    centre: float,
) -> NDArray[np.float64]:
    """The forced Ott–Antonsen vector field in real coordinates ``(x, y)``."""
    x, y = float(state[0]), float(state[1])
    real_forcing, imag_forcing = float(forcing[0]), float(forcing[1])
    linear = 0.5 * coupling - half_width
    half_k = 0.5 * coupling
    radius_squared = x * x + y * y
    autonomous = np.array(
        [
            linear * x - centre * y - half_k * radius_squared * x,
            linear * y + centre * x - half_k * radius_squared * y,
        ],
        dtype=np.float64,
    )
    control = np.array(
        [
            0.5 * (real_forcing * (1.0 - x * x + y * y) - 2.0 * imag_forcing * x * y),
            0.5 * (imag_forcing * (1.0 + x * x - y * y) - 2.0 * real_forcing * x * y),
        ],
        dtype=np.float64,
    )
    return autonomous + control


def _state_jacobian(
    state: NDArray[np.float64],
    forcing: NDArray[np.float64],
    coupling: float,
    half_width: float,
    centre: float,
) -> NDArray[np.float64]:
    """The ``∂field/∂state`` Jacobian (2×2)."""
    x, y = float(state[0]), float(state[1])
    real_forcing, imag_forcing = float(forcing[0]), float(forcing[1])
    linear = 0.5 * coupling - half_width
    half_k = 0.5 * coupling
    autonomous = np.array(
        [
            [linear - half_k * (3.0 * x * x + y * y), -centre - half_k * 2.0 * x * y],
            [centre - half_k * 2.0 * x * y, linear - half_k * (x * x + 3.0 * y * y)],
        ],
        dtype=np.float64,
    )
    control = np.array(
        [
            [-real_forcing * x - imag_forcing * y, real_forcing * y - imag_forcing * x],
            [imag_forcing * x - real_forcing * y, -imag_forcing * y - real_forcing * x],
        ],
        dtype=np.float64,
    )
    return autonomous + control


def _forcing_jacobian(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """The ``∂field/∂forcing`` Jacobian (2×2)."""
    x, y = float(state[0]), float(state[1])
    return np.array(
        [[0.5 * (1.0 - x * x + y * y), -x * y], [-x * y, 0.5 * (1.0 + x * x - y * y)]],
        dtype=np.float64,
    )


def _step_with_jacobians(
    state: NDArray[np.float64],
    forcing: NDArray[np.float64],
    dt: float,
    coupling: float,
    half_width: float,
    centre: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance one RK4 step and return ``(next_state, ∂next/∂state, ∂next/∂forcing)``."""
    identity = np.eye(2, dtype=np.float64)
    stage1 = _field(state, forcing, coupling, half_width, centre)
    d1_state = _state_jacobian(state, forcing, coupling, half_width, centre)
    d1_forcing = _forcing_jacobian(state)

    point2 = state + 0.5 * dt * stage1
    stage2 = _field(point2, forcing, coupling, half_width, centre)
    jac2 = _state_jacobian(point2, forcing, coupling, half_width, centre)
    d2_state = jac2 @ (identity + 0.5 * dt * d1_state)
    d2_forcing = jac2 @ (0.5 * dt * d1_forcing) + _forcing_jacobian(point2)

    point3 = state + 0.5 * dt * stage2
    stage3 = _field(point3, forcing, coupling, half_width, centre)
    jac3 = _state_jacobian(point3, forcing, coupling, half_width, centre)
    d3_state = jac3 @ (identity + 0.5 * dt * d2_state)
    d3_forcing = jac3 @ (0.5 * dt * d2_forcing) + _forcing_jacobian(point3)

    point4 = state + dt * stage3
    stage4 = _field(point4, forcing, coupling, half_width, centre)
    jac4 = _state_jacobian(point4, forcing, coupling, half_width, centre)
    d4_state = jac4 @ (identity + dt * d3_state)
    d4_forcing = jac4 @ (dt * d3_forcing) + _forcing_jacobian(point4)

    next_state = state + (dt / 6.0) * (stage1 + 2.0 * stage2 + 2.0 * stage3 + stage4)
    d_state = identity + (dt / 6.0) * (d1_state + 2.0 * d2_state + 2.0 * d3_state + d4_state)
    d_forcing = (dt / 6.0) * (d1_forcing + 2.0 * d2_forcing + 2.0 * d3_forcing + d4_forcing)
    return next_state, d_state, d_forcing


def _validate(
    z0: complex,
    forcing: NDArray[np.complex128],
    coupling: float,
    half_width: float,
    dt: float,
) -> None:
    if abs(z0) > 1.0 + 1e-9:
        raise ValueError(f"|z0| must not exceed 1 (it is an order parameter), got {abs(z0)}")
    if forcing.ndim != 1 or forcing.size < 1:
        raise ValueError("forcing must be a non-empty one-dimensional array")
    if coupling <= 0.0:
        raise ValueError(f"coupling must be positive, got {coupling}")
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")


def integrate_forced_collective(
    z0: complex,
    forcing: NDArray[np.complex128],
    coupling: float,
    half_width: float,
    dt: float,
    *,
    centre: float = 0.0,
) -> ForcedCollectiveTrajectory:
    r"""Integrate the forced Ott–Antonsen collective-phase flow with a per-step control series.

    Parameters
    ----------
    z0 : complex
        The initial order parameter ``z(0)`` (``|z0| ≤ 1``).
    forcing : numpy.ndarray
        The ``(n_steps,)`` complex external forcing ``F_k`` applied on step ``k``.
    coupling, half_width : float
        The coupling ``K`` and Lorentzian half-width ``Δ`` (both ``> 0``).
    dt : float
        The RK4 step (``> 0``).
    centre : float, optional
        The mean natural frequency ``ω₀``; defaults to ``0``.

    Returns
    -------
    ForcedCollectiveTrajectory
        The ``(n_steps + 1,)`` complex order-parameter trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    series = np.ascontiguousarray(forcing, dtype=np.complex128)
    _validate(z0, series, coupling, half_width, dt)
    n_steps = series.size
    state = np.array([z0.real, z0.imag], dtype=np.float64)
    path = np.empty((n_steps + 1, 2), dtype=np.float64)
    path[0] = state
    for index in range(n_steps):
        control = np.array([series[index].real, series[index].imag], dtype=np.float64)
        state = _step_with_jacobians(state, control, dt, coupling, half_width, centre)[0]
        path[index + 1] = state
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    order_parameter = np.ascontiguousarray(path[:, 0] + 1j * path[:, 1], dtype=np.complex128)
    return ForcedCollectiveTrajectory(times=times, order_parameter=order_parameter)


def collective_control_value_and_grad(
    z0: complex,
    forcing: NDArray[np.complex128],
    coupling: float,
    half_width: float,
    dt: float,
    *,
    target: complex,
    control_weight: float,
    centre: float = 0.0,
) -> CollectiveControlGradients:
    r"""Evaluate the terminal control objective and its exact gradients by the discrete adjoint.

    The objective is ``J = |z(T) − z_target|² + w Σ_k |F_k|² dt`` (terminal tracking plus control
    energy). The gradient with respect to the whole forcing series is computed by the reverse-mode
    discrete adjoint of the RK4 map, matching finite differences to machine precision.

    Parameters
    ----------
    z0 : complex
        The initial order parameter.
    forcing : numpy.ndarray
        The ``(n_steps,)`` complex forcing series.
    coupling, half_width : float
        The coupling ``K`` and Lorentzian half-width ``Δ``.
    dt : float
        The RK4 step.
    target : complex
        The desired terminal order parameter ``z_target`` (e.g. ``0`` to desynchronise).
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    centre : float, optional
        The mean natural frequency ``ω₀``; defaults to ``0``.

    Returns
    -------
    CollectiveControlGradients
        The cost and the forcing / initial-state gradients.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    series = np.ascontiguousarray(forcing, dtype=np.complex128)
    _validate(z0, series, coupling, half_width, dt)
    if control_weight < 0.0:
        raise ValueError(f"control_weight must be non-negative, got {control_weight}")
    n_steps = series.size

    state = np.array([z0.real, z0.imag], dtype=np.float64)
    controls = np.stack([series.real, series.imag], axis=1)
    state_jacobians = np.empty((n_steps, 2, 2), dtype=np.float64)
    forcing_jacobians = np.empty((n_steps, 2, 2), dtype=np.float64)
    for index in range(n_steps):
        state, d_state, d_forcing = _step_with_jacobians(
            state, controls[index], dt, coupling, half_width, centre
        )
        state_jacobians[index] = d_state
        forcing_jacobians[index] = d_forcing

    target_vector = np.array([target.real, target.imag], dtype=np.float64)
    residual = state - target_vector
    cost = float(residual @ residual + control_weight * float(np.sum(np.abs(series) ** 2)) * dt)

    adjoint = 2.0 * residual
    forcing_gradient = np.empty((n_steps, 2), dtype=np.float64)
    for index in range(n_steps - 1, -1, -1):
        forcing_gradient[index] = (
            forcing_jacobians[index].T @ adjoint + 2.0 * control_weight * controls[index] * dt
        )
        adjoint = state_jacobians[index].T @ adjoint

    gradient = np.ascontiguousarray(
        forcing_gradient[:, 0] + 1j * forcing_gradient[:, 1], dtype=np.complex128
    )
    return CollectiveControlGradients(
        cost=cost,
        forcing_gradient=gradient,
        initial_state_gradient=complex(adjoint[0], adjoint[1]),
    )


def optimise_collective_forcing(
    z0: complex,
    coupling: float,
    half_width: float,
    dt: float,
    n_steps: int,
    *,
    target: complex,
    control_weight: float,
    learning_rate: float,
    n_iterations: int,
    centre: float = 0.0,
    initial_forcing: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    r"""Steer the collective phase to ``target`` by gradient descent on the forcing series.

    Runs ``n_iterations`` steepest-descent updates ``F ← F − η ∇_F J`` using
    :func:`collective_control_value_and_grad`, returning the optimised forcing and the cost history.

    Parameters
    ----------
    z0 : complex
        The initial order parameter.
    coupling, half_width : float
        The coupling ``K`` and Lorentzian half-width ``Δ``.
    dt : float
        The RK4 step.
    n_steps : int
        The control-horizon length (``≥ 1``).
    target : complex
        The desired terminal order parameter.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    learning_rate : float
        The gradient-descent step size ``η`` (``> 0``).
    n_iterations : int
        The number of descent iterations (``≥ 1``).
    centre : float, optional
        The mean natural frequency ``ω₀``; defaults to ``0``.
    initial_forcing : numpy.ndarray, optional
        The starting ``(n_steps,)`` forcing; defaults to zeros (no control).

    Returns
    -------
    tuple of numpy.ndarray
        The optimised ``(n_steps,)`` forcing series and the ``(n_iterations + 1,)`` cost history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    forcing: NDArray[np.complex128]
    if initial_forcing is None:
        forcing = np.zeros(n_steps, dtype=np.complex128)
    else:
        forcing = np.ascontiguousarray(initial_forcing, dtype=np.complex128)
        if forcing.shape != (n_steps,):
            raise ValueError(f"initial_forcing must have shape ({n_steps},), got {forcing.shape}")

    history = np.empty(n_iterations + 1, dtype=np.float64)
    for iteration in range(n_iterations):
        gradients = collective_control_value_and_grad(
            z0,
            forcing,
            coupling,
            half_width,
            dt,
            target=target,
            control_weight=control_weight,
            centre=centre,
        )
        history[iteration] = gradients.cost
        forcing = forcing - learning_rate * gradients.forcing_gradient
    final = collective_control_value_and_grad(
        z0,
        forcing,
        coupling,
        half_width,
        dt,
        target=target,
        control_weight=control_weight,
        centre=centre,
    )
    history[n_iterations] = final.cost
    return np.ascontiguousarray(forcing, dtype=np.complex128), history


__all__ = [
    "CollectiveControlGradients",
    "ForcedCollectiveTrajectory",
    "collective_control_value_and_grad",
    "integrate_forced_collective",
    "optimise_collective_forcing",
]
