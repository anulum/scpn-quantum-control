# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ott–Antonsen reduction of the mean-field Kuramoto model
r"""Ott–Antonsen reduction of the mean-field Kuramoto model.

For the globally coupled Kuramoto model with a Lorentzian natural-frequency density
``g(ω) = (Δ/π) / ((ω − ω₀)² + Δ²)``, Ott & Antonsen (2008) showed that the dynamics collapse
exactly onto a two-dimensional manifold: the complex order parameter ``z = r e^{iψ}`` obeys the
closed ordinary differential equation

``ż = (iω₀ − Δ + K/2) z − (K/2) |z|² z``.

Its modulus follows ``ṙ = (K/2 − Δ) r − (K/2) r³`` while the phase drifts at ``ψ̇ = ω₀``, so the
incoherent state ``r = 0`` loses stability at the critical coupling ``K_c = 2Δ`` and the partially
synchronised branch ``r* = √(1 − 2Δ/K)`` appears above it (the same onset and steady state as the
mean-field self-consistency in :mod:`~oscillatools.accel.kuramoto_critical_coupling`).

This is an analysis layer over the synchronisation theory. The reduced flow is integrated with a
fixed-step RK4 in real ``(Re z, Im z)`` coordinates; the differentiable form propagates the
forward sensitivities ``∂z/∂K`` and ``∂z/∂Δ`` through the same RK4 (an augmented six-dimensional
system), so the terminal order parameter is returned with its exact gradient and the module adds
no compute kernel.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .kuramoto_critical_coupling import lorentzian_order_parameter

_Field = Callable[[NDArray[np.float64], float, float, float], NDArray[np.float64]]


def _field_real(
    state: NDArray[np.float64], coupling: float, half_width: float, centre: float
) -> NDArray[np.float64]:
    """Return the real-coordinate Ott–Antonsen vector field ``(ẋ, ẏ)`` at ``state = (x, y)``."""
    x, y = float(state[0]), float(state[1])
    linear = 0.5 * coupling - half_width
    radius_squared = x * x + y * y
    cubic = 0.5 * coupling * radius_squared
    return np.array(
        [
            linear * x - centre * y - cubic * x,
            linear * y + centre * x - cubic * y,
        ],
        dtype=np.float64,
    )


def _augmented_field(
    augmented: NDArray[np.float64], coupling: float, half_width: float, centre: float
) -> NDArray[np.float64]:
    """Return the field of the state plus the ``∂/∂K`` and ``∂/∂Δ`` forward sensitivities.

    ``augmented`` packs ``(x, y, ∂x/∂K, ∂y/∂K, ∂x/∂Δ, ∂y/∂Δ)``. Each sensitivity column evolves by
    the variational equation ``Ṡ = J(state) S + ∂f/∂p`` with the state Jacobian ``J`` and the
    explicit parameter derivatives of the vector field.
    """
    state = augmented[:2]
    x, y = float(state[0]), float(state[1])
    linear = 0.5 * coupling - half_width
    radius_squared = x * x + y * y
    half_k = 0.5 * coupling
    # State Jacobian J of the real field.
    jacobian = np.array(
        [
            [linear - half_k * (3.0 * x * x + y * y), -centre - half_k * 2.0 * x * y],
            [centre - half_k * 2.0 * x * y, linear - half_k * (x * x + 3.0 * y * y)],
        ],
        dtype=np.float64,
    )
    d_field_d_coupling = np.array(
        [0.5 * x * (1.0 - radius_squared), 0.5 * y * (1.0 - radius_squared)], dtype=np.float64
    )
    d_field_d_half_width = np.array([-x, -y], dtype=np.float64)
    field = _field_real(state, coupling, half_width, centre)
    sensitivity_coupling = jacobian @ augmented[2:4] + d_field_d_coupling
    sensitivity_half_width = jacobian @ augmented[4:6] + d_field_d_half_width
    return np.concatenate([field, sensitivity_coupling, sensitivity_half_width])


def _rk4_step(
    state: NDArray[np.float64],
    dt: float,
    coupling: float,
    half_width: float,
    centre: float,
    field: _Field,
) -> NDArray[np.float64]:
    """Advance ``state`` by one RK4 step of the (possibly augmented) ``field``."""
    k1 = field(state, coupling, half_width, centre)
    k2 = field(state + 0.5 * dt * k1, coupling, half_width, centre)
    k3 = field(state + 0.5 * dt * k2, coupling, half_width, centre)
    k4 = field(state + dt * k3, coupling, half_width, centre)
    return np.asarray(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)


def _validate(coupling: float, half_width: float, dt: float, n_steps: int) -> None:
    if coupling <= 0.0:
        raise ValueError(f"coupling must be positive, got {coupling}")
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")


def ott_antonsen_field(
    z: complex, coupling: float, half_width: float, *, centre: float = 0.0
) -> complex:
    r"""Ott–Antonsen vector field ``ż = (iω₀ − Δ + K/2) z − (K/2) |z|² z``.

    Parameters
    ----------
    z : complex
        The complex order parameter ``z = r e^{iψ}``.
    coupling : float
        The global coupling strength ``K``.
    half_width : float
        The Lorentzian half-width ``Δ``.
    centre : float, optional
        The mean natural frequency ``ω₀`` (default ``0``).

    Returns
    -------
    complex
        The time derivative ``ż``.
    """
    linear = 0.5 * coupling - half_width
    return complex((linear + 1j * centre) * z - 0.5 * coupling * abs(z) ** 2 * z)


def ott_antonsen_trajectory(
    z0: complex,
    coupling: float,
    half_width: float,
    dt: float,
    n_steps: int,
    *,
    centre: float = 0.0,
) -> NDArray[np.complex128]:
    r"""Integrate the reduced order-parameter flow with RK4.

    Parameters
    ----------
    z0 : complex
        The initial complex order parameter.
    coupling, half_width : float
        The coupling ``K`` and Lorentzian half-width ``Δ``.
    dt : float
        The integration step.
    n_steps : int
        The number of steps.
    centre : float, optional
        The mean natural frequency ``ω₀``.

    Returns
    -------
    numpy.ndarray
        The ``(n_steps + 1,)`` complex trajectory ``z(t)`` including the initial state.

    Raises
    ------
    ValueError
        If ``coupling`` or ``half_width`` or ``dt`` is non-positive, or ``n_steps < 1``.
    """
    _validate(coupling, half_width, dt, n_steps)
    state = np.array([z0.real, z0.imag], dtype=np.float64)
    trajectory = np.empty((n_steps + 1, 2), dtype=np.float64)
    trajectory[0] = state
    for step in range(n_steps):
        state = _rk4_step(state, dt, coupling, half_width, centre, _field_real)
        trajectory[step + 1] = state
    return np.ascontiguousarray(trajectory[:, 0] + 1j * trajectory[:, 1], dtype=np.complex128)


def ott_antonsen_order_parameter(
    z0: complex,
    coupling: float,
    half_width: float,
    dt: float,
    n_steps: int,
    *,
    centre: float = 0.0,
) -> NDArray[np.float64]:
    r"""Modulus trajectory ``r(t) = |z(t)|`` of the reduced flow.

    The magnitude of :func:`ott_antonsen_trajectory`; parameters are as there.
    """
    return np.abs(
        ott_antonsen_trajectory(z0, coupling, half_width, dt, n_steps, centre=centre)
    ).astype(np.float64)


def ott_antonsen_steady_state(coupling: float, half_width: float) -> float:
    r"""Steady-state order parameter ``r* = √(1 − 2Δ/K)`` of the reduced flow.

    Zero at or below the critical coupling ``K_c = 2Δ`` and the synchronised branch above it — the
    Ott–Antonsen fixed point, identical to the mean-field self-consistency steady state
    (:func:`~oscillatools.accel.kuramoto_critical_coupling.lorentzian_order_parameter`).

    Raises
    ------
    ValueError
        If ``coupling`` or ``half_width`` is non-positive.
    """
    return lorentzian_order_parameter(coupling, half_width)


def ott_antonsen_terminal_order_parameter_value_and_grad(
    z0: complex,
    coupling: float,
    half_width: float,
    dt: float,
    n_steps: int,
    *,
    centre: float = 0.0,
) -> tuple[float, float, float]:
    r"""Terminal order parameter ``r(T) = |z(T)|`` and its gradient ``(∂r/∂K, ∂r/∂Δ)``.

    Integrates the reduced flow together with its forward sensitivities ``∂z/∂K`` and ``∂z/∂Δ``
    (an augmented RK4) and contracts them with ``∂r/∂z`` at the final time. The initial state is
    independent of ``K`` and ``Δ``, so the sensitivities start at zero.

    Parameters
    ----------
    z0 : complex
        The initial complex order parameter.
    coupling, half_width : float
        The coupling ``K`` and Lorentzian half-width ``Δ``.
    dt : float
        The integration step.
    n_steps : int
        The number of steps.
    centre : float, optional
        The mean natural frequency ``ω₀``.

    Returns
    -------
    tuple
        ``(r_T, ∂r_T/∂K, ∂r_T/∂Δ)``.

    Raises
    ------
    ValueError
        If ``coupling`` or ``half_width`` or ``dt`` is non-positive, ``n_steps < 1``, or the
        terminal state is the origin (where ``r`` is not differentiable).
    """
    _validate(coupling, half_width, dt, n_steps)
    augmented = np.array([z0.real, z0.imag, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(n_steps):
        augmented = _rk4_step(augmented, dt, coupling, half_width, centre, _augmented_field)
    x, y = float(augmented[0]), float(augmented[1])
    radius = math.hypot(x, y)
    if radius == 0.0:
        raise ValueError("terminal order parameter is zero; r is not differentiable at the origin")
    d_radius_d_state = np.array([x / radius, y / radius], dtype=np.float64)
    grad_coupling = float(d_radius_d_state @ augmented[2:4])
    grad_half_width = float(d_radius_d_state @ augmented[4:6])
    return radius, grad_coupling, grad_half_width


__all__ = [
    "ott_antonsen_field",
    "ott_antonsen_order_parameter",
    "ott_antonsen_steady_state",
    "ott_antonsen_terminal_order_parameter_value_and_grad",
    "ott_antonsen_trajectory",
]
