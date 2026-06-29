# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Next-generation neural-mass exact mean field (QIF / theta neurons)
r"""Exact firing-rate mean field of a heterogeneous QIF / theta-neuron network.

A population of all-to-all coupled quadratic integrate-and-fire (QIF) neurons with
Lorentzian-distributed excitabilities admits, in the thermodynamic limit, an *exact* closed
description by two macroscopic variables — the population firing rate ``r`` and the mean
membrane potential ``V``. With the membrane time constant set to one this is

.. math::

    \dot r &= \frac{\Delta}{\pi} + 2\,r\,V, \\
    \dot V &= V^2 + \bar\eta + J\,r + I - \pi^2 r^2,

where ``Δ`` is the half-width of the Lorentzian excitability distribution, ``η̄`` its centre,
``J`` the synaptic coupling and ``I`` an external current. These are the next-generation
neural-mass equations: unlike heuristic firing-rate models they link the microscopic spiking
network to the macroscopic rate *exactly*, derived through the Ott–Antonsen ansatz applied to
the QIF / theta-neuron ensemble. The theta neuron is the QIF neuron on the circle through
``V = tan(θ/2)``, and the macroscopic ``(r, V)`` is the conformal image of the theta-population
Kuramoto order parameter ``Z``: ``W = π r + i V`` with ``Z = (1 − \bar W)/(1 + \bar W)``, so the
firing rate is ``r = π^{-1}\operatorname{Re} W`` and the mean potential ``V = \operatorname{Im} W``.

This bridges the Kuramoto synchronisation toolkit to computational neuroscience: the same
phase-population whose coherence the order parameter measures is, under the conformal map, a
spiking network whose macroscopic firing rate this module evolves. It is an analysis layer over
the two-dimensional reduced flow — a fixed-step RK4 with the analytic Jacobian and a forward-mode
differentiable terminal form — and adds no compute kernel.

Reference: Montbrió, Pazó & Roxin, *Phys. Rev. X* 5, 021028 (2015).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

#: A terminal objective ``L(r_N, V_N)`` on the final macroscopic state.
QifTerminalObjective = Callable[[float, float], float]

#: The gradient of a :data:`QifTerminalObjective`, returning ``(∂L/∂r_N, ∂L/∂V_N)``.
QifTerminalObjectiveGrad = Callable[[float, float], "tuple[float, float]"]


def qif_mean_field_rates(
    firing_rate: float,
    mean_potential: float,
    *,
    eta_bar: float,
    delta: float,
    coupling: float,
    current: float = 0.0,
) -> tuple[float, float]:
    r"""Return the exact firing-rate field ``(ṙ, V̇)`` of the QIF / theta mean field.

    Implements ``ṙ = Δ/π + 2 r V`` and ``V̇ = V² + η̄ + J r + I − π² r²``.

    Parameters
    ----------
    firing_rate : float
        The population firing rate ``r``.
    mean_potential : float
        The mean membrane potential ``V``.
    eta_bar : float
        The centre ``η̄`` of the Lorentzian excitability distribution.
    delta : float
        The half-width ``Δ`` of the Lorentzian distribution (``≥ 0``).
    coupling : float
        The synaptic coupling strength ``J``.
    current : float, optional
        The external current ``I``; defaults to ``0``.

    Returns
    -------
    tuple of float
        The time derivatives ``(ṙ, V̇)``.

    Raises
    ------
    ValueError
        If ``delta`` is negative.
    """
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    rate_derivative = delta / np.pi + 2.0 * firing_rate * mean_potential
    potential_derivative = (
        mean_potential * mean_potential
        + eta_bar
        + coupling * firing_rate
        + current
        - np.pi**2 * firing_rate * firing_rate
    )
    return float(rate_derivative), float(potential_derivative)


def qif_mean_field_jacobian(
    firing_rate: float, mean_potential: float, *, coupling: float
) -> NDArray[np.float64]:
    r"""Return the ``2 × 2`` Jacobian ``∂(ṙ, V̇)/∂(r, V)`` of the QIF mean field.

    The Jacobian is ``[[2V, 2r], [J − 2π² r, 2V]]``; it does not depend on the heterogeneity
    or the external current.

    Parameters
    ----------
    firing_rate : float
        The population firing rate ``r``.
    mean_potential : float
        The mean membrane potential ``V``.
    coupling : float
        The synaptic coupling strength ``J``.

    Returns
    -------
    numpy.ndarray
        The ``(2, 2)`` Jacobian in the ``(r, V)`` ordering.
    """
    return np.array(
        [
            [2.0 * mean_potential, 2.0 * firing_rate],
            [coupling - 2.0 * np.pi**2 * firing_rate, 2.0 * mean_potential],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class QifMeanFieldTrajectory:
    """A trajectory of the QIF / theta mean field sampled at every RK4 step.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    firing_rate : numpy.ndarray
        The population firing rate ``r(t)`` (length ``n_steps + 1``).
    mean_potential : numpy.ndarray
        The mean membrane potential ``V(t)`` (length ``n_steps + 1``).
    """

    times: NDArray[np.float64]
    firing_rate: NDArray[np.float64]
    mean_potential: NDArray[np.float64]

    @property
    def terminal_firing_rate(self) -> float:
        """The final population firing rate."""
        return float(self.firing_rate[-1])

    @property
    def terminal_mean_potential(self) -> float:
        """The final mean membrane potential."""
        return float(self.mean_potential[-1])


def _validate_run(delta: float, dt: float, n_steps: int) -> None:
    """Validate the shared range constraints of an integration / sensitivity run."""
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")


def integrate_qif_mean_field(
    firing_rate: float,
    mean_potential: float,
    *,
    eta_bar: float,
    delta: float,
    coupling: float,
    current: float = 0.0,
    dt: float,
    n_steps: int,
) -> QifMeanFieldTrajectory:
    r"""Integrate the QIF / theta mean field by a fixed-step RK4 over ``(r, V)``.

    Parameters
    ----------
    firing_rate, mean_potential : float
        The initial macroscopic state ``(r(0), V(0))``.
    eta_bar, delta, coupling, current : float
        The Lorentzian centre ``η̄``, half-width ``Δ`` (``≥ 0``), synaptic coupling ``J`` and
        external current ``I``.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).

    Returns
    -------
    QifMeanFieldTrajectory
        The sampled firing-rate and mean-potential trajectory.

    Raises
    ------
    ValueError
        If ``delta``/``dt``/``n_steps`` are out of range.
    """
    _validate_run(delta, dt, n_steps)

    def field(state: NDArray[np.float64]) -> NDArray[np.float64]:
        rate_derivative, potential_derivative = qif_mean_field_rates(
            float(state[0]),
            float(state[1]),
            eta_bar=eta_bar,
            delta=delta,
            coupling=coupling,
            current=current,
        )
        return np.array([rate_derivative, potential_derivative], dtype=np.float64)

    rate_history = np.empty(n_steps + 1, dtype=np.float64)
    potential_history = np.empty(n_steps + 1, dtype=np.float64)
    state = np.array([firing_rate, mean_potential], dtype=np.float64)
    rate_history[0] = state[0]
    potential_history[0] = state[1]
    for step in range(n_steps):
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        rate_history[step + 1] = state[0]
        potential_history[step + 1] = state[1]
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return QifMeanFieldTrajectory(times, rate_history, potential_history)


def qif_mean_field_fixed_point(
    *,
    eta_bar: float,
    delta: float,
    coupling: float,
    current: float = 0.0,
    max_rate: float = 50.0,
) -> tuple[float, float]:
    r"""Return a fixed point ``(r*, V*)`` of the QIF mean field with ``r* > 0``.

    Setting ``ṙ = 0`` gives ``V = −Δ/(2π r)``; substituting into ``V̇ = 0`` leaves a single
    equation in ``r`` which is solved by a bracketed root find on ``(0, max_rate]``. The system
    can be multistable; this returns the smallest positive-rate fixed point in the bracket.

    Parameters
    ----------
    eta_bar, delta, coupling, current : float
        The Lorentzian centre, half-width (``> 0``), synaptic coupling and external current.
    max_rate : float, optional
        The upper bracket for the firing rate; defaults to ``50``.

    Returns
    -------
    tuple of float
        The fixed point ``(r*, V*)``.

    Raises
    ------
    ValueError
        If ``delta``/``max_rate`` are out of range or no fixed point is bracketed.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive for a fixed point, got {delta}")
    if max_rate <= 0.0:
        raise ValueError(f"max_rate must be positive, got {max_rate}")

    def residual(rate: float) -> float:
        potential = -delta / (2.0 * np.pi * rate)
        return float(
            potential * potential + eta_bar + coupling * rate + current - np.pi**2 * rate * rate
        )

    lower = 1e-9
    if residual(lower) * residual(max_rate) > 0.0:
        raise ValueError("no positive-rate fixed point bracketed in (0, max_rate]")
    root = float(brentq(residual, lower, max_rate))
    return root, -delta / (2.0 * np.pi * root)


def qif_potential_from_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the QIF membrane potential ``V = tan(θ/2)`` of a theta-neuron phase."""
    return np.asarray(np.tan(np.asarray(theta, dtype=np.float64) / 2.0), dtype=np.float64)


def theta_from_qif_potential(potential: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the theta-neuron phase ``θ = 2 arctan(V)`` of a QIF membrane potential."""
    return np.asarray(2.0 * np.arctan(np.asarray(potential, dtype=np.float64)), dtype=np.float64)


def kuramoto_order_parameter_from_macro(firing_rate: float, mean_potential: float) -> complex:
    r"""Return the theta-population Kuramoto order parameter ``Z`` of a macroscopic state.

    Uses the conformal map ``W = π r + i V`` and ``Z = (1 − \bar W)/(1 + \bar W)``.
    """
    conformal = np.pi * firing_rate + 1j * mean_potential
    return complex((1.0 - np.conj(conformal)) / (1.0 + np.conj(conformal)))


def macro_from_kuramoto_order_parameter(order_parameter: complex) -> tuple[float, float]:
    r"""Return the macroscopic ``(r, V)`` of a theta-population Kuramoto order parameter ``Z``.

    Inverts :func:`kuramoto_order_parameter_from_macro`: ``W = (1 − \bar Z)/(1 + \bar Z)`` then
    ``r = π^{-1}\operatorname{Re} W`` and ``V = \operatorname{Im} W``.
    """
    conformal = (1.0 - np.conj(order_parameter)) / (1.0 + np.conj(order_parameter))
    return float(conformal.real / np.pi), float(conformal.imag)


@dataclass(frozen=True)
class QifMeanFieldGradients:
    """Gradients of a terminal objective through the QIF mean-field integrator.

    Attributes
    ----------
    initial_firing_rate : float
        ``∂L/∂r_0``.
    initial_mean_potential : float
        ``∂L/∂V_0``.
    eta_bar : float
        ``∂L/∂η̄``.
    delta : float
        ``∂L/∂Δ``.
    coupling : float
        ``∂L/∂J``.
    current : float
        ``∂L/∂I``.
    """

    initial_firing_rate: float
    initial_mean_potential: float
    eta_bar: float
    delta: float
    coupling: float
    current: float


def _qif_stage_derivative(
    state: NDArray[np.float64],
    sensitivity: NDArray[np.float64],
    eta_bar: float,
    delta: float,
    coupling: float,
    current: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(field, J·S + E)`` for one RK4 stage of the QIF mean field.

    The channel layout is ``[r₀, V₀, η̄, Δ, J, I]``; the explicit injection ``E`` carries the
    direct parameter dependence of the field.
    """
    rate_derivative, potential_derivative = qif_mean_field_rates(
        float(state[0]),
        float(state[1]),
        eta_bar=eta_bar,
        delta=delta,
        coupling=coupling,
        current=current,
    )
    field = np.array([rate_derivative, potential_derivative], dtype=np.float64)
    jacobian = qif_mean_field_jacobian(float(state[0]), float(state[1]), coupling=coupling)
    injection = np.zeros((2, 6), dtype=np.float64)
    injection[1, 2] = 1.0  # ∂V̇/∂η̄
    injection[0, 3] = 1.0 / np.pi  # ∂ṙ/∂Δ
    injection[1, 4] = float(state[0])  # ∂V̇/∂J = r
    injection[1, 5] = 1.0  # ∂V̇/∂I
    return field, jacobian @ sensitivity + injection


def qif_mean_field_terminal_value_and_grad(
    firing_rate: float,
    mean_potential: float,
    *,
    eta_bar: float,
    delta: float,
    coupling: float,
    current: float = 0.0,
    dt: float,
    n_steps: int,
    objective: QifTerminalObjective,
    objective_grad: QifTerminalObjectiveGrad,
) -> tuple[float, QifMeanFieldGradients]:
    r"""Differentiate a terminal objective through the QIF mean-field integrator.

    Evaluates ``L(r_N, V_N)`` and returns its gradients with respect to every input
    (``r_0, V_0, η̄, Δ, J, I``) by propagating the forward-mode sensitivity of the two-dimensional
    RK4 and contracting with the terminal cotangent ``(∂L/∂r_N, ∂L/∂V_N)``.

    Parameters
    ----------
    firing_rate, mean_potential : float
        The initial macroscopic state ``(r(0), V(0))``.
    eta_bar, delta, coupling, current : float
        The Lorentzian centre, half-width (``≥ 0``), synaptic coupling and external current.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).
    objective : callable
        The terminal objective ``L(r_N, V_N) → float``.
    objective_grad : callable
        Its gradient ``(r_N, V_N) → (∂L/∂r_N, ∂L/∂V_N)``.

    Returns
    -------
    tuple
        ``(value, QifMeanFieldGradients)``.

    Raises
    ------
    ValueError
        If ``delta``/``dt``/``n_steps`` are out of range.
    """
    _validate_run(delta, dt, n_steps)

    state = np.array([firing_rate, mean_potential], dtype=np.float64)
    sensitivity: NDArray[np.float64] = np.zeros((2, 6), dtype=np.float64)
    sensitivity[0, 0] = 1.0  # ∂r₀/∂r₀
    sensitivity[1, 1] = 1.0  # ∂V₀/∂V₀

    for _ in range(n_steps):
        k1, s1 = _qif_stage_derivative(state, sensitivity, eta_bar, delta, coupling, current)
        k2, s2 = _qif_stage_derivative(
            state + 0.5 * dt * k1,
            sensitivity + 0.5 * dt * s1,
            eta_bar,
            delta,
            coupling,
            current,
        )
        k3, s3 = _qif_stage_derivative(
            state + 0.5 * dt * k2,
            sensitivity + 0.5 * dt * s2,
            eta_bar,
            delta,
            coupling,
            current,
        )
        k4, s4 = _qif_stage_derivative(
            state + dt * k3, sensitivity + dt * s3, eta_bar, delta, coupling, current
        )
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        sensitivity = sensitivity + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    value = float(objective(float(state[0]), float(state[1])))
    grad_rate, grad_potential = objective_grad(float(state[0]), float(state[1]))
    cotangent = np.array([grad_rate, grad_potential], dtype=np.float64)
    flat = cotangent @ sensitivity
    return value, QifMeanFieldGradients(
        initial_firing_rate=float(flat[0]),
        initial_mean_potential=float(flat[1]),
        eta_bar=float(flat[2]),
        delta=float(flat[3]),
        coupling=float(flat[4]),
        current=float(flat[5]),
    )


__all__ = [
    "QifMeanFieldGradients",
    "QifMeanFieldTrajectory",
    "QifTerminalObjective",
    "QifTerminalObjectiveGrad",
    "integrate_qif_mean_field",
    "kuramoto_order_parameter_from_macro",
    "macro_from_kuramoto_order_parameter",
    "qif_mean_field_fixed_point",
    "qif_mean_field_jacobian",
    "qif_mean_field_rates",
    "qif_mean_field_terminal_value_and_grad",
    "qif_potential_from_theta",
    "theta_from_qif_potential",
]
