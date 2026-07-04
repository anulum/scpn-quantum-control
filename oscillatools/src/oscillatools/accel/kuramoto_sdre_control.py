# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — State-Dependent Riccati Equation feedback control of Kuramoto networks
r"""State-Dependent Riccati Equation (SDRE) feedback control of a Kuramoto network.

Where the adjoint optimal control (:mod:`oscillatools.accel.kuramoto_network_control`) is an
*open-loop* gradient optimiser over a control time series, SDRE control is a *closed-loop state
feedback* — it computes the input as a function of the current phases. SDRE is the nonlinear
generalisation of the LQR: the drift is written in state-dependent-coefficient form
``ẋ = A(x)\,x + B u`` and the feedback is ``u = -R^{-1} B^{\mathsf T} P(x)\,x``, where ``P(x)`` solves
the *algebraic* Riccati equation ``A(x)^{\mathsf T}P + P A(x) - P B R^{-1}B^{\mathsf T}P + Q = 0``
afresh at each state (Tahirovic, 2025).

For the Kuramoto network the deviation ``x = θ - θ^\star`` from a target configuration has an *exact*
state-dependent factorisation of its drift: using
``\sin(\Delta + \delta) - \sin\Delta = \cos\Delta\,\operatorname{sinc}\delta\,\delta
+ \sin\Delta\,\tfrac{\cos\delta - 1}{\delta}\,\delta`` the deviation drift becomes a weighted,
state-dependent graph Laplacian ``A(x)`` whose value at ``x = 0`` is exactly the network Jacobian at
the target (so the SDRE reduces to the LQR there). With ``B = I`` every oscillator is actuated, so the
Riccati equation always has a stabilising solution. A feed-forward term ``-f(θ^\star)`` makes the
target an equilibrium, and the SDRE feedback then drives the network onto it. It adds no compute
kernel beyond the dense Riccati solve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from .kuramoto_network_control import ControlledNetworkTrajectory
from .networked_kuramoto import networked_kuramoto_force


def _sinc(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sin(z)/z`` (unity at zero)."""
    return np.asarray(np.sinc(values / np.pi), dtype=np.float64)


def _cosine_ratio(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(cos(z) - 1)/z`` (zero at zero)."""
    safe = np.where(values == 0.0, 1.0, values)
    return np.asarray(
        np.where(np.abs(values) < 1e-12, 0.0, (np.cos(values) - 1.0) / safe), dtype=np.float64
    )


def _state_dependent_coefficient(
    phases: NDArray[np.float64], target: NDArray[np.float64], coupling: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The exact SDC matrix ``A(x)`` of the deviation drift (a state-dependent graph Laplacian)."""
    target_difference = target[None, :] - target[:, None]
    deviation_difference = (phases[None, :] - phases[:, None]) - target_difference
    weights = coupling * (
        np.cos(target_difference) * _sinc(deviation_difference)
        + np.sin(target_difference) * _cosine_ratio(deviation_difference)
    )
    matrix = weights.copy()
    np.fill_diagonal(matrix, -weights.sum(axis=1))
    return np.asarray(matrix, dtype=np.float64)


def _validate(
    phases: NDArray[np.float64],
    target: NDArray[np.float64],
    coupling: NDArray[np.float64],
    state_cost: float,
    control_cost: float,
) -> int:
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phases must be a one-dimensional array of length at least two")
    count = phases.size
    if target.shape != (count,):
        raise ValueError(f"target_phases must have shape ({count},), got {target.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not (
        np.all(np.isfinite(phases))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(coupling))
    ):
        raise ValueError("phases, target_phases and coupling must be finite")
    if state_cost <= 0.0:
        raise ValueError(f"state_cost must be positive, got {state_cost}")
    if control_cost <= 0.0:
        raise ValueError(f"control_cost must be positive, got {control_cost}")
    return count


def kuramoto_sdre_gain(
    phases: NDArray[np.float64],
    target_phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    state_cost: float,
    control_cost: float,
) -> NDArray[np.float64]:
    r"""The SDRE feedback gain ``G(θ) = R^{-1} P(θ)`` at the current phases.

    Solves the algebraic Riccati equation for the state-dependent coefficient ``A(θ)`` of the
    deviation drift (with ``B = I``, ``Q = q\,I``, ``R = r\,I``) and returns the feedback gain.

    Parameters
    ----------
    phases : numpy.ndarray
        The current phases ``θ`` (length ``N ≥ 2``).
    target_phases : numpy.ndarray
        The target configuration ``θ^⋆`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    state_cost : float
        The state-cost weight ``q`` (``> 0``); ``Q = q I``.
    control_cost : float
        The control-cost weight ``r`` (``> 0``); ``R = r I``.

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` SDRE feedback gain.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    target = np.ascontiguousarray(target_phases, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(angle, target, matrix, state_cost, control_cost)
    coefficient = _state_dependent_coefficient(angle, target, matrix)
    actuation = np.eye(count, dtype=np.float64)
    riccati = solve_continuous_are(
        coefficient, actuation, state_cost * actuation, control_cost * actuation
    )
    return np.asarray(riccati / control_cost, dtype=np.float64)


def sdre_control_input(
    phases: NDArray[np.float64],
    target_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    state_cost: float,
    control_cost: float,
) -> NDArray[np.float64]:
    r"""The full SDRE control ``u = -f(θ^⋆) - G(θ)\,(θ - θ^⋆)`` (feed-forward + feedback).

    Parameters
    ----------
    phases, target_phases, coupling, state_cost, control_cost
        As for :func:`kuramoto_sdre_gain`.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``); used for the feed-forward term.

    Returns
    -------
    numpy.ndarray
        The control input ``u`` (length ``N``).

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    target = np.ascontiguousarray(target_phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(angle, target, matrix, state_cost, control_cost)
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("omega must be finite")
    gain = kuramoto_sdre_gain(
        angle, target, matrix, state_cost=state_cost, control_cost=control_cost
    )
    feed_forward = frequencies + networked_kuramoto_force(target, matrix)
    return np.asarray(-feed_forward - gain @ (angle - target), dtype=np.float64)


def integrate_sdre_controlled_kuramoto(
    phases: NDArray[np.float64],
    target_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    state_cost: float,
    control_cost: float,
) -> ControlledNetworkTrajectory:
    r"""Integrate the SDRE-controlled Kuramoto network with RK4 and per-step feedback.

    Each step recomputes the SDRE control from the current phases (a sample-and-hold feedback) and
    advances ``θ̇ = ω + F(θ) + u`` by RK4 with that input held over the step, driving the network onto
    the target configuration.

    Parameters
    ----------
    phases, target_phases, omega, coupling, state_cost, control_cost
        As for :func:`sdre_control_input`.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    ControlledNetworkTrajectory
        The closed-loop ``(n_steps + 1, N)`` phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    target = np.ascontiguousarray(target_phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(angle, target, matrix, state_cost, control_cost)
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("omega must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    def field(state: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            frequencies + networked_kuramoto_force(state, matrix) + control, dtype=np.float64
        )

    trajectory = np.empty((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = angle
    current = angle
    for step in range(n_steps):
        control = sdre_control_input(
            current,
            target,
            frequencies,
            matrix,
            state_cost=state_cost,
            control_cost=control_cost,
        )
        k1 = field(current, control)
        k2 = field(current + 0.5 * dt * k1, control)
        k3 = field(current + 0.5 * dt * k2, control)
        k4 = field(current + dt * k3, control)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return ControlledNetworkTrajectory(times=times, phases=trajectory)


__all__ = [
    "integrate_sdre_controlled_kuramoto",
    "kuramoto_sdre_gain",
    "sdre_control_input",
]
