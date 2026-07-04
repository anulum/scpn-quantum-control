# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Watanabe–Strogatz exact finite-N reduction
r"""Watanabe–Strogatz exact finite-N reduction of identical sinusoidally-coupled oscillators.

A population of ``N`` *identical* oscillators driven by a common sinusoidal mean field,
``θ̇_j = ω + \operatorname{Im}(H\,e^{-iθ_j})`` with ``H = K Z`` the Kuramoto mean field
(``Z = N^{-1}\sum_k e^{iθ_k}``), is exactly integrable: every phase evolves by the *same*
time-dependent Möbius transformation of the unit disk. This is the Watanabe–Strogatz theorem,
the finite-``N`` companion to the thermodynamic-limit Ott–Antonsen reduction
(:mod:`oscillatools.accel.kuramoto_ott_antonsen`): instead of ``N`` phases the dynamics
collapses to three global variables — the Möbius map — acting on ``N`` constants of motion.

Representation
--------------
Writing ``w_j = e^{iθ_j}``, the per-oscillator dynamics is the common Riccati flow
``ẇ_j = iω w_j + \tfrac12(H − \bar H w_j^2)``, whose solution is the Möbius map ``w_j(t) =
M_t(w_j(0))``. The map is carried by a special-unitary ``SU(1,1)`` matrix
``M = \begin{psmallmatrix} α & β \\ \bar β & \bar α \end{psmallmatrix}`` with the conserved
invariant ``|α|^2 − |β|^2 = 1`` (the three real degrees of freedom of ``(α, β)`` under that
constraint are the three global Watanabe–Strogatz variables), evolving as

.. math::

    \dot α = \tfrac{i ω}{2} α + \tfrac{H}{2}\bar β, \qquad
    \dot β = \tfrac{i ω}{2} β + \tfrac{H}{2}\bar α,

and the phases are reconstructed by ``w_j(t) = (α b_j + β)/(\bar β\, b_j + \bar α)`` from the
fixed constants of motion ``b_j = e^{iθ_j(0)}``. In the thermodynamic limit with the constants
uniformly spread on the circle the order parameter ``Z`` obeys the identical-oscillator
Ott–Antonsen equation ``ż = iω z + \tfrac{K}{2}(z − \bar z z^2)``.

This is an analysis layer over the synchronisation dynamics — a fixed-step RK4 on the two
complex Möbius parameters plus the algebraic reconstruction — and adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WatanabeStrogatzTrajectory:
    """A Watanabe–Strogatz reduced trajectory of an identical-oscillator population.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    alpha : numpy.ndarray
        The Möbius parameter ``α(t)`` (complex, length ``n_steps + 1``).
    beta : numpy.ndarray
        The Möbius parameter ``β(t)`` (complex, length ``n_steps + 1``).
    order_parameter : numpy.ndarray
        The Kuramoto order parameter ``Z(t)`` (complex, length ``n_steps + 1``).
    constants : numpy.ndarray
        The ``N`` fixed constants of motion ``b_j = e^{iθ_j(0)}`` (complex).
    phases : numpy.ndarray
        The reconstructed ``(n_steps + 1, N)`` phase trajectory ``θ(t)``.
    """

    times: NDArray[np.float64]
    alpha: NDArray[np.complex128]
    beta: NDArray[np.complex128]
    order_parameter: NDArray[np.complex128]
    constants: NDArray[np.complex128]
    phases: NDArray[np.float64]

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final reconstructed phase vector."""
        return np.asarray(self.phases[-1], dtype=np.float64)

    @property
    def terminal_order_parameter(self) -> complex:
        """The final Kuramoto order parameter."""
        return complex(self.order_parameter[-1])


def watanabe_strogatz_constants(initial_phases: NDArray[np.float64]) -> NDArray[np.complex128]:
    r"""Return the Watanabe–Strogatz constants of motion ``b_j = e^{iθ_j(0)}``.

    These are fixed along the flow; the global Möbius map carries all the time dependence.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (one-dimensional, length ``N``).

    Returns
    -------
    numpy.ndarray
        The complex constants on the unit circle.

    Raises
    ------
    ValueError
        If ``initial_phases`` is not a non-empty one-dimensional array.
    """
    theta = np.ascontiguousarray(initial_phases, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 1:
        raise ValueError("initial_phases must be a non-empty one-dimensional array")
    return np.asarray(np.exp(1j * theta), dtype=np.complex128)


def _mobius_image(
    alpha: complex, beta: complex, constants: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Return the Möbius image ``(α b + β)/(β̄ b + ᾱ)`` of the constants."""
    return np.asarray(
        (alpha * constants + beta) / (np.conj(beta) * constants + np.conj(alpha)),
        dtype=np.complex128,
    )


def watanabe_strogatz_phases(
    alpha: complex, beta: complex, constants: NDArray[np.complex128]
) -> NDArray[np.float64]:
    r"""Reconstruct the phases ``θ_j`` from the Möbius parameters and the constants of motion."""
    return np.asarray(np.angle(_mobius_image(alpha, beta, constants)), dtype=np.float64)


def watanabe_strogatz_order_parameter(
    alpha: complex, beta: complex, constants: NDArray[np.complex128]
) -> complex:
    r"""Return the Kuramoto order parameter ``Z = N^{-1}\sum_j w_j`` of a Möbius state."""
    return complex(np.mean(_mobius_image(alpha, beta, constants)))


def watanabe_strogatz_invariant(alpha: complex, beta: complex) -> float:
    r"""Return the conserved ``SU(1,1)`` invariant ``|α|^2 − |β|^2`` (unity along the flow)."""
    return float(abs(alpha) ** 2 - abs(beta) ** 2)


def integrate_watanabe_strogatz(
    initial_phases: NDArray[np.float64],
    *,
    omega: float,
    coupling: float,
    dt: float,
    n_steps: int,
) -> WatanabeStrogatzTrajectory:
    r"""Integrate the identical-oscillator mean-field Kuramoto flow by the WS reduction.

    Advances the two complex Möbius parameters ``(α, β)`` (starting from the identity map) by a
    fixed-step RK4, with the common mean field ``H = K Z`` recomputed from the reconstructed order
    parameter each stage, and reconstructs the phases from the fixed constants of motion. The
    result is exact for identical oscillators: the reconstructed phases match a direct
    ``N``-oscillator integration to integrator precision.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (one-dimensional, length ``N``).
    omega : float
        The common natural frequency ``ω`` of the identical oscillators.
    coupling : float
        The mean-field coupling strength ``K``.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).

    Returns
    -------
    WatanabeStrogatzTrajectory
        The Möbius-parameter, order-parameter and reconstructed-phase trajectory.

    Raises
    ------
    ValueError
        If ``initial_phases`` is malformed or ``dt``/``n_steps`` are out of range.
    """
    constants = watanabe_strogatz_constants(initial_phases)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    count = constants.size

    def derivative(alpha: complex, beta: complex) -> tuple[complex, complex]:
        order_parameter = np.mean(_mobius_image(alpha, beta, constants))
        mean_field = coupling * order_parameter
        d_alpha = 0.5j * omega * alpha + 0.5 * mean_field * np.conj(beta)
        d_beta = 0.5j * omega * beta + 0.5 * mean_field * np.conj(alpha)
        return complex(d_alpha), complex(d_beta)

    alpha_history = np.empty(n_steps + 1, dtype=np.complex128)
    beta_history = np.empty(n_steps + 1, dtype=np.complex128)
    order_history = np.empty(n_steps + 1, dtype=np.complex128)
    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)

    alpha: complex = 1.0 + 0.0j
    beta: complex = 0.0 + 0.0j
    alpha_history[0] = alpha
    beta_history[0] = beta
    order_history[0] = np.mean(constants)
    phase_history[0] = np.angle(constants)

    for step in range(n_steps):
        a1, b1 = derivative(alpha, beta)
        a2, b2 = derivative(alpha + 0.5 * dt * a1, beta + 0.5 * dt * b1)
        a3, b3 = derivative(alpha + 0.5 * dt * a2, beta + 0.5 * dt * b2)
        a4, b4 = derivative(alpha + dt * a3, beta + dt * b3)
        alpha = alpha + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        beta = beta + (dt / 6.0) * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
        alpha_history[step + 1] = alpha
        beta_history[step + 1] = beta
        image = _mobius_image(alpha, beta, constants)
        order_history[step + 1] = np.mean(image)
        phase_history[step + 1] = np.angle(image)

    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return WatanabeStrogatzTrajectory(
        times=times,
        alpha=alpha_history,
        beta=beta_history,
        order_parameter=order_history,
        constants=constants,
        phases=phase_history,
    )


__all__ = [
    "WatanabeStrogatzTrajectory",
    "integrate_watanabe_strogatz",
    "watanabe_strogatz_constants",
    "watanabe_strogatz_invariant",
    "watanabe_strogatz_order_parameter",
    "watanabe_strogatz_phases",
]
