# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Higher-order Watanabe–Strogatz reduction
r"""Higher-order Watanabe–Strogatz reduction for harmonically coupled identical oscillators.

The classical Watanabe–Strogatz reduction collapses ``N`` identical phase oscillators driven by a
*first-harmonic* common field ``H = K Z`` to three collective Möbius variables plus ``N`` constants of
motion. The higher-order extension (Jain, 2025) does the same for a **``p``-th harmonic** coupling,

.. math::

    \dot\theta_j = \omega + K\,\operatorname{Im}\bigl(Z_p\,e^{-ip\theta_j}\bigr), \qquad
    Z_p = \frac1N\sum_k e^{ip\theta_k},

which is the mean field of the higher-harmonic (Daido / simplex-type) interaction
``\sin(p(\theta_k-\theta_j))``. Under the change of variable ``\varphi = p\theta`` the dynamics is
exactly the classical Watanabe–Strogatz flow for ``\varphi`` with frequency ``p\omega`` and coupling
``pK``, so the same ``SU(1,1)`` Möbius map carries all the time dependence: the constants
``b_j = e^{ip\theta_j(0)}`` are fixed and the ``p``-fold phases ``\varphi_j = p\theta_j`` are the
Möbius images ``(\alpha b_j + \beta)/(\bar\beta b_j + \bar\alpha)``. The reduction is exact — the
reconstructed ``e^{ip\theta_j}`` match a direct ``N``-oscillator integration to integrator precision —
and recovers the classical reduction at ``p = 1``. It adds no compute kernel.

The ``SU(1,1)`` invariant ``|\alpha|^2 - |\beta|^2`` equals one along the exact flow. As the ensemble
fully synchronises the Möbius parameters diverge to the boundary of the Poincaré disk (a coordinate
singularity, not a dynamical one), so the numerically computed invariant loses precision through
cancellation there even though the *reconstructed* phases and order parameter — read off the Möbius
*ratio* — stay exact; the invariant is conserved to machine precision whenever the parameters remain
bounded (any non-fully-synchronising regime).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_watanabe_strogatz import _mobius_image


@dataclass(frozen=True)
class HigherOrderWatanabeStrogatzTrajectory:
    """A higher-order Watanabe–Strogatz trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times.
    alpha, beta : numpy.ndarray
        The ``(n_steps + 1,)`` complex Möbius parameters ``(α, β)`` of the ``φ = pθ`` flow.
    order_parameter : numpy.ndarray
        The ``(n_steps + 1,)`` ``p``-th harmonic order parameter ``Z_p = ⟨e^{ipθ}⟩``.
    constants : numpy.ndarray
        The ``(N,)`` fixed constants of motion ``b_j = e^{ipθ_j(0)}``.
    harmonic_phases : numpy.ndarray
        The ``(n_steps + 1, N)`` reconstructed ``p``-fold phases ``φ_j = pθ_j`` (modulo ``2π``).
    harmonic : int
        The coupling harmonic ``p``.
    """

    times: NDArray[np.float64]
    alpha: NDArray[np.complex128]
    beta: NDArray[np.complex128]
    order_parameter: NDArray[np.complex128]
    constants: NDArray[np.complex128]
    harmonic_phases: NDArray[np.float64]
    harmonic: int

    @property
    def invariant(self) -> NDArray[np.float64]:
        """The ``SU(1,1)`` invariant ``|α|² − |β|²`` along the flow (unity)."""
        return np.asarray(np.abs(self.alpha) ** 2 - np.abs(self.beta) ** 2, dtype=np.float64)


def integrate_higher_order_watanabe_strogatz(
    initial_phases: NDArray[np.float64],
    *,
    omega: float,
    coupling: float,
    harmonic: int,
    dt: float,
    n_steps: int,
) -> HigherOrderWatanabeStrogatzTrajectory:
    r"""Integrate the ``p``-th harmonic identical-oscillator flow by the Watanabe–Strogatz reduction.

    Advances the two complex Möbius parameters ``(α, β)`` of the ``φ = pθ`` flow by a fixed-step RK4,
    with the common field recomputed from the reconstructed ``p``-th harmonic order parameter each
    stage, and reconstructs the ``p``-fold phases from the fixed constants of motion.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (one-dimensional, length ``N``).
    omega : float
        The common natural frequency ``ω``.
    coupling : float
        The mean-field coupling strength ``K``.
    harmonic : int
        The coupling harmonic ``p`` (``≥ 1``); ``p = 1`` recovers the classical reduction.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``).

    Returns
    -------
    HigherOrderWatanabeStrogatzTrajectory
        The Möbius-parameter, harmonic order-parameter and reconstructed ``p``-fold-phase trajectory.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    theta = np.ascontiguousarray(initial_phases, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 1:
        raise ValueError("initial_phases must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(theta)):
        raise ValueError("initial_phases must be finite")
    if not np.isfinite(omega):
        raise ValueError(f"omega must be finite, got {omega}")
    if not np.isfinite(coupling):
        raise ValueError(f"coupling must be finite, got {coupling}")
    if harmonic < 1:
        raise ValueError(f"harmonic must be positive, got {harmonic}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    count = theta.size
    constants = np.ascontiguousarray(np.exp(1j * harmonic * theta), dtype=np.complex128)
    frequency = harmonic * omega
    gain = harmonic * coupling

    def derivative(alpha: complex, beta: complex) -> tuple[complex, complex]:
        order_parameter = np.mean(_mobius_image(alpha, beta, constants))
        field = gain * order_parameter
        d_alpha = 0.5j * frequency * alpha + 0.5 * field * np.conj(beta)
        d_beta = 0.5j * frequency * beta + 0.5 * field * np.conj(alpha)
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
    return HigherOrderWatanabeStrogatzTrajectory(
        times=times,
        alpha=alpha_history,
        beta=beta_history,
        order_parameter=order_history,
        constants=constants,
        harmonic_phases=phase_history,
        harmonic=harmonic,
    )


__all__ = [
    "HigherOrderWatanabeStrogatzTrajectory",
    "integrate_higher_order_watanabe_strogatz",
]
