# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lyapunov spectrum of the networked Kuramoto model
r"""Lyapunov spectrum of the networked Kuramoto model.

The Lyapunov exponents measure the average exponential growth rates of infinitesimal phase
perturbations along a trajectory. A perturbation ``δθ`` obeys the variational equation
``δθ̇ = J(θ(t)) δθ`` with the networked stability Jacobian ``J``; integrating a set of orthonormal
perturbation vectors alongside the state and periodically reorthonormalising them with a QR
decomposition (the Benettin algorithm) yields the spectrum from the accumulated logarithms of the
stretching factors.

The uniform-shift direction lies in the kernel of ``J`` at every state, so one exponent is always
zero — the Goldstone mode of the global phase-rotation symmetry. At a fixed point the spectrum
reduces to the real parts of the eigenvalues of the (constant) Jacobian, and the sum of the
exponents equals the time-averaged trace of ``J`` (the phase-space contraction rate) at any state.

This is an analysis layer over the differentiable simulation lane: the state and its tangent
vectors are co-integrated with a fixed-step RK4 built from the multi-language
:func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_force` and
:func:`~oscillatools.accel.networked_kuramoto.networked_kuramoto_jacobian`, so the module
adds no compute kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian


def _joint_rk4_step(
    theta: NDArray[np.float64],
    tangents: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the state and its tangent vectors by one joint RK4 step.

    The state follows ``θ̇ = ω + F(θ)`` and the tangents follow ``Q̇ = J(θ) Q``, with the
    networked force ``F`` and Jacobian ``J`` evaluated at every RK4 stage.
    """

    def state_rate(state: NDArray[np.float64]) -> NDArray[np.float64]:
        return omega + networked_kuramoto_force(state, coupling)

    def tangent_rate(
        state: NDArray[np.float64], vectors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(networked_kuramoto_jacobian(state, coupling) @ vectors, dtype=np.float64)

    k1_state = state_rate(theta)
    k1_tan = tangent_rate(theta, tangents)
    k2_state = state_rate(theta + 0.5 * dt * k1_state)
    k2_tan = tangent_rate(theta + 0.5 * dt * k1_state, tangents + 0.5 * dt * k1_tan)
    k3_state = state_rate(theta + 0.5 * dt * k2_state)
    k3_tan = tangent_rate(theta + 0.5 * dt * k2_state, tangents + 0.5 * dt * k2_tan)
    k4_state = state_rate(theta + dt * k3_state)
    k4_tan = tangent_rate(theta + dt * k3_state, tangents + dt * k3_tan)
    next_theta = theta + (dt / 6.0) * (k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state)
    next_tan = tangents + (dt / 6.0) * (k1_tan + 2.0 * k2_tan + 2.0 * k3_tan + k4_tan)
    return (
        np.ascontiguousarray(next_theta, dtype=np.float64),
        np.ascontiguousarray(next_tan, dtype=np.float64),
    )


def _validate(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    num_exponents: int,
    reorth_interval: int,
    transient_steps: int,
) -> None:
    count = theta0.size
    if count < 1:
        raise ValueError("at least one oscillator is required")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not 1 <= num_exponents <= count:
        raise ValueError(f"num_exponents must be in [1, {count}], got {num_exponents}")
    if not 1 <= reorth_interval <= n_steps:
        raise ValueError(f"reorth_interval must be in [1, {n_steps}], got {reorth_interval}")
    if transient_steps < 0:
        raise ValueError(f"transient_steps must be non-negative, got {transient_steps}")
    last_reorth = (n_steps // reorth_interval) * reorth_interval
    if last_reorth <= transient_steps:
        raise ValueError(
            "transient_steps leaves no post-transient reorthonormalisation; reduce it or extend "
            "the integration"
        )


def lyapunov_spectrum(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    num_exponents: int | None = None,
    reorth_interval: int = 1,
    transient_steps: int = 0,
) -> NDArray[np.float64]:
    r"""Lyapunov exponents of the networked Kuramoto flow via the Benettin QR algorithm.

    Co-integrates the state and ``num_exponents`` orthonormal tangent vectors with a joint RK4,
    reorthonormalising the tangents every ``reorth_interval`` steps and averaging the logarithms of
    the QR stretching factors over the post-transient time.

    Parameters
    ----------
    theta0 : numpy.ndarray
        The ``(N,)`` initial phases.
    omega : numpy.ndarray
        The ``(N,)`` natural frequencies.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The integration step.
    n_steps : int
        The number of integration steps.
    num_exponents : int, optional
        How many exponents to compute (the leading ones); defaults to the full ``N``.
    reorth_interval : int, optional
        The number of steps between QR reorthonormalisations.
    transient_steps : int, optional
        Steps to evolve before averaging begins (the tangents are still reorthonormalised, so they
        align with the dominant directions, but their stretching is not accumulated).

    Returns
    -------
    numpy.ndarray
        The ``num_exponents`` Lyapunov exponents, ordered by decreasing value.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, ``dt`` or ``n_steps`` is non-positive,
        ``num_exponents`` is outside ``[1, N]``, ``reorth_interval`` is outside ``[1, n_steps]``,
        or ``transient_steps`` leaves no post-transient reorthonormalisation.
    """
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = phases.size
    exponent_count = count if num_exponents is None else num_exponents
    _validate(
        phases, frequencies, matrix, dt, n_steps, exponent_count, reorth_interval, transient_steps
    )

    tangents = np.ascontiguousarray(np.eye(count, exponent_count), dtype=np.float64)
    log_sums = np.zeros(exponent_count, dtype=np.float64)
    accumulated_time = 0.0
    for step in range(n_steps):
        phases, tangents = _joint_rk4_step(phases, tangents, frequencies, matrix, dt)
        if (step + 1) % reorth_interval == 0:
            tangents, upper = np.linalg.qr(tangents)
            if step >= transient_steps:
                log_sums += np.log(np.abs(np.diag(upper)))
                accumulated_time += reorth_interval * dt
    exponents = log_sums / accumulated_time
    return np.ascontiguousarray(np.sort(exponents)[::-1], dtype=np.float64)


def maximal_lyapunov_exponent(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    reorth_interval: int = 1,
    transient_steps: int = 0,
) -> float:
    r"""Largest Lyapunov exponent of the networked Kuramoto flow.

    The maximal exponent — positive in a chaotic regime, zero at a marginally stable (Goldstone)
    state, negative for a contracting one. Computes a single tangent vector; parameters are as for
    :func:`lyapunov_spectrum`.
    """
    return float(
        lyapunov_spectrum(
            theta0,
            omega,
            coupling,
            dt,
            n_steps,
            num_exponents=1,
            reorth_interval=reorth_interval,
            transient_steps=transient_steps,
        )[0]
    )


__all__ = [
    "lyapunov_spectrum",
    "maximal_lyapunov_exponent",
]
