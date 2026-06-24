# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Noisy Kuramoto mean-field / Fokker–Planck onset and stationary coherence
r"""Fokker–Planck mean-field theory of the noisy Kuramoto model — onset and stationary coherence.

For the noisy mean-field Kuramoto ``dθ = (ω + K r sin(ψ − θ)) dt + √(2D)\,dW`` the population
density obeys a Fokker–Planck equation. Linearising about incoherence gives the **noisy onset**

.. math::

    K_c(D) = \frac{2}{\displaystyle\int g(ω)\,\frac{D}{D^2 + ω^2}\,dω},

which reduces to the deterministic Kuramoto value ``2/(π g(0))`` as ``D → 0`` (the Lorentzian
kernel ``D/(D²+ω²) → π δ(ω)``) and, for a Lorentzian frequency density of half-width ``γ``,
gives the linear shift ``K_c = 2(γ + D)`` — noise adds to the frequency width.

Above the onset the **stationary order parameter** is the self-consistent solution of the
Fokker–Planck fixed point. Each oscillator of frequency ``ω`` relaxes (with ``ψ = 0``) to the
stationary density of a tilted periodic potential ``φ(θ) = −(ω θ + K r cos θ)/D``,

.. math::

    ρ(θ\mid ω) \propto e^{-φ(θ)} \int_θ^{θ+2π} e^{φ(θ')}\,dθ',

whose local coherence ``z(ω) = ∫ e^{iθ} ρ(θ\mid ω)\,dθ`` is averaged over ``g`` to close the
loop ``r = ∫ g(ω)\,\mathrm{Re}\,z(ω)\,dω``. The density is evaluated in the difference exponent
and stabilised with a log-sum-exp, so the strongly tilted (large ``ω``, small ``D``) regime does
not overflow.

This is an analysis layer composing NumPy / SciPy quadrature and the deterministic onset of
:mod:`scpn_quantum_control.accel.kuramoto_critical_coupling` (reused for the ``D = 0`` limit), so
it adds no compute kernel. The matching finite-population simulation is
:func:`~scpn_quantum_control.accel.kuramoto_noisy.integrate_noisy_kuramoto`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

from .kuramoto_critical_coupling import critical_coupling, synchronised_order_parameter

#: A symmetric unimodal natural-frequency density ``g(ω)`` (the signature shared with
#: :mod:`scpn_quantum_control.accel.kuramoto_critical_coupling`).
FrequencyDensity = Callable[[float], float]


def lorentzian_noisy_critical_coupling(half_width: float, diffusion: float) -> float:
    r"""Noisy critical coupling ``K_c = 2(γ + D)`` for a Lorentzian frequency density (closed form).

    The noise intensity adds linearly to the half-width: the deterministic onset ``2γ`` becomes
    ``2(γ + D)``.

    Parameters
    ----------
    half_width : float
        The Lorentzian half-width ``γ > 0``.
    diffusion : float
        The diffusion / noise intensity ``D ≥ 0``.

    Returns
    -------
    float
        The noisy critical coupling.

    Raises
    ------
    ValueError
        If ``half_width`` is not positive or ``diffusion`` is negative.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be non-negative, got {diffusion}")
    return 2.0 * (half_width + diffusion)


def _frequency_grid(
    frequency_limit: float, n_frequency: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a symmetric frequency grid and its trapezoidal weights."""
    omega = np.linspace(-frequency_limit, frequency_limit, n_frequency, dtype=np.float64)
    weights = np.full(n_frequency, omega[1] - omega[0], dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return omega, weights


def _validate_theory(diffusion: float, frequency_limit: float, n_frequency: int) -> None:
    """Validate the diffusion and the frequency-quadrature parameters."""
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be non-negative, got {diffusion}")
    if frequency_limit <= 0.0:
        raise ValueError(f"frequency_limit must be positive, got {frequency_limit}")
    if n_frequency < 3:
        raise ValueError(f"n_frequency must be at least 3, got {n_frequency}")


def noisy_critical_coupling(
    density: FrequencyDensity,
    diffusion: float,
    *,
    frequency_limit: float = 60.0,
    n_frequency: int = 1201,
) -> float:
    r"""Fokker–Planck onset ``K_c = 2 / ∫ g(ω) D/(D²+ω²) dω`` of the noisy Kuramoto model.

    The Lorentzian kernel ``D/(D²+ω²)`` is integrated against the frequency density on a symmetric
    trapezoidal grid. As ``D → 0`` the kernel collapses to ``π δ(ω)`` and the onset returns the
    deterministic Kuramoto value ``2/(π g(0))`` (delegated to
    :func:`~scpn_quantum_control.accel.kuramoto_critical_coupling.critical_coupling`).

    Parameters
    ----------
    density : callable
        The natural-frequency density ``g`` (see :data:`FrequencyDensity`).
    diffusion : float
        The diffusion / noise intensity ``D ≥ 0``.
    frequency_limit : float, optional
        The half-width of the symmetric frequency grid; must cover the bulk of ``g``.
    n_frequency : int, optional
        The number of frequency grid points (``≥ 3``).

    Returns
    -------
    float
        The noisy critical coupling ``K_c(D)``.

    Raises
    ------
    ValueError
        If ``diffusion`` is negative, the grid parameters are out of range, or the kernel integral
        is not strictly positive.
    """
    _validate_theory(diffusion, frequency_limit, n_frequency)
    if diffusion == 0.0:
        return critical_coupling(density)
    omega, weights = _frequency_grid(frequency_limit, n_frequency)
    g_values = np.array([density(float(value)) for value in omega], dtype=np.float64)
    kernel = diffusion / (diffusion * diffusion + omega * omega)
    integral = float(np.sum(weights * g_values * kernel))
    if integral <= 0.0:
        raise ValueError("the Fokker–Planck kernel integral must be positive")
    return 2.0 / integral


def _stationary_local_order(
    omega: float, mean_field: float, diffusion: float, n_phase: int
) -> float:
    r"""Return ``Re z(ω)`` for the tilted-potential stationary density at field ``K r``.

    The unnormalised density ``ρ_m ∝ Σ_j exp[(ω(−j dθ) + mean_field(cos θ_m − cos θ_{m+j}))/D]``
    follows from the tilted-potential integral written in the difference exponent (so it stays
    bounded under strong tilt); a log-sum-exp keeps the row sums overflow-safe.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, n_phase, endpoint=False, dtype=np.float64)
    dtheta = 2.0 * math.pi / n_phase
    cosine = np.cos(theta)
    shift = np.arange(n_phase)
    rolled = (shift[:, None] + shift[None, :]) % n_phase
    cosine_shift = cosine[rolled]  # cos θ_{m+j}
    exponent = (
        omega * (-(shift * dtheta))[None, :] + mean_field * (cosine[:, None] - cosine_shift)
    ) / diffusion
    log_density = logsumexp(exponent, axis=1)
    log_density -= log_density.max()
    density = np.exp(log_density)
    density /= density.sum()
    return float(np.sum(np.cos(theta) * density))


def noisy_stationary_order_parameter(
    coupling: float,
    density: FrequencyDensity,
    diffusion: float,
    *,
    n_phase: int = 256,
    frequency_limit: float = 50.0,
    n_frequency: int = 401,
    root_tolerance: float = 1e-10,
) -> float:
    r"""Self-consistent Fokker–Planck stationary order parameter of the noisy Kuramoto model.

    Below the noisy onset ``K_c(D)`` the only stationary solution is incoherence and ``0`` is
    returned. Above it the self-consistency ``r = ∫ g(ω) Re z(ω; K r) dω`` is closed with the
    tilted-potential stationary density (:func:`_stationary_local_order`) and solved for ``r`` by a
    bracketed Brent root-find on ``(0, 1]``. For ``D = 0`` the deterministic self-consistency of
    :func:`~scpn_quantum_control.accel.kuramoto_critical_coupling.synchronised_order_parameter` is
    used.

    Parameters
    ----------
    coupling : float
        The global coupling strength ``K > 0``.
    density : callable
        The symmetric unimodal natural-frequency density ``g`` (see :data:`FrequencyDensity`).
    diffusion : float
        The diffusion / noise intensity ``D ≥ 0``.
    n_phase : int, optional
        The number of phase grid points for the stationary density (``≥ 8``).
    frequency_limit : float, optional
        The half-width of the symmetric frequency grid; must cover the bulk of ``g``.
    n_frequency : int, optional
        The number of frequency grid points (``≥ 3``).
    root_tolerance : float, optional
        The absolute tolerance of the Brent root-find on ``r``.

    Returns
    -------
    float
        The stationary order parameter ``r`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``coupling`` is not positive, ``diffusion`` is negative, or the grid parameters are out
        of range.
    """
    if coupling <= 0.0:
        raise ValueError(f"coupling must be positive, got {coupling}")
    _validate_theory(diffusion, frequency_limit, n_frequency)
    if n_phase < 8:
        raise ValueError(f"n_phase must be at least 8, got {n_phase}")
    if diffusion == 0.0:
        return synchronised_order_parameter(coupling, density)
    if coupling <= noisy_critical_coupling(
        density, diffusion, frequency_limit=frequency_limit, n_frequency=n_frequency
    ):
        return 0.0

    omega, weights = _frequency_grid(frequency_limit, n_frequency)
    g_weights = weights * np.array([density(float(value)) for value in omega], dtype=np.float64)

    def coherence(radius: float) -> float:
        local = np.array(
            [
                _stationary_local_order(float(value), coupling * radius, diffusion, n_phase)
                for value in omega
            ],
            dtype=np.float64,
        )
        return float(np.sum(g_weights * local))

    # With D > 0 every oscillator's stationary density is noise-broadened, so the local coherence
    # stays below one and the truncated frequency weights sum to less than one — hence
    # ``coherence(1) < 1`` strictly while ``coherence(r) > r`` just above the onset, and the Brent
    # root-find always brackets the non-trivial fixed point on ``(0, 1)``.
    return float(
        brentq(lambda radius: coherence(radius) - radius, root_tolerance, 1.0, xtol=root_tolerance)
    )


__all__ = [
    "FrequencyDensity",
    "lorentzian_noisy_critical_coupling",
    "noisy_critical_coupling",
    "noisy_stationary_order_parameter",
]
