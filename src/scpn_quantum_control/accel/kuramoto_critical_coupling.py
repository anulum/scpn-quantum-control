# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Critical coupling and self-consistent order parameter of the mean-field Kuramoto model
r"""Critical coupling and self-consistent order parameter of the mean-field Kuramoto model.

For the globally coupled Kuramoto model ``θ̇_i = ω_i + K r sin(ψ − θ_i)`` with a symmetric,
unimodal natural-frequency density ``g(ω)`` centred at the mean frequency, synchronisation sets in
at the **critical coupling**

``K_c = 2 / (π g(0))``

(Kuramoto 1975; Strogatz 2000). Below ``K_c`` the incoherent state ``r = 0`` is the only solution;
above it a partially synchronised branch ``r > 0`` appears, fixed by the self-consistency condition

``r = K r ∫_{−π/2}^{π/2} cos²θ · g(K r sinθ) dθ``

(the locked oscillators contribute; the drifting ones average to zero for symmetric ``g``).
Dividing by ``r`` gives the equation ``1 = K ∫_{−π/2}^{π/2} cos²θ · g(K r sinθ) dθ`` solved here for
``r``. The Lorentzian density admits the closed form ``r = √(1 − K_c/K)`` with ``K_c = 2γ``, which
this module both provides directly and reproduces numerically.

This is an analysis layer over the synchronisation theory: the closed forms are evaluated directly
and the general self-consistency is solved with SciPy adaptive quadrature and a bracketed Brent
root-find, so the module adds no compute kernel.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.integrate import quad
from scipy.optimize import brentq

_FrequencyDensity = Callable[[float], float]

# The locked oscillators occupy phases θ ∈ [−π/2, π/2] relative to the mean field (where the
# restoring force has a stable branch); the self-consistency integral runs over that interval.
_LOCKED_PHASE_LIMIT = math.pi / 2.0


def critical_coupling(density: _FrequencyDensity) -> float:
    r"""Critical coupling ``K_c = 2 / (π g(0))`` for a symmetric unimodal frequency density.

    Parameters
    ----------
    density : callable
        The natural-frequency density ``g``; only its value at the centre ``g(0)`` enters the
        onset of synchronisation.

    Returns
    -------
    float
        The critical coupling ``K_c``.

    Raises
    ------
    ValueError
        If ``g(0)`` is not strictly positive.
    """
    centre = float(density(0.0))
    if centre <= 0.0:
        raise ValueError(f"density at the centre must be positive, got g(0)={centre}")
    return 2.0 / (math.pi * centre)


def lorentzian_density(half_width: float) -> _FrequencyDensity:
    r"""Return the Lorentzian (Cauchy) density ``g(ω) = (γ/π) / (ω² + γ²)``.

    Parameters
    ----------
    half_width : float
        The half-width at half-maximum ``γ > 0``.

    Raises
    ------
    ValueError
        If ``half_width`` is not strictly positive.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width}")

    def density(omega: float) -> float:
        return (half_width / math.pi) / (omega * omega + half_width * half_width)

    return density


def gaussian_density(std: float) -> _FrequencyDensity:
    r"""Return the Gaussian density ``g(ω) = exp(−ω²/2σ²) / (σ√(2π))``.

    Parameters
    ----------
    std : float
        The standard deviation ``σ > 0``.

    Raises
    ------
    ValueError
        If ``std`` is not strictly positive.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    normalisation = 1.0 / (std * math.sqrt(2.0 * math.pi))

    def density(omega: float) -> float:
        return normalisation * math.exp(-0.5 * (omega / std) ** 2)

    return density


def lorentzian_critical_coupling(half_width: float) -> float:
    r"""Critical coupling ``K_c = 2γ`` for a Lorentzian density (closed form).

    Parameters
    ----------
    half_width : float
        The half-width at half-maximum ``γ > 0``.

    Raises
    ------
    ValueError
        If ``half_width`` is not strictly positive.
    """
    if half_width <= 0.0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    return 2.0 * half_width


def gaussian_critical_coupling(std: float) -> float:
    r"""Critical coupling ``K_c = σ√(8/π)`` for a Gaussian density (closed form).

    Parameters
    ----------
    std : float
        The standard deviation ``σ > 0``.

    Raises
    ------
    ValueError
        If ``std`` is not strictly positive.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}")
    return std * math.sqrt(8.0 / math.pi)


def lorentzian_order_parameter(coupling: float, half_width: float) -> float:
    r"""Synchronisation order parameter ``r = √(1 − K_c/K)`` for a Lorentzian (closed form).

    Returns ``0`` below the critical coupling ``K_c = 2γ`` and the exact partially synchronised
    branch above it.

    Parameters
    ----------
    coupling : float
        The global coupling strength ``K > 0``.
    half_width : float
        The Lorentzian half-width ``γ > 0``.

    Raises
    ------
    ValueError
        If ``coupling`` or ``half_width`` is not strictly positive.
    """
    if coupling <= 0.0:
        raise ValueError(f"coupling must be positive, got {coupling}")
    critical = lorentzian_critical_coupling(half_width)
    if coupling <= critical:
        return 0.0
    return math.sqrt(1.0 - critical / coupling)


def synchronised_order_parameter(
    coupling: float,
    density: _FrequencyDensity,
    *,
    root_tolerance: float = 1e-12,
) -> float:
    r"""Self-consistent synchronisation order parameter for a general frequency density.

    Solves ``1 = K ∫_{−π/2}^{π/2} cos²θ · g(K r sinθ) dθ`` for ``r`` by SciPy adaptive quadrature
    of the locked-oscillator integral and a bracketed Brent root-find. Below the critical coupling
    ``K_c = 2 / (π g(0))`` the only solution is the incoherent state and ``0`` is returned; in the
    rare over-coherent case where the self-consistency has no sub-unity root the coherence is
    clamped to the ``r = 1`` ceiling.

    Parameters
    ----------
    coupling : float
        The global coupling strength ``K > 0``.
    density : callable
        The symmetric unimodal natural-frequency density ``g``.
    root_tolerance : float, optional
        The absolute tolerance of the Brent root-find on ``r``.

    Returns
    -------
    float
        The order parameter ``r`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``coupling`` is not strictly positive or ``g(0)`` is not strictly positive.
    """
    if coupling <= 0.0:
        raise ValueError(f"coupling must be positive, got {coupling}")
    critical = critical_coupling(density)
    if coupling <= critical:
        return 0.0

    def self_consistency(radius: float) -> float:
        integral, _ = quad(
            lambda theta: math.cos(theta) ** 2 * density(coupling * radius * math.sin(theta)),
            -_LOCKED_PHASE_LIMIT,
            _LOCKED_PHASE_LIMIT,
        )
        return coupling * float(integral) - 1.0

    # The self-consistency is positive as r → 0 (because K > K_c) and decreases as the unimodal
    # density is sampled away from its peak; the root lies in (0, 1] when it dips below zero.
    if self_consistency(1.0) >= 0.0:
        return 1.0
    return float(brentq(self_consistency, root_tolerance, 1.0, xtol=root_tolerance))


__all__ = [
    "critical_coupling",
    "gaussian_critical_coupling",
    "gaussian_density",
    "lorentzian_critical_coupling",
    "lorentzian_density",
    "lorentzian_order_parameter",
    "synchronised_order_parameter",
]
