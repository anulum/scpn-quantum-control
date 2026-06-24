# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Time-delayed Kuramoto mean-field theory, collective-frequency branches
r"""Mean-field theory of the time-delayed Kuramoto model — collective-frequency branches.

For the mean-field coupling of *identical* oscillators (natural frequency ``ω₀``) the fully
phase-locked state ``θ_j(t) = Ω t`` rotates at a collective frequency ``Ω`` that the delay shifts
away from ``ω₀``. Substituting the locked ansatz into ``θ̇_j = ω₀ + K · mean_k sin(θ_k(t-τ) − θ_j(t))``
gives the Yeung–Strogatz (1999) self-consistency

.. math::

    Ω = ω₀ - K \sin(Ω τ).

For small ``K τ`` this has a single root, but once ``K τ`` exceeds ``1`` the sine term folds the
right-hand side back on itself and the equation acquires **several** roots: coexisting
phase-locked branches at different frequencies. A branch is linearly stable when the in-phase
perturbation decays, which for the mean-field locked state is the condition

.. math::

    1 + K τ \cos(Ω τ) > 0,

so the stable and unstable branches interleave. The result is **delay-induced multistability** —
the same coupling and delay admit multiple stable synchronised frequencies, and which one is
realised depends on the initial history (reproduced by the simulation in
:mod:`scpn_quantum_control.accel.kuramoto_delayed`).

This is an analysis layer: the roots of the scalar self-consistency are bracketed on a scan and
refined with :func:`scipy.optimize.brentq`, so the module adds no compute kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


def synchronised_frequency_residual(
    frequency: float,
    natural_frequency: float,
    coupling: float,
    delay: float,
) -> float:
    r"""Return the self-consistency residual ``ω₀ − K sin(Ω τ) − Ω`` at ``Ω = frequency``.

    A collective frequency ``Ω`` is a phase-locked branch exactly when this residual vanishes
    (the Yeung–Strogatz self-consistency ``Ω = ω₀ − K sin(Ω τ)``).

    Parameters
    ----------
    frequency : float
        The trial collective frequency ``Ω``.
    natural_frequency : float
        The common natural frequency ``ω₀`` of the identical oscillators.
    coupling : float
        The mean-field coupling strength ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``).

    Returns
    -------
    float
        The residual ``ω₀ − K sin(Ω τ) − Ω``.
    """
    return float(natural_frequency - coupling * np.sin(frequency * delay) - frequency)


def synchronised_frequency_roots(
    natural_frequency: float,
    coupling: float,
    delay: float,
    *,
    scan_points: int = 4001,
    root_tolerance: float = 1e-12,
) -> NDArray[np.float64]:
    r"""Return all collective-frequency roots of ``Ω = ω₀ − K sin(Ω τ)``.

    Because ``|sin| ≤ 1`` every root lies in ``[ω₀ − |K|, ω₀ + |K|]``; the residual
    :func:`synchronised_frequency_residual` is scanned for sign changes on that interval and each
    bracket is refined with :func:`scipy.optimize.brentq`. At small ``K τ`` there is a single
    root; once ``K τ > 1`` several roots appear (delay-induced multistability).

    Parameters
    ----------
    natural_frequency : float
        The common natural frequency ``ω₀``.
    coupling : float
        The mean-field coupling strength ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``).
    scan_points : int, optional
        The number of grid points used to bracket the roots (``≥ 2``); defaults to ``4001``. The
        scan must be fine enough to separate neighbouring roots, which are spaced by about
        ``π / τ``.
    root_tolerance : float, optional
        The absolute tolerance passed to the bracketed root finder; defaults to ``1e-12``.

    Returns
    -------
    numpy.ndarray
        The sorted distinct roots ``Ω`` (frequencies of the phase-locked branches).

    Raises
    ------
    ValueError
        If ``delay`` is not positive or ``scan_points`` is below ``2``.
    """
    if delay <= 0.0:
        raise ValueError(f"delay must be positive, got {delay}")
    if scan_points < 2:
        raise ValueError(f"scan_points must be at least 2, got {scan_points}")
    span = abs(coupling)
    grid = np.linspace(
        natural_frequency - span - 1.0,
        natural_frequency + span + 1.0,
        scan_points,
    )
    residual = natural_frequency - coupling * np.sin(grid * delay) - grid
    # The +1 scan margins make the residual ≥ 1 at the left end and ≤ −1 at the right end
    # (the −Ω term dominates the bounded sine there), so there is always at least one sign change
    # to bracket; a grid point landing exactly on a root keeps the adjacent brackets valid (the
    # bracketed solver accepts a zero endpoint), so the sign-change scan captures every branch.
    roots: list[float] = []
    for index in np.where(np.diff(np.sign(residual)) != 0)[0]:
        left, right = float(grid[index]), float(grid[index + 1])
        root = brentq(
            synchronised_frequency_residual,
            left,
            right,
            args=(natural_frequency, coupling, delay),
            xtol=root_tolerance,
        )
        roots.append(float(root))
    unique = np.unique(np.round(roots, 10))
    return np.asarray(unique, dtype=np.float64)


def synchronised_branch_stability(
    frequency: float,
    coupling: float,
    delay: float,
) -> float:
    r"""Return the stability indicator ``1 + K τ cos(Ω τ)`` of a phase-locked branch.

    The mean-field locked branch at collective frequency ``Ω`` is linearly stable when this
    indicator is positive and unstable when it is negative (the Yeung–Strogatz in-phase stability
    condition). The magnitude measures how strongly the branch attracts or repels.

    Parameters
    ----------
    frequency : float
        The branch collective frequency ``Ω``.
    coupling : float
        The mean-field coupling strength ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``).

    Returns
    -------
    float
        The stability indicator ``1 + K τ cos(Ω τ)``.
    """
    return 1.0 + coupling * delay * float(np.cos(frequency * delay))


def is_synchronised_branch_stable(
    frequency: float,
    coupling: float,
    delay: float,
) -> bool:
    """Return whether the phase-locked branch at ``frequency`` is linearly stable.

    A thin wrapper over :func:`synchronised_branch_stability`: the branch is stable when the
    indicator ``1 + K τ cos(Ω τ)`` is strictly positive.
    """
    return synchronised_branch_stability(frequency, coupling, delay) > 0.0


def stable_synchronised_frequencies(
    natural_frequency: float,
    coupling: float,
    delay: float,
    *,
    scan_points: int = 4001,
    root_tolerance: float = 1e-12,
) -> NDArray[np.float64]:
    r"""Return only the linearly stable collective-frequency branches.

    Combines :func:`synchronised_frequency_roots` with the stability filter
    :func:`is_synchronised_branch_stable`: the self-consistency roots that satisfy
    ``1 + K τ cos(Ω τ) > 0``. At large ``K τ`` several survive — the coexisting stable
    synchronised frequencies of delay-induced multistability.

    Parameters
    ----------
    natural_frequency : float
        The common natural frequency ``ω₀``.
    coupling : float
        The mean-field coupling strength ``K``.
    delay : float
        The coupling delay ``τ`` (``> 0``).
    scan_points : int, optional
        The bracketing resolution forwarded to :func:`synchronised_frequency_roots`.
    root_tolerance : float, optional
        The root tolerance forwarded to :func:`synchronised_frequency_roots`.

    Returns
    -------
    numpy.ndarray
        The sorted stable collective frequencies ``Ω``.
    """
    roots = synchronised_frequency_roots(
        natural_frequency,
        coupling,
        delay,
        scan_points=scan_points,
        root_tolerance=root_tolerance,
    )
    stable = [root for root in roots if is_synchronised_branch_stable(root, coupling, delay)]
    return np.asarray(stable, dtype=np.float64)


__all__ = [
    "is_synchronised_branch_stable",
    "stable_synchronised_frequencies",
    "synchronised_branch_stability",
    "synchronised_frequency_residual",
    "synchronised_frequency_roots",
]
