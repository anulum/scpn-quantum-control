# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lyapunov / contraction certificates for Kuramoto synchronisation
r"""Lyapunov and contraction certificates for the stability of a Kuramoto synchronised state.

Beyond the toolkit's linear stability spectrum, this module issues *certificates* — verifiable
guarantees, with a rate, of stability over a region rather than only at a point. For a network with
symmetric coupling ``K`` the Kuramoto flow is a gradient flow ``\dot\theta = -\nabla V`` of the
rotating-frame potential

.. math::

    V(\theta) = -\tfrac12\sum_{ij} K_{ij}\cos(\theta_i - \theta_j) - \sum_i \omega_i\theta_i,

so ``V`` is a **Lyapunov function**: ``\dot V = \nabla V\cdot\dot\theta = -\lVert\dot\theta\rVert^2 \le 0``,
non-increasing along every trajectory (Dörfler & Bullo, 2014).

A sharper, region-wise guarantee follows from **contraction analysis** (Lohmiller & Slotine, 1998).
The synchronisation Jacobian is the negative weighted graph Laplacian ``J = -L(w)`` with edge weights
``w_{ij} = K_{ij}\cos(\theta_i - \theta_j)``. Inside the *phase-cohesive* region — where every coupled
pair satisfies ``|\theta_i - \theta_j| < \pi/2`` so all weights are positive — ``L(w)`` is a genuine
Laplacian and ``J`` is negative semidefinite with the only null direction the global phase shift. The
network is therefore *contracting* transverse to that mode at the rate given by the algebraic
connectivity (Fiedler value) of ``L(w)``: any two trajectories converge exponentially, so the region
holds a unique, exponentially stable phase-locked state. A cohesive, contracting configuration is thus
certified stable. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian


@dataclass(frozen=True)
class SynchronisationCertificate:
    """A Lyapunov / contraction certificate for a candidate synchronised state.

    Attributes
    ----------
    phase_cohesiveness : float
        The largest coupled pairwise phase difference ``max_{K_ij≠0} |θ_i - θ_j|`` (wrapped to
        ``[0, π]``).
    is_cohesive : bool
        Whether the configuration is phase-cohesive (``phase_cohesiveness < cohesiveness_threshold``).
    contraction_rate : float
        The transverse contraction rate (the algebraic connectivity of ``L(w)``); positive iff the
        configuration is contracting.
    is_contracting : bool
        Whether ``contraction_rate > 0``.
    lyapunov_value : float
        The Lyapunov potential ``V`` at the configuration.
    lyapunov_decrease_rate : float
        ``\\dot V = -\\lVert\\dot\\theta\\rVert^2`` at the configuration (``≤ 0``).
    is_certified : bool
        Whether the configuration lies in a cohesive, contracting region — certified to hold a unique,
        exponentially stable phase-locked state.
    """

    phase_cohesiveness: float
    is_cohesive: bool
    contraction_rate: float
    is_contracting: bool
    lyapunov_value: float
    lyapunov_decrease_rate: float
    is_certified: bool


def _validate(
    phases: NDArray[np.float64], coupling: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    angle = np.ascontiguousarray(phases, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if angle.ndim != 1 or angle.size < 2:
        raise ValueError("phases must be a one-dimensional array of length at least two")
    count = angle.size
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if not (np.all(np.isfinite(angle)) and np.all(np.isfinite(matrix))):
        raise ValueError("phases and coupling must be finite")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("coupling must be symmetric")
    return angle, matrix


def phase_cohesiveness(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> float:
    r"""The largest coupled pairwise phase difference ``max_{K_ij≠0} |θ_i - θ_j|`` (wrapped to ``[0, π]``)."""
    angle, matrix = _validate(phases, coupling)
    difference = angle[:, None] - angle[None, :]
    wrapped = np.abs((difference + np.pi) % (2.0 * np.pi) - np.pi)
    coupled = matrix != 0.0
    if not np.any(coupled):
        return 0.0
    return float(np.max(wrapped[coupled]))


def contraction_rate(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> float:
    r"""The transverse contraction rate (the algebraic connectivity of ``L(w) = -J``).

    Returns the negative of the largest eigenvalue of the synchronisation Jacobian transverse to the
    global phase-shift mode; positive iff the configuration is contracting.
    """
    angle, matrix = _validate(phases, coupling)
    jacobian = networked_kuramoto_jacobian(angle, matrix)
    eigenvalues = np.linalg.eigvalsh(0.5 * (jacobian + jacobian.T))
    return float(-eigenvalues[-2])


def synchronisation_potential(
    phases: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    r"""The Lyapunov potential ``V = -½ Σ K_ij cos(θ_i-θ_j) - Σ ω_i θ_i`` (gradient ``∇V = -f``)."""
    angle, matrix = _validate(phases, coupling)
    frequencies = _validate_omega(omega, angle.size)
    difference = angle[:, None] - angle[None, :]
    interaction = -0.5 * float(np.sum(matrix * np.cos(difference)))
    return interaction - float(np.sum(frequencies * angle))


def potential_decrease_rate(
    phases: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> float:
    r"""The Lyapunov decrease ``\dot V = -\lVert\dot\theta\rVert^2`` along the flow (``≤ 0``)."""
    angle, matrix = _validate(phases, coupling)
    frequencies = _validate_omega(omega, angle.size)
    field = frequencies + networked_kuramoto_force(angle, matrix)
    return -float(field @ field)


def _validate_omega(omega: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("omega must be finite")
    return frequencies


def certify_synchronisation(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    cohesiveness_threshold: float = np.pi / 2.0,
) -> SynchronisationCertificate:
    r"""Certify the stability of a candidate synchronised state by cohesiveness and contraction.

    Parameters
    ----------
    phases : numpy.ndarray
        The candidate phase configuration ``θ`` (length ``N ≥ 2``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The symmetric ``(N, N)`` coupling matrix ``K``.
    cohesiveness_threshold : float
        The phase-cohesiveness bound (default ``π/2``); must lie in ``(0, π]``.

    Returns
    -------
    SynchronisationCertificate
        The cohesiveness, contraction rate, Lyapunov value and decrease, and the certified verdict.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    _validate(phases, coupling)
    if not 0.0 < cohesiveness_threshold <= np.pi:
        raise ValueError(
            f"cohesiveness_threshold must lie in (0, pi], got {cohesiveness_threshold}"
        )
    cohesiveness = phase_cohesiveness(phases, coupling)
    rate = contraction_rate(phases, coupling)
    is_cohesive = cohesiveness < cohesiveness_threshold
    is_contracting = rate > 0.0
    return SynchronisationCertificate(
        phase_cohesiveness=cohesiveness,
        is_cohesive=is_cohesive,
        contraction_rate=rate,
        is_contracting=is_contracting,
        lyapunov_value=synchronisation_potential(phases, omega, coupling),
        lyapunov_decrease_rate=potential_decrease_rate(phases, omega, coupling),
        is_certified=is_cohesive and is_contracting,
    )


__all__ = [
    "SynchronisationCertificate",
    "certify_synchronisation",
    "contraction_rate",
    "phase_cohesiveness",
    "potential_decrease_rate",
    "synchronisation_potential",
]
