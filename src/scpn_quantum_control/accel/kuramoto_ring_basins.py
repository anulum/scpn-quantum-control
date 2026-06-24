# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Basins of attraction of twisted states on a Kuramoto ring
r"""Basins of attraction of the synchronised and twisted states on a Kuramoto ring.

``N`` identical oscillators on a ring, each coupled to its ``R`` nearest neighbours on either side,
``θ̇_j = (K/2R) Σ_{k=1}^{R} [sin(θ_{j+k} − θ_j) + sin(θ_{j−k} − θ_j)]``. With a symmetric coupling
the flow is the gradient descent ``θ̇ = −∇E`` of the interaction energy
``E = −½ Σ_{j,l} K_{jl} cos(θ_l − θ_j)`` (Wiley, Strogatz & Girvan 2006), so every trajectory relaxes
monotonically to a fixed point. The fixed points are the **q-twisted states** ``θ_j = 2πqj/N``; the
gradient stopping criterion ``‖θ̇‖ < tol`` reads off convergence and the winding number ``q``
classifies which state was reached.

A twisted state's stability follows in closed form from the circulant Jacobian:
``λ_m = (K/R) Σ_{k=1}^{R} cos(2πqk/N) [cos(2πmk/N) − 1]`` for ``m = 0 … N−1`` (``λ_0 = 0`` is the
global-phase Goldstone mode). For nearest-neighbour coupling this reduces to stability iff
``cos(2πq/N) > 0``, i.e. ``|q| < N/4`` — the synchronised state ``q = 0`` together with the
low-winding twisted states. The basin estimate is a Monte-Carlo over random initial phases relaxed by
this gradient flow, tallying the winding number each one lands in; the synchronised basin shrinks
below one for short-range coupling and fills to one as the range grows to all-to-all (only ``q = 0``
remains stable).

This is an analysis layer over the ring dynamics: the relaxation composes the polyglot
:func:`~scpn_quantum_control.accel.networked_kuramoto.networked_kuramoto_force` and the
stability eigenvalues are evaluated in closed form, so the module adds no compute kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force


def ring_coupling_matrix(
    n: int, coupling_range: int, *, coupling: float = 1.0
) -> NDArray[np.float64]:
    r"""Symmetric circulant coupling matrix of a ring with ``coupling_range`` nearest neighbours.

    Each oscillator couples to its ``R = coupling_range`` neighbours on either side with weight
    ``K / (2R)``, so the row sum is ``K`` and the synchronised state is a fixed point.

    Parameters
    ----------
    n : int
        The number of oscillators on the ring (``≥ 3``).
    coupling_range : int
        The number ``R`` of neighbours coupled on each side (``1 ≤ R ≤ (n − 1) // 2``, so the two
        sides never overlap).
    coupling : float, optional
        The total coupling strength ``K`` (non-zero; positive for synchronising coupling).

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` symmetric coupling matrix with zero diagonal.

    Raises
    ------
    ValueError
        If ``n``, ``coupling_range`` or ``coupling`` is out of range.
    """
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    if not 1 <= coupling_range <= (n - 1) // 2:
        raise ValueError(
            f"coupling_range must be in [1, {(n - 1) // 2}] for n={n}, got {coupling_range}"
        )
    if coupling == 0.0:
        raise ValueError("coupling must be non-zero")
    weight = coupling / (2.0 * coupling_range)
    matrix = np.zeros((n, n), dtype=np.float64)
    indices = np.arange(n)
    for offset in range(1, coupling_range + 1):
        matrix[indices, (indices + offset) % n] = weight
        matrix[indices, (indices - offset) % n] = weight
    return matrix


def twisted_state(n: int, winding: int) -> NDArray[np.float64]:
    r"""The q-twisted state ``θ_j = 2π·winding·j / N`` on a ring of ``n`` oscillators.

    Parameters
    ----------
    n : int
        The number of oscillators (``≥ 1``).
    winding : int
        The winding number ``q`` (phase advances by ``2πq`` around the ring).

    Returns
    -------
    numpy.ndarray
        The ``n`` twisted phases.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return 2.0 * math.pi * winding * np.arange(n, dtype=np.float64) / n


def winding_number(theta: NDArray[np.float64]) -> int:
    r"""The winding number ``q = (1/2π) Σ_j wrap(θ_{j+1} − θ_j)`` of a ring phase configuration.

    The nearest-neighbour phase differences are wrapped to ``(−π, π]`` and summed around the ring;
    the total is an integer multiple of ``2π`` for a twisted state.

    Parameters
    ----------
    theta : numpy.ndarray
        The ``n`` ring phases (``n ≥ 1``).

    Returns
    -------
    int
        The winding number.

    Raises
    ------
    ValueError
        If ``theta`` is not a non-empty one-dimensional array.
    """
    phases = np.ascontiguousarray(theta, dtype=np.float64)
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("theta must be a non-empty one-dimensional array")
    differences = np.diff(np.concatenate([phases, phases[:1]]))
    wrapped = (differences + math.pi) % (2.0 * math.pi) - math.pi
    return int(round(float(np.sum(wrapped)) / (2.0 * math.pi)))


def twisted_state_eigenvalues(
    n: int, winding: int, coupling_range: int, *, coupling: float = 1.0
) -> NDArray[np.float64]:
    r"""Closed-form Jacobian eigenvalues of a q-twisted state on a ring.

    Returns ``λ_m = (K/R) Σ_{k=1}^{R} cos(2πqk/N) [cos(2πmk/N) − 1]`` for ``m = 0 … N−1``; ``λ_0 = 0``
    is the global-phase Goldstone mode. The state is linearly stable when every other eigenvalue is
    negative.

    Parameters
    ----------
    n : int
        The number of oscillators (``≥ 3``).
    winding : int
        The winding number ``q``.
    coupling_range : int
        The neighbour range ``R`` (``1 ≤ R ≤ (n − 1) // 2``).
    coupling : float, optional
        The total coupling strength ``K``.

    Returns
    -------
    numpy.ndarray
        The ``n`` eigenvalues, ordered by mode index ``m``.

    Raises
    ------
    ValueError
        If ``n`` or ``coupling_range`` is out of range.
    """
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}")
    if not 1 <= coupling_range <= (n - 1) // 2:
        raise ValueError(
            f"coupling_range must be in [1, {(n - 1) // 2}] for n={n}, got {coupling_range}"
        )
    neighbours = np.arange(1, coupling_range + 1)
    twist_cosines = np.cos(2.0 * math.pi * winding * neighbours / n)
    modes = np.arange(n)
    mode_cosines = np.cos(2.0 * math.pi * np.outer(modes, neighbours) / n) - 1.0
    weight = coupling / coupling_range
    return np.asarray(weight * (mode_cosines @ twist_cosines), dtype=np.float64)


def is_twisted_state_stable(
    n: int, winding: int, coupling_range: int, *, coupling: float = 1.0, tolerance: float = 1e-9
) -> bool:
    r"""Whether the q-twisted state is linearly stable (all non-Goldstone eigenvalues negative).

    For nearest-neighbour coupling this is equivalent to ``|q| < N / 4``.

    Parameters
    ----------
    n : int
        The number of oscillators (``≥ 3``).
    winding : int
        The winding number ``q``.
    coupling_range : int
        The neighbour range ``R``.
    coupling : float, optional
        The total coupling strength ``K`` (positive for synchronising coupling).
    tolerance : float, optional
        The margin below zero an eigenvalue must clear to count as stable (a marginal ``λ_m = 0`` is
        not asymptotically stable).

    Returns
    -------
    bool
        Whether the state is linearly stable.
    """
    eigenvalues = twisted_state_eigenvalues(n, winding, coupling_range, coupling=coupling)
    transverse = np.delete(eigenvalues, 0)  # drop the Goldstone mode m = 0 (always λ_0 = 0)
    return bool(float(transverse.max()) < -tolerance)


def _relax_to_fixed_point(
    initial_phases: NDArray[np.float64],
    coupling_matrix: NDArray[np.float64],
    dt: float,
    max_steps: int,
    force_tolerance: float,
) -> tuple[NDArray[np.float64], bool]:
    """Relax ``θ̇ = networked force`` by RK4, stopping when the gradient (force) norm falls below tol."""
    phases = np.array(initial_phases, dtype=np.float64)
    for _ in range(max_steps):
        k1 = networked_kuramoto_force(phases, coupling_matrix)
        if float(np.max(np.abs(k1))) < force_tolerance:
            return phases, True
        k2 = networked_kuramoto_force(phases + 0.5 * dt * k1, coupling_matrix)
        k3 = networked_kuramoto_force(phases + 0.5 * dt * k2, coupling_matrix)
        k4 = networked_kuramoto_force(phases + dt * k3, coupling_matrix)
        phases = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    converged = (
        float(np.max(np.abs(networked_kuramoto_force(phases, coupling_matrix)))) < force_tolerance
    )
    return phases, converged


@dataclass(frozen=True)
class BasinEstimate:
    """Monte-Carlo tally of the winding numbers reached from random initial phases on a ring.

    Attributes
    ----------
    n_samples : int
        The number of random initial conditions drawn.
    n_converged : int
        How many relaxed to a fixed point within the step budget.
    winding_values : numpy.ndarray
        The distinct winding numbers reached (sorted), among the converged samples.
    winding_counts : numpy.ndarray
        The number of converged samples landing in each winding number, aligned with
        ``winding_values``.
    """

    n_samples: int
    n_converged: int
    winding_values: NDArray[np.int64]
    winding_counts: NDArray[np.int64]

    @property
    def converged_fraction(self) -> float:
        """The fraction of samples that relaxed to a fixed point."""
        return self.n_converged / self.n_samples

    @property
    def sync_basin_fraction(self) -> float:
        """The fraction of converged samples that reached the synchronised state ``q = 0``."""
        return self.basin_fraction(0)

    @property
    def mean_winding(self) -> float:
        """The count-weighted mean winding number over the converged samples (``0`` if none)."""
        if self.n_converged == 0:
            return 0.0
        return float(np.dot(self.winding_values, self.winding_counts) / self.n_converged)

    def basin_fraction(self, winding: int) -> float:
        """The fraction of converged samples that reached the given winding number."""
        if self.n_converged == 0:
            return 0.0
        match = np.flatnonzero(self.winding_values == winding)
        if match.size == 0:
            return 0.0
        return float(self.winding_counts[match[0]]) / self.n_converged


def estimate_ring_basins(
    n: int,
    coupling_range: int,
    n_samples: int,
    *,
    coupling: float = 1.0,
    dt: float = 0.1,
    max_steps: int = 2000,
    force_tolerance: float = 1e-6,
    seed: int,
) -> BasinEstimate:
    r"""Estimate the basins of attraction of the twisted states on a Kuramoto ring by Monte-Carlo.

    Draws ``n_samples`` random phase configurations (uniform on ``[−π, π)``), relaxes each by the
    gradient flow of the ring and tallies the winding number it lands in. The synchronised basin is
    the ``q = 0`` share; short-range coupling leaves a finite share in the twisted states while a
    range approaching all-to-all funnels every sample into ``q = 0``.

    Parameters
    ----------
    n : int
        The number of oscillators (``≥ 3``).
    coupling_range : int
        The neighbour range ``R``.
    n_samples : int
        The number of random initial conditions (``≥ 1``).
    coupling : float, optional
        The total coupling strength ``K`` (positive for synchronising coupling).
    dt : float, optional
        The RK4 relaxation time step (``> 0``).
    max_steps : int, optional
        The maximum relaxation steps per sample (``≥ 1``).
    force_tolerance : float, optional
        The gradient-norm threshold below which a trajectory is a converged fixed point (``> 0``).
    seed : int
        The seed of the Monte-Carlo random generator (required for reproducibility).

    Returns
    -------
    BasinEstimate
        The winding-number tally and the derived basin fractions.

    Raises
    ------
    ValueError
        If ``n_samples``, ``dt``, ``max_steps`` or ``force_tolerance`` is out of range (the coupling
        matrix validates ``n``, ``coupling_range`` and ``coupling``).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if force_tolerance <= 0.0:
        raise ValueError(f"force_tolerance must be positive, got {force_tolerance}")

    matrix = ring_coupling_matrix(n, coupling_range, coupling=coupling)
    rng = np.random.default_rng(seed)
    windings: list[int] = []
    for _ in range(n_samples):
        initial = rng.uniform(-math.pi, math.pi, size=n)
        relaxed, converged = _relax_to_fixed_point(initial, matrix, dt, max_steps, force_tolerance)
        if converged:
            windings.append(winding_number(relaxed))

    if windings:
        values, counts = np.unique(np.array(windings, dtype=np.int64), return_counts=True)
    else:
        values = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.int64)
    return BasinEstimate(
        n_samples=n_samples,
        n_converged=len(windings),
        winding_values=values,
        winding_counts=counts.astype(np.int64),
    )


__all__ = [
    "BasinEstimate",
    "estimate_ring_basins",
    "is_twisted_state_stable",
    "ring_coupling_matrix",
    "twisted_state",
    "twisted_state_eigenvalues",
    "winding_number",
]
