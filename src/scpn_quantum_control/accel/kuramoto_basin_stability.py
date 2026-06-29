# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Basin stability of synchronisation (Menck–Kurths)
r"""Basin stability of the synchronised state — a nonlinear robustness measure.

Linear stability tells whether a synchronised state survives *infinitesimal* perturbations; it
says nothing about *how large* a perturbation it survives. Basin stability (Menck, Heitzig,
Marwan & Kurths, *Nat. Phys.* 2013) is the complementary nonlinear measure: the probability that a
random finite perturbation of the synchronised state returns to it — the fraction of the
perturbation ensemble that lies in the basin of attraction. It is estimated by Monte-Carlo:
draw perturbations (of a single node, of ``k`` nodes, or of all nodes), integrate the dynamics,
and count how many relax back to synchrony.

The two measures can *disagree* — and for higher-order (simplex / hyperedge) coupling they can move
in opposite directions: adding higher-order interactions may improve linear stability while
shrinking the basin (or the reverse), so a synchronised state can be linearly stable yet have a
basin stability well below one because a competing attractor captures part of the perturbation
ensemble. This module evaluates basin stability for *any* Kuramoto force — pairwise, networked,
triadic, simplex or the general hyperedge force re-exported from
:mod:`scpn_quantum_control.accel` — so the higher-order linear-vs-basin divergence is measurable
directly.

This is an analysis layer over the synchronisation dynamics: a fixed-step RK4 over the supplied
force with a Monte-Carlo perturbation sweep, and it adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .order_parameter_observables import order_parameter

#: A phase-coupling force ``F(θ)`` already closed over its coupling parameters (any Kuramoto force
#: bound to its coupling — pairwise, networked, triadic, simplex or hyperedge).
PhaseForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class BasinStabilityEstimate:
    """A Monte-Carlo basin-stability estimate of a synchronised state.

    Attributes
    ----------
    basin_fraction : float
        The estimated basin stability — the fraction of perturbations that returned to synchrony.
    n_samples : int
        The number of Monte-Carlo perturbations drawn.
    n_returned : int
        The number that relaxed back to the synchronised state.
    standard_error : float
        The binomial standard error ``sqrt(p (1 - p) / n)`` of the estimate.
    """

    basin_fraction: float
    n_samples: int
    n_returned: int
    standard_error: float


def _integrate(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: PhaseForce,
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Advance ``θ̇ = ω + F(θ)`` by ``n_steps`` fixed-step RK4 steps."""
    state = phases

    def field(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(omega + force(theta), dtype=np.float64)

    for _ in range(n_steps):
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(state, dtype=np.float64)


def synchronisation_basin_stability(
    force: PhaseForce,
    omega: NDArray[np.float64],
    synchronised_state: NDArray[np.float64],
    *,
    n_samples: int,
    n_perturbed_nodes: int,
    dt: float,
    n_steps: int,
    perturbation: float = np.pi,
    return_tolerance: float = 0.02,
    seed: int,
) -> BasinStabilityEstimate:
    r"""Estimate the basin stability of a synchronised state by Monte-Carlo perturbation.

    Repeatedly perturbs ``n_perturbed_nodes`` randomly chosen oscillators of the synchronised state
    by a uniform phase shift in ``[-perturbation, perturbation]``, integrates ``θ̇ = ω + F(θ)``, and
    counts the fraction that return to the synchronised coherence (final order parameter within
    ``return_tolerance`` of the synchronised value).

    Parameters
    ----------
    force : callable
        The phase-coupling force ``F(θ)`` (any Kuramoto force bound to its coupling).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    synchronised_state : numpy.ndarray
        The synchronised phase configuration whose basin is probed (length ``N``).
    n_samples : int
        The number of Monte-Carlo perturbations (``≥ 1``).
    n_perturbed_nodes : int
        The number of randomly chosen nodes perturbed per sample (``1 ≤ k ≤ N``); ``1`` is the
        single-node basin stability, ``N`` perturbs the whole network.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps used to relax each perturbation (``≥ 1``).
    perturbation : float, optional
        The half-width of the uniform phase perturbation; defaults to ``π`` (the full circle).
    return_tolerance : float, optional
        The order-parameter tolerance for counting a perturbation as returned; defaults to ``0.02``.
    seed : int
        The seed of the Monte-Carlo perturbation generator.

    Returns
    -------
    BasinStabilityEstimate
        The basin fraction, sample counts and binomial standard error.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    reference = np.ascontiguousarray(synchronised_state, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    count = int(frequencies.size)
    if reference.shape != (count,):
        raise ValueError(f"synchronised_state must have shape {(count,)}, got {reference.shape}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not 1 <= n_perturbed_nodes <= count:
        raise ValueError(
            f"n_perturbed_nodes must satisfy 1 <= k <= {count}, got {n_perturbed_nodes}"
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if perturbation <= 0.0:
        raise ValueError(f"perturbation must be positive, got {perturbation}")
    if return_tolerance <= 0.0:
        raise ValueError(f"return_tolerance must be positive, got {return_tolerance}")

    reference_coherence = float(order_parameter(reference))
    generator = np.random.default_rng(seed)
    returned = 0
    for _ in range(n_samples):
        perturbed = reference.copy()
        nodes = generator.choice(count, size=n_perturbed_nodes, replace=False)
        perturbed[nodes] += generator.uniform(-perturbation, perturbation, size=n_perturbed_nodes)
        final = _integrate(perturbed, frequencies, force, dt, n_steps)
        if abs(float(order_parameter(final)) - reference_coherence) <= return_tolerance:
            returned += 1

    fraction = returned / n_samples
    standard_error = float(np.sqrt(fraction * (1.0 - fraction) / n_samples))
    return BasinStabilityEstimate(
        basin_fraction=fraction,
        n_samples=n_samples,
        n_returned=returned,
        standard_error=standard_error,
    )


__all__ = [
    "BasinStabilityEstimate",
    "PhaseForce",
    "synchronisation_basin_stability",
]
