# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gauge/lattice confinement cross-check
"""Joint confinement report: quantum Wilson loops vs classical U(1) lattice MC.

Two complementary routes probe gauge dynamics on the same SCPN coupling
topology:

* the quantum route (``gauge.confinement_analysis``) measures Wilson-loop
  expectations in the XY ground state and extracts a string tension from the
  triangle/square area law; and
* the classical route (``psi_field``) samples the compact U(1) plaquette
  action on the identical graph with Hybrid Monte Carlo and reads the
  lowest-order string tension, topological charge, and link observables.

The two routes probe DIFFERENT regimes (ground-state quantum expectation vs
finite-``beta`` classical thermal ensemble), so this surface reports both
side by side rather than asserting equality; the shared topology makes it
the natural comparison for the BKT/vortex lane, and it is the production
consumer that turns the ``psi_field`` lattice engine from a tested leaf into
load-bearing code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..psi_field.lattice import hmc_update
from ..psi_field.observables import (
    average_link,
    string_tension_from_wilson,
    topological_charge,
)
from ..psi_field.scpn_mapping import scpn_to_lattice
from .confinement import ConfinementResult, confinement_analysis


@dataclass(frozen=True)
class GaugeLatticeCrosscheck:
    """Side-by-side confinement report from the quantum and lattice routes."""

    quantum: ConfinementResult
    lattice_string_tension: float | None
    lattice_mean_plaquette: float
    lattice_n_plaquettes: int
    lattice_topological_charge: float
    lattice_average_link_magnitude: float
    hmc_acceptance_rate: float
    n_thermalisation_steps: int
    beta: float

    @property
    def both_tensions_available(self) -> bool:
        """Whether both routes produced a finite string tension."""
        return self.quantum.string_tension is not None and self.lattice_string_tension is not None


def crosscheck_confinement_on_lattice(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    beta: float = 1.0,
    n_thermalisation: int = 200,
    n_leapfrog: int = 10,
    step_size: float = 0.1,
    seed: int | None = None,
) -> GaugeLatticeCrosscheck:
    """Run both confinement probes on one coupling topology.

    Parameters
    ----------
    K : NDArray[np.float64]
        Symmetric coupling matrix, shape ``(n, n)``; non-zero entries define
        the gauge-link graph for both routes.
    omega : NDArray[np.float64]
        Natural frequencies, shape ``(n,)`` (quantum route only).
    beta : float, optional
        Inverse gauge coupling of the classical lattice ensemble; positive.
    n_thermalisation : int, optional
        HMC updates before measuring; at least 1.
    n_leapfrog : int, optional
        Leapfrog steps per HMC update; at least 1.
    step_size : float, optional
        Leapfrog step size; positive.
    seed : int or None, optional
        Lattice RNG seed for reproducible sampling.

    Returns
    -------
    GaugeLatticeCrosscheck
        The quantum confinement result plus classical lattice observables
        measured after thermalisation, with the HMC acceptance rate as a
        sampling-health indicator.

    Raises
    ------
    ValueError
        If ``K`` is not square-symmetric, ``omega`` has the wrong shape, or
        a sampling parameter is out of range.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    omega_arr = np.asarray(omega, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K must be square, got shape {K_arr.shape}")
    if not np.allclose(K_arr, K_arr.T):
        raise ValueError("K must be symmetric")
    if omega_arr.shape != (K_arr.shape[0],):
        raise ValueError(f"omega shape must be ({K_arr.shape[0]},), got {omega_arr.shape}")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if n_thermalisation < 1:
        raise ValueError("n_thermalisation must be >= 1")
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog must be >= 1")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    quantum = confinement_analysis(K_arr, omega_arr)

    lattice = scpn_to_lattice(K_arr, omega_arr, beta=beta, seed=seed)
    accepted = 0
    for _ in range(n_thermalisation):
        step_accepted, _ = hmc_update(lattice.gauge, n_leapfrog=n_leapfrog, step_size=step_size)
        accepted += int(step_accepted)

    plaquettes = lattice.gauge.measure_plaquettes()

    return GaugeLatticeCrosscheck(
        quantum=quantum,
        lattice_string_tension=string_tension_from_wilson(lattice.gauge),
        lattice_mean_plaquette=plaquettes.mean_plaquette,
        lattice_n_plaquettes=plaquettes.n_plaquettes,
        lattice_topological_charge=topological_charge(lattice.gauge),
        lattice_average_link_magnitude=float(abs(average_link(lattice.gauge))),
        hmc_acceptance_rate=accepted / n_thermalisation,
        n_thermalisation_steps=n_thermalisation,
        beta=beta,
    )


__all__ = ["GaugeLatticeCrosscheck", "crosscheck_confinement_on_lattice"]
