# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SCPN Hierarchy to Lattice Mapping
"""Map the 15+1 SCPN layer hierarchy onto a lattice gauge topology.

Each SCPN layer becomes a lattice site. The K_nm coupling matrix
defines the graph adjacency: stronger coupling → stronger gauge link.
The Ψ-field amplitude at each site represents the consciousness
projection strength at that layer.

This module bridges the abstract SCPN hierarchy to the concrete
lattice gauge simulation, enabling Ψ-field dynamics on the actual
SCPN topology instead of a generic hypercubic lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from .infoton import InfitonField, create_infoton
from .lattice import U1LatticGauge


@dataclass
class SCPNLattice:
    """SCPN hierarchy mapped to a gauge field and infoton.

    Attributes
    ----------
    gauge
        U(1) gauge field defined by the SCPN coupling topology.
    infoton
        Matter field living on the lattice sites.
    K
        Coupling matrix used to construct the topology.
    omega
        Natural frequencies associated with the sites.
    n_layers
        Number of mapped SCPN layers.

    """

    gauge: U1LatticGauge
    infoton: InfitonField
    K: NDArray[np.float64]
    omega: NDArray[np.float64]
    n_layers: int


def scpn_to_lattice(
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    beta: float = 1.0,
    mass_sq: float = 1.0,
    quartic_coupling: float = 0.1,
    gauge_coupling: float = 1.0,
    seed: int | None = None,
) -> SCPNLattice:
    """Build a lattice gauge + infoton from SCPN K_nm coupling.

    The K_nm matrix defines the graph topology. Link weights are
    proportional to K_nm values, so stronger SCPN coupling produces
    stronger gauge links and more correlated Ψ-field dynamics.

    Parameters
    ----------
    K
        Coupling matrix. The Paper 27 16-layer matrix is used by default.
    omega
        Natural frequencies. The matching prefix of ``OMEGA_N_16`` is used by
        default.
    beta
        Gauge inverse coupling ``1 / g**2``; larger values favour order.
    mass_sq
        Infoton mass squared; positive values select the symmetric phase.
    quartic_coupling
        Infoton quartic ``lambda * |phi|**4`` coupling.
    gauge_coupling
        Gauge-matter coupling.
    seed
        Seed shared by gauge-link and infoton initialization.

    Returns
    -------
    SCPNLattice
        Coupled gauge and matter fields with their SCPN inputs.

    """
    if K is None:
        K = build_knm_paper27()
    if omega is None:
        omega = OMEGA_N_16[: K.shape[0]].copy()

    n = K.shape[0]
    gauge = U1LatticGauge(K, beta=beta, seed=seed)
    infoton = create_infoton(
        n_sites=n,
        mass_sq=mass_sq,
        coupling=quartic_coupling,
        gauge_coupling=gauge_coupling,
        seed=seed,
    )

    return SCPNLattice(
        gauge=gauge,
        infoton=infoton,
        K=K,
        omega=omega,
        n_layers=n,
    )
