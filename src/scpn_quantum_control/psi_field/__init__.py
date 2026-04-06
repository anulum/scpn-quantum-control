# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ψ-field Lattice Gauge Theory
"""U(1) lattice gauge simulator for the SCPN Ψ-field.

The SCPN defines a Ψ-field as a compact U(1) gauge field carrying the
infoton boson. This package implements the lattice gauge dynamics on
arbitrary graph topologies (not restricted to hypercubic lattices),
enabling simulation on the actual SCPN 15+1 layer hierarchy.

Modules:
    lattice — U(1) gauge field on arbitrary graphs with HMC update
    infoton — scalar field coupled to gauge (lattice scalar QED)
    scpn_mapping — SCPN layer hierarchy to lattice topology
    observables — Polyakov loop, topological charge, string tension
"""

from .infoton import InfitonField, gauge_covariant_kinetic
from .lattice import PlaquetteResult, U1LatticGauge, hmc_update
from .observables import (
    polyakov_loop,
    string_tension_from_wilson,
    topological_charge,
)
from .scpn_mapping import SCPNLattice, scpn_to_lattice

__all__ = [
    "U1LatticGauge",
    "PlaquetteResult",
    "hmc_update",
    "InfitonField",
    "gauge_covariant_kinetic",
    "SCPNLattice",
    "scpn_to_lattice",
    "polyakov_loop",
    "topological_charge",
    "string_tension_from_wilson",
]
