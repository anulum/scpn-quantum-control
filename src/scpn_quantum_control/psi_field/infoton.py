# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Infoton Field (Scalar QED on Lattice)
"""Infoton: scalar field coupled to the U(1) gauge via covariant derivative.

The SCPN Ψ-field carries the infoton boson — a complex scalar field φ
living on lattice sites, minimally coupled to the U(1) gauge field A_ij.

Lattice scalar QED action:
    S_matter = Σ_⟨ij⟩ |D_ij φ|² + Σ_i (m² |φ_i|² + λ |φ_i|⁴)

where the gauge-covariant lattice derivative is:
    D_ij φ = φ_j − exp(igA_ij) φ_i

This preserves local U(1) gauge invariance:
    φ_i → exp(iα_i) φ_i,  A_ij → A_ij + α_i − α_j

Ref:
    - Creutz, "Quarks, Gluons and Lattices" (1983), Ch. 15
    - Smit, "Introduction to Quantum Fields on a Lattice" (2002)
    - SCPN Paper 27: Ψ-field and infoton concept (Šotek, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import U1LatticGauge

try:
    from scpn_quantum_engine import gauge_covariant_kinetic_rust as _kin_rust

    _HAS_RUST_GAUGE = True
except ImportError:
    _HAS_RUST_GAUGE = False


@dataclass
class InfitonField:
    """Complex scalar field on lattice sites.

    φ_i = re_i + i × im_i, stored as complex array.
    """

    values: np.ndarray  # complex array, shape (n_sites,)
    mass_sq: float  # m²
    coupling: float  # λ (quartic self-coupling)
    gauge_coupling: float  # g (gauge-matter coupling)

    @property
    def n_sites(self) -> int:
        return len(self.values)

    def density(self) -> np.ndarray:
        """Charge density |φ_i|² at each site."""
        return np.asarray(np.abs(self.values) ** 2)

    def total_charge(self) -> float:
        """Total scalar charge Σ |φ_i|²."""
        return float(np.sum(self.density()))

    def potential_energy(self) -> float:
        """V = Σ_i (m² |φ_i|² + λ |φ_i|⁴)."""
        rho = self.density()
        return float(np.sum(self.mass_sq * rho + self.coupling * rho**2))


def gauge_covariant_kinetic(
    field: InfitonField,
    gauge: U1LatticGauge,
) -> float:
    """Gauge-covariant kinetic energy (hopping term).

    T = Σ_⟨ij⟩ (|φ_i|² + |φ_j|² − 2 Re(φ_i* × U_ij × φ_j))

    where U_ij = exp(igA_ij). This is gauge-invariant under:
        φ_i → exp(iα_i) φ_i, A_ij → A_ij + (α_i − α_j)/g

    Derivation: expand |φ_i − U_ji φ_j|² using U_ji = U_ij*.
    The hopping term Re(φ_i* U_ij φ_j) transforms as:
        φ_i'* U_ij' φ_j' = φ_i* e^{-iα_i} × e^{iα_i} U_ij e^{-iα_j} × e^{iα_j} φ_j
                          = φ_i* U_ij φ_j  ✓

    Ref: Rothe, "Lattice Gauge Theories" (2012), Eq. (3.15)
    """
    if _HAS_RUST_GAUGE and gauge.n_edges > 0:
        phi = field.values
        edges_arr = np.array(gauge.edges, dtype=np.int64)
        return float(
            _kin_rust(
                phi.real.copy(),
                phi.imag.copy(),
                gauge.links,
                edges_arr,
                field.gauge_coupling,
            )
        )

    phi = field.values
    g = field.gauge_coupling
    total = 0.0

    for idx, (i, j) in enumerate(gauge.edges):
        a_ij = gauge.links[idx]
        u_ij = np.exp(1j * g * a_ij)
        hopping = phi[i].conjugate() * u_ij * phi[j]
        total += float(np.abs(phi[i]) ** 2 + np.abs(phi[j]) ** 2 - 2.0 * hopping.real)

    return total


def matter_action(
    field: InfitonField,
    gauge: U1LatticGauge,
) -> float:
    """Total matter action S_matter = T + V."""
    return gauge_covariant_kinetic(field, gauge) + field.potential_energy()


def create_infoton(
    n_sites: int,
    mass_sq: float = 1.0,
    coupling: float = 0.1,
    gauge_coupling: float = 1.0,
    amplitude: float = 0.1,
    seed: int | None = None,
) -> InfitonField:
    """Create an infoton field with random initial values.

    Small random perturbation around zero (symmetric phase).
    """
    rng = np.random.default_rng(seed)
    values = amplitude * (rng.standard_normal(n_sites) + 1j * rng.standard_normal(n_sites))
    return InfitonField(
        values=values,
        mass_sq=mass_sq,
        coupling=coupling,
        gauge_coupling=gauge_coupling,
    )
