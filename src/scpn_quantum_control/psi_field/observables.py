# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lattice Gauge Observables
"""Gauge-invariant observables for the U(1) Ψ-field lattice.

Implements standard lattice gauge theory observables adapted for
the SCPN graph topology:
    - Polyakov loop: order parameter for confinement
    - Topological charge: winding number of gauge field
    - String tension: extracted from Wilson loop area law

These complement the quantum Wilson loop measurements in gauge/
by providing classical lattice Monte Carlo observables on the
same SCPN topology.
"""

from __future__ import annotations

import numpy as np

from .lattice import U1LatticGauge

try:
    from scpn_quantum_engine import topological_charge_rust as _topo_rust

    _HAS_RUST_GAUGE = True
except ImportError:
    _HAS_RUST_GAUGE = False


def polyakov_loop(gauge: U1LatticGauge, path: list[int]) -> complex:
    """Polyakov loop along a specified path through the lattice.

    P = Π_{consecutive pairs (i,j) in path} U_ij = exp(iΣ A_ij)

    For SCPN: a path through layers L1→L2→...→L16 measures the
    total phase accumulated across the hierarchy.

    Args:
        gauge: U(1) lattice gauge field
        path: ordered list of site indices forming the path
    """
    phase = 0.0
    for idx in range(len(path) - 1):
        i, j = path[idx], path[idx + 1]
        phase += gauge._edge_link(i, j)
    return complex(np.exp(1j * phase))


def topological_charge(gauge: U1LatticGauge) -> float:
    """Topological charge Q = (1/2π) Σ_plaq θ_plaq.

    For compact U(1), the topological charge counts the net number
    of vortices (winding number). θ_plaq is the plaquette angle
    wrapped to [−π, π).

    On the SCPN graph, vortices represent topological defects in
    the Ψ-field — points where the phase winds by ±2π.
    Uses Rust when available.
    """
    if _HAS_RUST_GAUGE and len(gauge.plaquettes) > 0:
        return float(
            _topo_rust(
                gauge.links,
                gauge._tri_flat,
                gauge._tri_signs,
                len(gauge.plaquettes),
            )
        )

    q = 0.0
    for plaq in gauge.plaquettes:
        i, j = plaq[0]
        _, k = plaq[1]
        phase = gauge._edge_link(i, j) + gauge._edge_link(j, k) - gauge._edge_link(i, k)
        # Wrap to [−π, π)
        wrapped = (phase + np.pi) % (2 * np.pi) - np.pi
        q += wrapped
    return q / (2 * np.pi)


def string_tension_from_wilson(
    gauge: U1LatticGauge,
    n_measurements: int = 100,
) -> float | None:
    """Extract string tension from plaquette expectation.

    σ = −log(⟨Re(U_plaq)⟩) / a²

    where a is the lattice spacing (set to 1). Returns None if
    no plaquettes exist or ⟨Re(U_plaq)⟩ ≤ 0.

    Caveat: this is the lowest-order estimate. For precision,
    use Creutz ratios from Wilson loops of different sizes.
    """
    result = gauge.measure_plaquettes()
    if result.n_plaquettes == 0 or result.mean_plaquette <= 0:
        return None
    return float(-np.log(result.mean_plaquette))


def average_link(gauge: U1LatticGauge) -> complex:
    """Average link variable ⟨U⟩ = (1/N_links) Σ exp(iA_ij).

    Not gauge-invariant, but useful as a diagnostic. In the ordered
    phase (large β), ⟨U⟩ → 1. In the disordered phase, ⟨U⟩ → 0.
    """
    if gauge.n_edges == 0:
        return complex(0.0)
    return complex(np.mean(np.exp(1j * gauge.links)))
