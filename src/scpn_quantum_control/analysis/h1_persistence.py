# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — H1 Persistence
"""H1 persistence at the BKT transition: does p_h1 = 0.72 emerge?

The persistent homology H1 measures topological loops (vortices)
in the oscillator phase configuration. At the BKT transition:

    - Below T_BKT: few persistent H1 features (bound vortex pairs)
    - At T_BKT: H1 persistence peaks (vortex pair unbinding onset)
    - Above T_BKT: H1 saturates (free vortices everywhere)

The fraction of H1-persistent cycles at T_BKT is the quantum TCBO's
p_h1. If this fraction at the critical coupling K_c equals 0.72,
the consciousness gate threshold is derived from BKT physics.

Method:
    1. Scan K from weak to strong coupling
    2. At each K, measure vortex density (H1 proxy) from quantum state
    3. Identify K_c as the coupling where vortex density changes fastest
    4. Report p_h1 at K_c
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import build_knm_paper27
from ..gauge.vortex_detector import measure_vortex_density


@dataclass
class H1PersistenceResult:
    """H1 persistence scan result."""

    k_critical: float  # K where d(vortex_density)/dK is maximum
    p_h1_at_critical: float  # vortex density at K_c
    k_values: np.ndarray
    vortex_densities: np.ndarray
    derivative: np.ndarray
    deviation_from_target: float  # |p_h1 - 0.72|


def scan_h1_persistence(
    omega: np.ndarray,
    k_range: tuple[float, float] = (0.01, 3.0),
    n_points: int = 25,
) -> H1PersistenceResult:
    """Scan vortex density (H1 proxy) across coupling strength.

    The critical coupling K_c is where the vortex density changes
    most rapidly (maximum |d(rho_v)/dK|).
    """
    n = len(omega)
    k_values = np.linspace(k_range[0], k_range[1], n_points)
    densities = np.zeros(n_points)

    for i, kb in enumerate(k_values):
        K = build_knm_paper27(L=n, K_base=float(kb))
        vr = measure_vortex_density(K, omega)
        densities[i] = vr.vortex_density

    # Numerical derivative
    dk = k_values[1] - k_values[0]
    deriv = np.gradient(densities, dk)

    # K_c = argmax |d(rho_v)/dK|
    peak_idx = int(np.argmax(np.abs(deriv)))
    k_c = float(k_values[peak_idx])
    p_h1 = float(densities[peak_idx])
    deviation = abs(p_h1 - 0.72)

    return H1PersistenceResult(
        k_critical=k_c,
        p_h1_at_critical=p_h1,
        k_values=k_values,
        vortex_densities=densities,
        derivative=deriv,
        deviation_from_target=deviation,
    )
