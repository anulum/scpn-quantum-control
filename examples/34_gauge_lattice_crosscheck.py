# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Gauge/lattice cross-check demo
"""Joint confinement probe: quantum Wilson loops vs classical U(1) lattice MC.

Runs both confinement routes on one small SCPN coupling topology — the
quantum ground-state Wilson-loop analysis and a Hybrid-Monte-Carlo-sampled
compact U(1) lattice ensemble on the identical graph — and prints the two
string tensions and the lattice observables side by side.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.gauge import crosscheck_confinement_on_lattice


def main() -> None:
    print("Gauge/lattice confinement cross-check demo")
    print("=" * 50)

    n = 4
    K = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = 0.7
    K[0, 2] = K[2, 0] = 0.4
    omega = np.array([0.15, -0.1, 0.05, -0.02])

    result = crosscheck_confinement_on_lattice(K, omega, beta=2.0, n_thermalisation=150, seed=7)

    print(f"topology: {n}-site ring + chord, {result.lattice_n_plaquettes} plaquettes")
    print(f"quantum string tension:  {result.quantum.string_tension}")
    print(f"quantum confined:        {result.quantum.is_confined}")
    print(f"lattice string tension:  {result.lattice_string_tension}")
    print(f"lattice mean plaquette:  {result.lattice_mean_plaquette:.4f}")
    print(f"topological charge:      {result.lattice_topological_charge:.4f}")
    print(f"HMC acceptance rate:     {result.hmc_acceptance_rate:.2f}")
    print("claim boundary: the two routes probe different regimes (ground state")
    print("vs thermal ensemble) on a shared topology; side-by-side report only,")
    print("no equality assertion, no hardware or scaling claim.")


if __name__ == "__main__":
    main()
