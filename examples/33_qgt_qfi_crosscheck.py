# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QGT/QFI cross-check demo
"""Cross-validate spectral QFI against the quantum geometric tensor route.

Computes the coupling-parameter QFI of the XY ground state on a small ring by
two independent physics routes — the exact spectral sum rule and four times
the Fubini-Study metric from finite-difference ground states — and reports
their agreement plus the Berry curvature (zero for this time-reversal
symmetric ground state).
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis import crosscheck_qfi_geometric


def main() -> None:
    print("QGT/QFI cross-check demo")
    print("=" * 50)

    n = 4
    K = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = 0.7
    omega = np.array([0.15, -0.1, 0.05, -0.02])

    result = crosscheck_qfi_geometric(K, omega)

    print(f"system: {n}-site XY ring, {len(result.coupling_pairs)} coupling parameters")
    print(f"spectral QFI diagonal:  {np.round(np.diag(result.spectral.qfi_matrix), 6)}")
    print(f"geometric QFI diagonal: {np.round(np.diag(result.qfi_geometric), 6)}")
    print(f"max |difference|:       {result.max_abs_difference:.3e}")
    print(f"max relative diff:      {result.max_rel_difference:.3e}")
    print(f"max |Berry curvature|:  {result.max_abs_berry_curvature:.3e}")
    print(f"routes agree:           {result.agrees}")
    print("claim boundary: exact-diagonalisation cross-check on a small system;")
    print("no hardware, scaling, or metrological-advantage claim.")


if __name__ == "__main__":
    main()
