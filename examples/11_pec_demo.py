# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""PEC error mitigation demo: quasi-probability decomposition + Monte Carlo."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control import OMEGA_N_16, QuantumKuramotoSolver, build_knm_paper27
from scpn_quantum_control.mitigation.pec import pec_sample


def main() -> None:
    K = build_knm_paper27(L=2)
    solver = QuantumKuramotoSolver(2, K, OMEGA_N_16[:2])
    qc = solver.evolve(time=0.3, trotter_steps=2)

    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(qc)
    exact = 2 * float(sv.probabilities([0])[0]) - 1

    print("PEC Error Mitigation Demo")
    print("=" * 50)
    print(f"Circuit: {qc.size()} gates, depth {qc.depth()}")
    print(f"Exact <Z_0> = {exact:.6f}\n")

    for p in [0.005, 0.01, 0.05]:
        result = pec_sample(qc, p, 2000, observable_qubit=0, rng=np.random.default_rng(42))
        err = abs(result.mitigated_value - exact)
        print(
            f"p={p:.3f}: PEC={result.mitigated_value:+.4f}, error={err:.4f}, overhead={result.overhead:.1f}"
        )


if __name__ == "__main__":
    main()
