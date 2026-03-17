# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Trapped-ion backend demo: noise model + all-to-all transpilation."""

from __future__ import annotations

from qiskit_aer import AerSimulator

from scpn_quantum_control import OMEGA_N_16, QuantumKuramotoSolver, build_knm_paper27
from scpn_quantum_control.hardware.trapped_ion import (
    transpile_for_trapped_ion,
    trapped_ion_noise_model,
)


def main() -> None:
    K = build_knm_paper27(L=4)
    solver = QuantumKuramotoSolver(4, K, OMEGA_N_16[:4])
    qc = solver.evolve(time=0.3, trotter_steps=2)

    print("Trapped-Ion Backend Demo")
    print("=" * 50)
    print(f"Original: {qc.size()} gates, depth {qc.depth()}")

    qc_ion = transpile_for_trapped_ion(qc)
    print(f"Transpiled: {qc_ion.size()} gates, depth {qc_ion.depth()}")
    print(f"Basis gates: {sorted(set(g.operation.name for g in qc_ion))}")

    noise = trapped_ion_noise_model()
    sim = AerSimulator(noise_model=noise)
    qc_meas = qc_ion.copy()
    qc_meas.measure_all()
    result = sim.run(qc_meas, shots=4096).result()
    counts = result.get_counts()
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]
    print("\nTop 5 outcomes (noisy trapped-ion sim):")
    for state, count in top:
        print(f"  |{state}> : {count} ({count / 4096 * 100:.1f}%)")


if __name__ == "__main__":
    main()
