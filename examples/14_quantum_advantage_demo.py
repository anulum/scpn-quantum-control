# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum advantage scaling demo: classical vs quantum timing."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark


def main() -> None:
    print("Quantum Advantage Scaling Demo")
    print("=" * 50)

    results = run_scaling_benchmark(sizes=[4, 6, 8, 10], t_max=0.3, dt=0.1)

    print(f"{'N':>4} {'Classical (ms)':>15} {'Quantum (ms)':>15} {'Ratio':>8}")
    print("-" * 46)
    for r in results:
        ratio = r.t_classical_ms / r.t_quantum_ms if r.t_quantum_ms > 0 else float("inf")
        t_c = f"{r.t_classical_ms:.1f}" if np.isfinite(r.t_classical_ms) else "INF"
        print(f"{r.n_qubits:4d} {t_c:>15} {r.t_quantum_ms:15.1f} {ratio:8.1f}")

    cross = results[0].crossover_predicted
    if cross:
        print(f"\nPredicted crossover: {cross} qubits")
    else:
        print("\nCrossover: insufficient data for extrapolation")


if __name__ == "__main__":
    main()
