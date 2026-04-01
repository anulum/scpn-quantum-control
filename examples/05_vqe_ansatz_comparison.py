# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Compare three VQE ansatze on the 4-oscillator Kuramoto Hamiltonian.

Prints energy, parameter count, and iteration count for each ansatz.
No QPU needed — runs entirely on statevector simulation.
"""

from scpn_quantum_control.hardware.classical import classical_exact_diag
from scpn_quantum_control.phase.ansatz_bench import run_ansatz_benchmark


def main():
    n = 4
    exact = classical_exact_diag(n)

    print(f"Exact ground energy ({n}q): {exact['ground_energy']:.6f}\n")

    results = run_ansatz_benchmark(n_qubits=n, maxiter=200, reps=2)

    print(f"{'Ansatz':<20} {'Params':>6} {'Energy':>10} {'Gap':>10} {'Evals':>6}")
    print("-" * 56)
    for r in results:
        gap = abs(r["energy"] - exact["ground_energy"])
        print(
            f"{r['ansatz']:<20} {r['n_params']:>6} {r['energy']:>10.4f} {gap:>10.4f} {r['n_evals']:>6}"
        )


if __name__ == "__main__":
    main()
