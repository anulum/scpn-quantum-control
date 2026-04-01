# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Demonstration of Persistent Homology on Quantum Simulation Data.

Extracts correlation distance matrices from simulated quantum measurement
counts and computes persistent homology Betti intervals to detect the
topological signatures of the synchronization transition.
"""

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis.quantum_persistent_homology import (
    _RIPSER_AVAILABLE,
    quantum_persistent_homology,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.fast_classical import fast_sparse_evolution


def _simulate_measurement_counts(psi: np.ndarray, n: int, shots: int = 1000) -> tuple[dict, dict]:
    sv = Statevector(psi)

    # Measure in X basis
    import qiskit.quantum_info as qi

    sv_x = sv.copy()
    for q in range(n):
        sv_x = sv_x.evolve(qi.Operator.from_label("H"), [q])
    x_counts = sv_x.sample_counts(shots)

    # Measure in Y basis
    sv_y = sv.copy()
    for q in range(n):
        sv_y = sv_y.evolve(qi.Operator.from_label("Sdg"), [q])
        sv_y = sv_y.evolve(qi.Operator.from_label("H"), [q])
    y_counts = sv_y.sample_counts(shots)

    return x_counts, y_counts


def main():
    if not _RIPSER_AVAILABLE:
        print("Ripser not available. Please run: pip install ripser")
        return

    n = 6
    K_topo = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    print(f"Simulating {n}-qubit Kuramoto system + Quantum Persistent Homology...")
    print(f"{'K_base':<10} {'p_H1':<12} {'1-Cycles':<10}")
    print("-" * 35)

    for K_base in np.linspace(0.0, 3.0, 6):
        K = K_topo * K_base

        # Evolve system using high-performance sparse engine
        res = fast_sparse_evolution(K, omega, t_total=2.0, n_steps=1)
        psi_final = res["final_state"]

        # Simulate noisy measurements
        x_counts, y_counts = _simulate_measurement_counts(psi_final, n, shots=5000)

        # Run Quantum Persistent Homology pipeline
        ph_res = quantum_persistent_homology(
            x_counts, y_counts, n_qubits=n, persistence_threshold=0.1
        )

        print(f"{K_base:<10.2f} {ph_res.p_h1:<12.4f} {ph_res.n_h1_persistent:<10}")


if __name__ == "__main__":
    main()
