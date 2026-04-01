# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Demonstration of the Synchronization Witness Operator.

This script constructs the formal quantum synchronization witness operator W_sync
and evaluates its expectation value across the classical Kuramoto transition.
A negative expectation value ⟨W_sync⟩ < 0 indicates the system has entered
the synchronized phase.
"""

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.analysis.sync_witness import build_correlation_witness_operator
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.fast_classical import fast_sparse_evolution


def main():
    n = 4
    K_topo = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]

    # The threshold R_c separating synchronized from incoherent is roughly 0.5
    threshold = 0.5

    print(f"Building {n}-qubit Synchronization Witness Operator (threshold = {threshold})...")
    W_sync = build_correlation_witness_operator(n, threshold=threshold)

    print("Sweeping coupling strength K_base...")
    print(f"{'K_base':<10} {'⟨W_sync⟩':<12} {'Synchronized?'}")
    print("-" * 40)

    for K_base in np.linspace(0.0, 3.0, 10):
        K = K_topo * K_base

        # Evolve system using high-performance sparse engine
        res = fast_sparse_evolution(K, omega, t_total=2.0, n_steps=1)
        psi_final = res["final_state"]
        sv = Statevector(psi_final)

        # Evaluate witness operator expectation value
        w_val = sv.expectation_value(W_sync).real

        is_synced = w_val < 0
        print(f"{K_base:<10.2f} {w_val:<12.4f} {is_synced}")


if __name__ == "__main__":
    main()
