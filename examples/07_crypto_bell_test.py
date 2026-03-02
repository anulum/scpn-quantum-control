"""Bell inequality test on the K_nm ground state.

Builds the 4-oscillator XY ground state via VQE, then evaluates
the CHSH S-parameter for nearest-neighbour qubit pairs.
S > 2 certifies entanglement.  No QPU needed.
"""

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.crypto.entanglement_qkd import bell_inequality_test
from scpn_quantum_control.phase.phase_vqe import PhaseVQE

n = 4
K = build_knm_paper27(L=n)
omega = OMEGA_N_16[:n]

print(f"=== Bell test on {n}-qubit K_nm ground state ===\n")

vqe = PhaseVQE(K, omega, ansatz_reps=2)
sol = vqe.solve(maxiter=200, seed=0)
print(f"VQE energy:  {sol['vqe_energy']:.6f}")
print(f"Exact energy: {sol['exact_energy']:.6f}")
print(f"Gap: {sol['energy_gap']:.6f}\n")

H = knm_to_hamiltonian(K, omega)
ansatz = knm_to_ansatz(K, reps=2)
bound = ansatz.assign_parameters(sol["optimal_params"])
sv = Statevector.from_instruction(bound)

print("CHSH S-parameter for nearest-neighbour pairs:")
for a, b in [(0, 1), (1, 2), (2, 3)]:
    result = bell_inequality_test(sv, qubit_a=a, qubit_b=b, n_total=n)
    status = "ENTANGLED" if result["S"] > 2.0 else "classical"
    print(f"  qubits ({a},{b}): S = {result['S']:.4f}  [{status}]")
