# Larger-System Submission Extension Readiness

- Generated: `2026-05-20T19:39:03+00:00`
- Backend: `ibm_fez` (`live_ibm_backend`)
- Hardware submitted: `False`
- Ready QPU estimate, one backend: `1152.0s (19.20 min)`
- Ready QPU estimate, Fez+Marrakesh pair: `2304.0s (38.40 min)`
- IBM usage probe: `True`
- IBM seconds remaining: `2440.0`
- One-backend live budget fit: `True`
- Two-backend live budget fit: `True`
- Remaining after one backend: `1288.0s (21.47 min)`
- Remaining after two backends: `136.0s (2.27 min)`

| Candidate | Paper | n | Status | Circuits | QPU estimate | Representative depth | 2Q gates |
|---|---|---:|---|---:|---:|---:|---:|
| `phase3_reduced_pauli_n6` | `submission_005_phase3_reduced_pauli_entanglement` | 6 | ready_for_qpu_preregistration | 388 | 213.4s (3.56 min) | 406 | 200 |
| `phase3_reduced_pauli_n8` | `submission_005_phase3_reduced_pauli_entanglement` | 8 | ready_for_qpu_preregistration | 580 | 319.0s (5.32 min) | 440 | 280 |
| `fim_replication_zne_n6` | `submission_004_scpn_fim_hamiltonian` | 6 | ready_for_qpu_preregistration | 160 | 88.0s (1.47 min) | 762 | 362 |
| `fim_replication_zne_n8` | `submission_004_scpn_fim_hamiltonian` | 8 | ready_for_qpu_preregistration | 352 | 193.6s (3.23 min) | 902 | 684 |
| `methods_ansatz_energy_n6` | `submission_003_rust_vqe_methods` | 6 | ready_for_qpu_preregistration | 73 | 73.0s (1.22 min) | 91 | 20 |
| `methods_ansatz_energy_n8` | `submission_003_rust_vqe_methods` | 8 | ready_for_qpu_preregistration | 265 | 265.0s (4.42 min) | 125 | 28 |
| `methods_gpu_scaling_n16` | `submission_003_rust_vqe_methods` | 16 | ready_for_gpu_execution | 0 | 0.0s (0.00 min) | 261 | 60 |
| `methods_gpu_scaling_n20` | `submission_003_rust_vqe_methods` | 20 | ready_for_gpu_execution | 0 | 0.0s (0.00 min) | 329 | 76 |

## Recommended Order

1. `phase3_reduced_pauli_n6`
2. `phase3_reduced_pauli_n8`
3. `fim_replication_zne_n6`
4. `fim_replication_zne_n8`
5. `methods_ansatz_energy_n6`
6. `methods_ansatz_energy_n8`
7. `methods_gpu_scaling_n16`
8. `methods_gpu_scaling_n20`

## Claim Boundary

Use n=6/n=8 QPU lanes only if live backend transpilation remains below depth and two-qubit gates. Use n=16/n=20 as local GPU/classical scaling evidence for methods, not as hardware evidence.
