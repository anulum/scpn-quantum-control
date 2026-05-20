# Larger-System Submission Extension Runbook

Generated: 2026-05-20.

This runbook prepares the larger-system extension lanes requested for the
submission paper stack and records the hardware jobs submitted from that
preparation pass.

## Budget Position

- Live IBM allocation remaining at probe time: 3160 s.
- Live IBM allocation after methods and FIM submissions: 2768 s.
- Live IBM allocation after methods, FIM, and Phase 3 submissions: 2652 s.
- Live IBM allocation after final post-submission probe: 2440 s.
- Conservative one-backend estimate for all n=6/n=8 QPU lanes: 1152 s.
- Conservative Fez+Marrakesh estimate for all n=6/n=8 QPU lanes: 2304 s.
- Estimated remaining margin after both backends: 856 s.
- Estimated full-stack margin after submitted methods/FIM lanes: 464 s.
- Estimated full-stack margin after submitted methods/FIM/Phase 3 lanes: 348 s.
- Estimated full-stack margin at final post-submission probe: 136 s.
- n=16 and n=20 are kept as local construction/transpilation scaling evidence,
  not QPU evidence.

The canonical readiness artifacts are:

- `data/large_system_submission_extensions/large_system_submission_extension_readiness_20260520T193857Z.json`
- `data/large_system_submission_extensions/large_system_submission_extension_readiness_20260520T193903Z.json`
- `data/large_system_submission_extensions/large_system_submission_extension_readiness_20260520T193909Z.json`
- `docs/campaigns/large_system_submission_extension_readiness_20260520T193857Z.md`
- `docs/campaigns/large_system_submission_extension_readiness_20260520T193903Z.md`
- `docs/campaigns/large_system_submission_extension_readiness_20260520T193909Z.md`

## Prepared Candidate Order

1. `phase3_reduced_pauli_n6`
2. `phase3_reduced_pauli_n8`
3. `fim_replication_zne_n6`
4. `fim_replication_zne_n8`
5. `methods_ansatz_energy_n6`
6. `methods_ansatz_energy_n8`
7. `methods_gpu_scaling_n16`
8. `methods_gpu_scaling_n20`

## Methods Paper Lanes

The methods-paper n=6 and n=8 IBM readiness lanes are prepared on Fez and
Marrakesh. These are currently the direct submit-ready larger-system lanes.

Prepared readiness artifacts:

- `data/rust_vqe_methods/ansatz_ibm_validation_readiness_ibm_fez_20260520T190928Z.json`
- `data/rust_vqe_methods/ansatz_ibm_validation_readiness_ibm_fez_20260520T190947Z.json`
- `data/rust_vqe_methods/ansatz_ibm_validation_readiness_ibm_marrakesh_20260520T191002Z.json`
- `data/rust_vqe_methods/ansatz_ibm_validation_readiness_ibm_marrakesh_20260520T191021Z.json`

Submission commands, if the budget gate is still acceptable at execution time:

```bash
.venv-linux/bin/python scripts/submit_methods_ansatz_ibm_validation.py --backend ibm_fez --n-qubits 6 --physical-qubits 21 22 23 24 25 26 --max-qpu-seconds 3600 --experiment-id rust_vqe_methods_ansatz_validation_n6_fez_2026-05-20 --submit --confirm-budget
.venv-linux/bin/python scripts/submit_methods_ansatz_ibm_validation.py --backend ibm_fez --n-qubits 8 --physical-qubits 21 22 23 24 25 26 27 28 --max-qpu-seconds 3600 --experiment-id rust_vqe_methods_ansatz_validation_n8_fez_2026-05-20 --submit --confirm-budget
.venv-linux/bin/python scripts/submit_methods_ansatz_ibm_validation.py --backend ibm_marrakesh --n-qubits 6 --physical-qubits 2 3 4 5 6 7 --max-qpu-seconds 3600 --experiment-id rust_vqe_methods_ansatz_validation_n6_marrakesh_2026-05-20 --submit --confirm-budget
.venv-linux/bin/python scripts/submit_methods_ansatz_ibm_validation.py --backend ibm_marrakesh --n-qubits 8 --physical-qubits 1 2 3 4 5 6 7 8 --max-qpu-seconds 3600 --experiment-id rust_vqe_methods_ansatz_validation_n8_marrakesh_2026-05-20 --submit --confirm-budget
```

Submitted methods jobs:

| Experiment | Backend | n | Job ID | Submission artifact |
|---|---|---:|---|---|
| `rust_vqe_methods_ansatz_validation_n6_fez_2026-05-20` | `ibm_fez` | 6 | `d870islg7okc73elnmmg` | `data/rust_vqe_methods/ansatz_ibm_validation_submission_ibm_fez_20260520T192017Z.json` |
| `rust_vqe_methods_ansatz_validation_n8_fez_2026-05-20` | `ibm_fez` | 8 | `d870j15g7okc73elnms0` | `data/rust_vqe_methods/ansatz_ibm_validation_submission_ibm_fez_20260520T192034Z.json` |
| `rust_vqe_methods_ansatz_validation_n6_marrakesh_2026-05-20` | `ibm_marrakesh` | 6 | `d870j52s46sc73f7j40g` | `data/rust_vqe_methods/ansatz_ibm_validation_submission_ibm_marrakesh_20260520T192050Z.json` |
| `rust_vqe_methods_ansatz_validation_n8_marrakesh_2026-05-20` | `ibm_marrakesh` | 8 | `d870j99789is73909fpg` | `data/rust_vqe_methods/ansatz_ibm_validation_submission_ibm_marrakesh_20260520T192107Z.json` |

## FIM Hamiltonian Lanes

The FIM n=6 and n=8 Fez/Marrakesh replication/ZNE lanes were prepared and
submitted after generalising `scripts/submit_fim_replication_zne_ibm.py` beyond
the original n=4 execution path.

Submitted FIM jobs:

| Experiment | Backend | n | Job ID | Submission artifact |
|---|---|---:|---|---|
| `scpn_fim_replication_zne_n6_fez_2026-05-20` | `ibm_fez` | 6 | `d870ld1789is73909hrg` | `data/scpn_fim_hamiltonian/fim_replication_zne_submission_ibm_fez_20260520T192537Z.json` |
| `scpn_fim_replication_zne_n8_fez_2026-05-20` | `ibm_fez` | 8 | `d870lhqs46sc73f7j6ag` | `data/scpn_fim_hamiltonian/fim_replication_zne_submission_ibm_fez_20260520T192555Z.json` |
| `scpn_fim_replication_zne_n6_marrakesh_2026-05-20` | `ibm_marrakesh` | 6 | `d870llp789is73909i60` | `data/scpn_fim_hamiltonian/fim_replication_zne_submission_ibm_marrakesh_20260520T192612Z.json` |
| `scpn_fim_replication_zne_n8_marrakesh_2026-05-20` | `ibm_marrakesh` | 8 | `d870lq9789is73909id0` | `data/scpn_fim_hamiltonian/fim_replication_zne_submission_ibm_marrakesh_20260520T192630Z.json` |

## Methods n=16/n=20 Local Scaling

The n=16/n=20 scaling evidence was generated as local construction and
transpilation evidence with error bars:

- `data/rust_vqe_methods/vqe_convergence_methods_2026-05-20-large-system-extension.json`
- `data/rust_vqe_methods/vqe_convergence_aggregate_2026-05-20-large-system-extension.csv`
- `data/rust_vqe_methods/ansatz_scaling_error_bars_2026-05-20-large-system-extension.csv`

This evidence must be described as local scaling only. It must not be described
as IBM hardware evidence.

## Phase 3 and FIM Larger-n Runner Status

The n=6/n=8 Phase 3 and FIM lanes are resource-ready by live Fez/Marrakesh
transpilation probes. FIM is handled by the parameterised
`scripts/submit_fim_replication_zne_ibm.py`; Phase 3 larger-system lanes are
handled by dedicated `scripts/submit_phase3_large_system_ibm.py`.

Submitted Phase 3 larger-system jobs:

| Experiment | Backend | n | Job ID | Submission artifact |
|---|---|---:|---|---|
| `phase3_large_system_n6_fez_2026-05-20` | `ibm_fez` | 6 | `d870qa8p0eas73dlj6kg` | `data/phase3_entanglement_tomography/phase3_large_system_submission_ibm_fez_20260520T193603Z.json` |
| `phase3_large_system_n8_fez_2026-05-20` | `ibm_fez` | 8 | `d870qf9789is73909nu0` | `data/phase3_entanglement_tomography/phase3_large_system_submission_ibm_fez_20260520T193622Z.json` |
| `phase3_large_system_n6_marrakesh_2026-05-20` | `ibm_marrakesh` | 6 | `d870qj9789is73909o2g` | `data/phase3_entanglement_tomography/phase3_large_system_submission_ibm_marrakesh_20260520T193640Z.json` |
| `phase3_large_system_n8_marrakesh_2026-05-20` | `ibm_marrakesh` | 8 | `d870qo0p0eas73dlj75g` | `data/phase3_entanglement_tomography/phase3_large_system_submission_ibm_marrakesh_20260520T193658Z.json` |

## Retrieval and Reduction Status

All twelve submitted IBM jobs reported `DONE` in the bounded post-submission
status probe and were retrieved.

### Methods hardware ansatz validation

| Backend | n | Raw-count artefact | Analysis artefact | Snapshot |
|---|---:|---|---|---|
| `ibm_fez` | 6 | `data/rust_vqe_methods/ansatz_ibm_validation_raw_counts_ibm_fez_20260520T195028Z.json` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_fez_20260520T195028Z.json` | `efficient_su2` has the lowest mitigated energy in this lane. |
| `ibm_fez` | 8 | `data/rust_vqe_methods/ansatz_ibm_validation_raw_counts_ibm_fez_20260520T195040Z.json` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_fez_20260520T195040Z.json` | `efficient_su2` has the lowest mitigated energy in this lane. |
| `ibm_marrakesh` | 6 | `data/rust_vqe_methods/ansatz_ibm_validation_raw_counts_ibm_marrakesh_20260520T195051Z.json` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_marrakesh_20260520T195051Z.json` | `efficient_su2` has the lowest mitigated energy in this lane. |
| `ibm_marrakesh` | 8 | `data/rust_vqe_methods/ansatz_ibm_validation_raw_counts_ibm_marrakesh_20260520T195102Z.json` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_marrakesh_20260520T195102Z.json` | `efficient_su2` has the lowest mitigated energy in this lane. |

This is valid n=6/n=8 hardware evidence for the methods paper, but it does not
promote the topology-informed ansatz as backend-general superior on these
larger lanes. The paper claim must say that the hardware validation was run and
that the observed larger-lane ranking is ansatz- and backend-dependent.

### FIM Hamiltonian n=6/n=8 replication/ZNE

| Backend | n | Raw-count artefact | Analysis artefact | Mean abs raw scale-1 delta | Mean abs raw linear-ZNE delta |
|---|---:|---|---|---:|---:|
| `ibm_fez` | 6 | `data/scpn_fim_hamiltonian/fim_replication_zne_raw_counts_ibm_fez_20260520T195137Z.json` | `data/scpn_fim_hamiltonian/fim_replication_zne_analysis_ibm_fez_20260520T195137Z.json` | 0.4091 | 0.4021 |
| `ibm_fez` | 8 | `data/scpn_fim_hamiltonian/fim_replication_zne_raw_counts_ibm_fez_20260520T195149Z.json` | `data/scpn_fim_hamiltonian/fim_replication_zne_analysis_ibm_fez_20260520T195149Z.json` | 0.1479 | 0.1480 |
| `ibm_marrakesh` | 6 | `data/scpn_fim_hamiltonian/fim_replication_zne_raw_counts_ibm_marrakesh_20260520T195202Z.json` | `data/scpn_fim_hamiltonian/fim_replication_zne_analysis_ibm_marrakesh_20260520T195202Z.json` | 0.2474 | 0.2541 |
| `ibm_marrakesh` | 8 | `data/scpn_fim_hamiltonian/fim_replication_zne_raw_counts_ibm_marrakesh_20260520T195214Z.json` | `data/scpn_fim_hamiltonian/fim_replication_zne_analysis_ibm_marrakesh_20260520T195214Z.json` | 0.4231 | 0.4203 |

This closes the single-backend/no-ZNE criticism for the FIM paper without
changing the claim boundary: FIM protection remains a falsified or
backend-dependent hypothesis, not a promoted protection mechanism.

### Phase 3 n=6/n=8 larger-system reduced-Pauli/ZNE lanes

Dedicated retrieval and dynamic-width reduction are implemented in
`scripts/retrieve_phase3_large_system_ibm.py`.

| Backend | n | Raw-count artefact | Analysis artefact | Linear-ZNE mean abs deviation | Readout-mitigated linear-ZNE mean abs deviation |
|---|---:|---|---|---:|---:|
| `ibm_fez` | 6 | `data/phase3_entanglement_tomography/phase3_large_system_raw_counts_20260520T195233Z_ibm_fez_n6.json` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T195233Z_ibm_fez_n6.json` | 0.1688 | 0.1577 |
| `ibm_fez` | 8 | `data/phase3_entanglement_tomography/phase3_large_system_raw_counts_20260520T195245Z_ibm_fez_n8.json` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T195245Z_ibm_fez_n8.json` | 0.1395 | 0.0980 |
| `ibm_marrakesh` | 6 | `data/phase3_entanglement_tomography/phase3_large_system_raw_counts_20260520T195300Z_ibm_marrakesh_n6.json` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T195300Z_ibm_marrakesh_n6.json` | 0.3298 | 0.0688 |
| `ibm_marrakesh` | 8 | `data/phase3_entanglement_tomography/phase3_large_system_raw_counts_20260520T195312Z_ibm_marrakesh_n8.json` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T195312Z_ibm_marrakesh_n8.json` | 0.3795 | 0.0604 |

Each Phase 3 lane has 36 reduced-Pauli channels. The n=6/n=8 data directly
addresses the earlier "all hardware is n=4" limitation. Marrakesh shows a large
readout-mitigation shift on the larger lanes, so the paper should treat this as
a larger-width stress result with explicit readout sensitivity rather than as a
simple monotone extension of the n=4 narrative.

## Current Next Paper-Integration Slice

- Update the Phase 3 paper with the n=6/n=8 larger-system table, Kingston
  third-backend variant, Marrakesh n=8 repeat, and the readout-sensitivity
  boundary.
- Update the FIM paper with the second-backend, n=6/n=8, and ZNE replication
  table.
- Update the methods paper with the n=6/n=8 hardware validation result and the
  n=8 seed-23 sensitivity repeat while explicitly stating that the larger-lane
  ranking is not the same as the original n=4 topology-informed ansatz signal.
- Rebuild PDFs after the paper text and figure/table updates, not before.

## Additional Variant Batch

The follow-up batch was executed after the first larger-system results to
address whether one more useful variant class should be run while QPU time was
still available. The added variants were:

| Experiment | Backend | n | Job ID | Status | Analysis artefact |
|---|---|---:|---|---|---|
| `phase3_large_system_n6_kingston_2026-05-20` | `ibm_kingston` | 6 | `d871a3p789is7390a9p0` | `DONE` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T201613Z_ibm_kingston_n6.json` |
| `phase3_large_system_n8_kingston_2026-05-20` | `ibm_kingston` | 8 | `d871ab8p0eas73dljo90` | `ERROR` | Provider-side COS retrieval failure before quantum execution; `quantum_seconds` was zero. |
| `phase3_large_system_n8_kingston_retry1_2026-05-20` | `ibm_kingston` | 8 | `d871bv1789is7390abng` | `DONE` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T201817Z_ibm_kingston_n8.json` |
| `phase3_large_system_n8_marrakesh_readout_repeat_2026-05-20` | `ibm_marrakesh` | 8 | `d871aj1789is7390aa9g` | `DONE` | `data/phase3_entanglement_tomography/phase3_large_system_analysis_20260520T201657Z_ibm_marrakesh_n8.json` |
| `rust_vqe_methods_ansatz_validation_n8_seed23_fez_2026-05-20` | `ibm_fez` | 8 | `d871atas46sc73f7ju40` | `DONE` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_fez_20260520T201721Z.json` |
| `rust_vqe_methods_ansatz_validation_n8_seed23_marrakesh_2026-05-20` | `ibm_marrakesh` | 8 | `d871b90p0eas73dljpdg` | `DONE` | `data/rust_vqe_methods/ansatz_ibm_validation_analysis_ibm_marrakesh_20260520T201741Z.json` |

The Kingston Phase 3 lanes add a third-backend larger-width replication. The
Kingston n=6 full readout calibration matrix was singular, so the reducer used a
pseudoinverse replay and the paper treats that lane as a sensitivity boundary
rather than ordinary full correlated readout mitigation. The Kingston n=8 retry
completed with a pseudoinverse readout replay and a lower mitigated linear-ZNE
mean deviation than the raw linear-ZNE estimate.

The Marrakesh n=8 repeat reproduced the earlier large raw/linear-ZNE deviation
scale while again showing a small fully readout-mitigated linear-ZNE mean
deviation. This strengthens the paper's readout-sensitivity boundary: the
larger-width result is not a simple monotone extension of the n=4 reduced-Pauli
story.

The methods seed-23 repeats preserve the larger-width ordering in which
`efficient_su2` has the lowest mitigated energy on both Fez and Marrakesh. This
turns the methods hardware claim into a seed-sensitivity result rather than a
single-seed anecdote.

After the batch, the live IBM usage probe reported 1577 remaining seconds for
the account. No further QPU submissions were needed for this slice.
