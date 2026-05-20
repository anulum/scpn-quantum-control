<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Phase 3 third-backend ZNE extension -->

# Phase 3 Third-Backend ZNE Extension: `ibm_kingston`

Date: 2026-05-20

## Decision

The fourth paper review requested a stronger endpoint before treating the Phase
3 paper as complete: either a third-backend ZNE replication or an `n=6`
extension. The implemented lane is a third-backend `n=4` ZNE replication,
because it reuses the already preregistered five-channel reduced-Pauli ZNE
stress test without introducing a rushed new `n=6` protocol.

Accessible IBM backends for the account at selection time:

| Backend | Qubits | Operational | Pending jobs |
|---|---:|---:|---:|
| `ibm_fez` | 156 | `True` | 0 |
| `ibm_marrakesh` | 156 | `True` | 1 |
| `ibm_kingston` | 156 | `True` | 38 |

`ibm_sherbrooke` and `ibm_torino` were probed first and were not available to
the account. Since `ibm_marrakesh` and `ibm_fez` are already in the manuscript,
`ibm_kingston` is the only accessible third-backend candidate.

## Readiness

No-submit readiness command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/phase3_entanglement_tomography_ibm.py \
  --backend ibm_kingston \
  --zne-subset-rows data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez_pinned_full_readout.csv \
  --zne-noise-scales 1,3,5 \
  --max-depth 1900 \
  --max-total-gates 5000
```

Readiness artefact:

- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T030151Z.json`
- SHA256: `858520263a8a855e12438391311545159a6971e42818131cd98756da0cf1a165`

Readiness result:

| Field | Value |
|---|---:|
| Status | `readiness_passed` |
| Physical qubits | `[141, 142, 143, 144]` |
| Main circuits | 45 |
| Readout circuits | 16 |
| Total circuits | 61 |
| Estimated QPU minutes | 0.5591666666666667 |
| Budget ceiling minutes | 25.0 |
| Maximum transpiled depth | 1686 |
| Maximum basis-expansion ratio | 4.574585635359116 |

## Submission

Approved submission command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/phase3_entanglement_tomography_ibm.py \
  --backend ibm_kingston \
  --zne-subset-rows data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez_pinned_full_readout.csv \
  --zne-noise-scales 1,3,5 \
  --max-depth 1900 \
  --max-total-gates 5000 \
  --submit \
  --confirm-budget
```

Pending execution artefact:

- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T030211Z.json`
- SHA256 after pending-job registration:
  `56baf1812ba8193fc67b05f27fcf99ae8ce4960a9e16e604e9c8b09e7884d635`

Queued IBM jobs:

| Role | Job ID | Status at registration |
|---|---|---|
| Main ZNE circuits | `d86i8fas46sc73f70vg0` | `QUEUED` |
| Full 16-state readout calibration | `d86ijr9789is738vnh30` | `QUEUED` |

The main job was submitted by the guarded runner. The readout job was submitted
independently after the main job entered the IBM queue, because the guarded
runner waits for the main result before launching readout. The local sequential
runner was then stopped to prevent a duplicate readout submission.

## Claim Boundary

This is a pending third-backend ZNE replication only. It is not yet a result and
must not be described as confirming or falsifying the existing Fez/Marrakesh
pattern until both jobs complete, raw counts are retrieved, the ZNE reducer is
run, and the paper is updated from committed artefacts.

## Next Gate

When both IBM jobs are `DONE`:

1. retrieve raw counts into the pending Kingston artefact;
2. run `scripts/analyse_phase3_entanglement_zne.py` with a Kingston result tag;
3. compare DLA and FIM channel-level ZNE behaviour against Fez;
4. update the LaTeX manuscript and PDF with either replication, partial
   replication, or falsification language.
