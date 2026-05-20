<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — DLA Multi-Device Replication Preregistration -->

# DLA Multi-Device Replication Preregistration

Date: 2026-05-06

This preregistration prepares a minimal second-backend DLA parity replication.
It does not submit an IBM job, reserve backend time, or authorise QPU spend.

## Scientific Question

Does the promoted `n=4` parity-sector/excitation-number correlated leakage
asymmetry replicate on a second Heron-class backend, or is it specific to the
`ibm_kingston` device/calibration window?

## Claim Boundary

Supported after successful execution and analysis:

- backend-transfer evidence for the same small `n=4` observable;
- comparison of sign, magnitude, and per-depth stability versus the promoted
  `ibm_kingston` Phase 2 A+G result;
- no claim beyond the sampled backend, calibration window, circuit family, and
  initial-state pair.

Blocked even after a positive result:

- broad quantum advantage;
- DLA-parity-only causality;
- monotone scaling;
- backend-universal protection;
- GUESS mitigation validation;
- full readout-matrix mitigation unless a complete basis calibration is also
  submitted.

## Target Backend Rule

Use a second Heron-class backend only. Do not rerun on `ibm_kingston` for this
specific replication item. The actual backend name must be selected from the
live IBM account-visible backend list immediately before readiness checks.

## Circuit Matrix

| Field | Value |
|-------|-------|
| `n` | `4` |
| Coupling model | same heterogeneous Kuramoto-XY matrix as Phase 2 A+G |
| States | `0011` even reference, `0001` odd reference |
| Depths | `4, 6, 8, 10, 14, 20` |
| Repetitions | `12` per state/depth |
| Main shots | `4096` per circuit |
| Readout states | `0011`, `0001`, `0000`, `1111` |
| Readout shots | `8192` per circuit |

Circuit count:

- main circuits: `2 states x 6 depths x 12 reps = 144`;
- readout circuits: `4`;
- total circuits: `148`.

## QPU-Time Estimate

Expected IBM-reported QPU time: `3-6` minutes if live-transpiled depths remain
comparable to Phase 2 A+G.

Budget ceiling for this preregistered block: `10` IBM-reported QPU minutes.

Abort before submission if the live estimate exceeds the ceiling or if the
remaining allocation cannot cover the block plus a 25 % safety margin.

## Live Readiness Gates

Before submission:

- confirm backend is Heron-class and not `ibm_kingston`;
- confirm backend is account-visible and operational in the IBM dashboard;
- generate circuits from committed code only;
- live-transpile every circuit on the selected backend;
- reject if max transpiled depth exceeds the Phase 2 A+G envelope by more than
  25 %;
- reject if max two-qubit gate count exceeds the Phase 2 A+G envelope by more
  than 25 %;
- record backend name, timestamp, calibration snapshot metadata, depth summary,
  two-qubit gate summary, shot count, circuit count, and estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Primary observable:

- parity leakage per state/depth/repetition.

Primary comparison:

- odd-reference leakage minus even-reference leakage at matched depth.

Promoted summaries:

- per-depth mean asymmetry;
- per-depth Welch/permutation checks where sample size permits;
- Fisher-style combined statistic only as descriptive and with dependence
  caveat;
- sign agreement with Phase 2 A+G;
- effect-size comparison against Phase 2 A+G.

Readout handling:

- use the four readout states for exact-state parity-confusion correction;
- do not claim full `2^n x 2^n` confusion-matrix mitigation from this four-state
  calibration.

## Falsification Rules

The backend-transfer claim is weakened or rejected if:

- the sign reverses for most promoted depths;
- the effect collapses inside readout-only uncertainty;
- the circuit-depth envelope is materially worse than Phase 2 A+G;
- temporal drift or backend calibration metadata makes the comparison
  uninterpretable.

If the result is null or opposite-sign, publish it as a multi-device boundary,
not as a failed experiment.

## Output Artefacts

Expected paths after approved execution:

- `data/phase3_multidevice_dla/phase3_multidevice_<backend>_<timestamp>.json`;
- `data/phase3_multidevice_dla/phase3_multidevice_summary_<date>.json`;
- `data/phase3_multidevice_dla/phase3_multidevice_row_metrics_<date>.csv`;
- `docs/campaigns/phase3_multidevice_dla_manifest_<date>.md`.

Each artefact must include job ID, backend, circuit metadata, raw counts,
SHA256 hashes, depth/gate summaries, and reproduction commands.

## Submission Boundary

This preregistration is complete. QPU execution remains blocked until backend
selection, live readiness artefacts, budget confirmation, and explicit approval
are completed in a separate task.
