<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — DLA State/Layout Randomisation Preregistration -->

# DLA State/Layout Randomisation Preregistration

Date: 2026-05-06

This preregistration prepares a systematic state/layout randomisation control
for the DLA parity hardware programme. It does not submit an IBM job, reserve
backend time, or authorise QPU spend.

## Scientific Question

How much of the promoted `n=4` leakage asymmetry is explained by parity sector,
excitation count, physical qubit layout, readout properties, and coupling-map
placement?

## Claim Boundary

Supported after successful execution and analysis:

- mechanism-separation evidence for the existing small-system DLA parity
  observation;
- state-level and layout-level leakage summaries;
- identification of whether excitation count or layout explains a substantial
  share of the original contrast.

Blocked even after a positive result:

- DLA-parity-only causality;
- backend-universal protection;
- monotone scaling;
- quantum advantage;
- full readout-matrix mitigation, unless all 16 basis states are calibrated for
  every selected layout.

## Backend and Layout Rule

Use one Heron-class backend selected immediately before live readiness checks.
The backend may be `ibm_kingston` only if the explicit purpose is same-device
mechanism separation rather than multi-device transfer.

Select three connected four-qubit windows from the live coupling map:

- each window must support the generated circuit topology after transpilation;
- windows should minimise recent readout error and two-qubit error where the
  provider exposes calibration metadata;
- no window may be selected manually after seeing outcome counts.

## Circuit Matrix

| Field | Value |
|-------|-------|
| `n` | `4` |
| Coupling model | same heterogeneous Kuramoto-XY matrix as Phase 2 A+G |
| States | `0011`, `0101`, `0001`, `0010`, `0111` |
| Depths | `6, 8, 10, 14` |
| Layouts | `3` connected four-qubit windows |
| Repetitions | `8` per state/depth/layout |
| Main shots | `4096` per circuit |
| Readout states | all five prepared states per layout |
| Readout shots | `8192` per circuit |

Circuit count:

- main circuits: `5 states x 4 depths x 3 layouts x 8 reps = 480`;
- readout circuits: `5 states x 3 layouts = 15`;
- total circuits: `495`.

## QPU-Time Estimate

Expected IBM-reported QPU time: `8-15` minutes if live-transpiled depths remain
within the popcount-control envelope.

Budget ceiling for this preregistered block: `20` IBM-reported QPU minutes.

Abort before submission if the live estimate exceeds the ceiling or if the
remaining allocation cannot cover the block plus a 25 % safety margin.

## Live Readiness Gates

Before submission:

- confirm backend is Heron-class, account-visible, and operational;
- select three connected four-qubit windows before outcome data exists;
- record window qubits, calibration timestamp, readout errors, and two-qubit
  error summaries where available;
- generate circuits from committed code only;
- live-transpile every circuit on the selected backend and layout;
- reject if max depth exceeds the completed popcount-control depth envelope by
  more than 25 %;
- reject if max two-qubit gate count exceeds the completed popcount-control
  envelope by more than 25 %;
- record shot count, circuit count, expected QPU minutes, depth summary, and
  two-qubit gate summary;
- get explicit approval immediately before submission.

## Analysis Plan

Primary observables:

- parity leakage;
- exact-state retention;
- excitation-count leakage where the observable is well defined;
- readout-only exact-state retention for the prepared calibration states.

Primary model:

- compare leakage by state, parity sector, excitation count, depth, and layout;
- report layout-stratified and layout-pooled summaries;
- treat layout as a grouping factor, not as a nuisance to average away without
  reporting.

Promoted summaries:

- per-state/depth/layout leakage table;
- parity-sector contrast at matched excitation controls;
- same-popcount within-sector swap contrasts;
- excitation-inversion contrast;
- layout variance and worst/best layout spread;
- sign agreement with the original Phase 2 A+G contrast where applicable.

Readout handling:

- use exact-state readout calibrations for the five prepared states;
- do not claim full `2^n x 2^n` confusion-matrix mitigation from five-state
  calibration;
- record whether a future complete 16-state basis calibration is justified.

## Falsification Rules

The clean parity-sector explanation is weakened if:

- same-popcount within-sector swaps are comparable to or larger than the
  original parity contrast;
- layout variance dominates the parity/excitation effect;
- the sign changes across selected layouts;
- readout-only correction removes the promoted sign;
- excitation count explains the observed ordering better than parity sector.

If the result is mixed, report it as mechanism-separation evidence rather than
as confirmation or failure of the original paper.

## Output Artefacts

Expected paths after approved execution:

- `data/phase3_state_layout_dla/phase3_state_layout_<backend>_<timestamp>.json`;
- `data/phase3_state_layout_dla/phase3_state_layout_summary_<date>.json`;
- `data/phase3_state_layout_dla/phase3_state_layout_row_metrics_<date>.csv`;
- `data/phase3_state_layout_dla/phase3_state_layout_layout_metrics_<date>.csv`;
- `docs/phase3_state_layout_dla_manifest_<date>.md`.

Each artefact must include job ID, backend, layout mapping, calibration
metadata, raw counts, SHA256 hashes, depth/gate summaries, and reproduction
commands.

## Submission Boundary

This preregistration is complete. QPU execution remains blocked until backend
selection, live layout selection, transpilation readiness artefacts, budget
confirmation, and explicit approval are completed in a separate task.
