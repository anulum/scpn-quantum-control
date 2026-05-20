<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Entanglement/Tomography Preregistration -->

# Entanglement Entropy / Tomography Check Preregistration

Date: 2026-05-06

This preregistration defines a bounded entanglement-structure check for the
Kuramoto-XY and FIM hardware programme. It does not submit IBM jobs, reserve
backend time, or authorise QPU spend.

## Scientific Question

Can a small, explicitly cost-bounded tomography or shadow-tomography block
distinguish state-preparation/readout artefacts from genuine changes in
two-qubit and half-chain entanglement structure?

## Claim Boundary

Supported after successful execution and analysis:

- small-system entanglement witness, purity, or reduced-density-matrix
  consistency checks for a specified circuit family;
- comparison between exact classical predictions and hardware-estimated
  reduced observables;
- evidence that a leakage or retention contrast is or is not accompanied by
  measurable entanglement-structure change.

Blocked even after a positive result:

- scalable tomography;
- quantum advantage;
- broad many-body-localisation proof;
- backend-general entanglement dynamics;
- full-state reconstruction beyond the explicitly calibrated small system;
- claims about unmeasured subsystems or unmeasured depths.

## Candidate Measurement Modes

Use the cheapest mode that can answer the question.

| Mode | Default use | Main limitation |
|------|-------------|-----------------|
| Reduced two-qubit tomography | validate pairwise correlations on selected logical edges | does not reconstruct the full state |
| Half-chain tomography for `n=4` | estimate two-qubit reduced density matrix across the `2|2` cut | not scalable beyond small `n` |
| Classical shadows | estimate selected Pauli observables and purity-like diagnostics | requires careful basis bookkeeping |
| Full state tomography | allowed only for `n=4` and only if explicitly justified | highest circuit count and strongest overfit risk |

Default promoted mode: reduced two-qubit tomography plus selected half-chain
Pauli measurements for `n=4`.

Classical shadows may replace the default mode if the implementation records
the random basis seed, basis per shot block, observable estimator, and confidence
interval method.

## Offline Readiness Matrix

Default no-QPU readiness scope:

| Field | Value |
|-------|-------|
| `n` | `4` |
| Circuit families | DLA parity A+G, FIM `lambda=0` vs `lambda=4` |
| States | `0011`, `0001`, and one FIM-sector reference state |
| Depths | one shallow depth and one promoted signal depth per family |
| Observables | selected pairwise Pauli correlators, half-chain purity proxy, parity survival |
| Classical reference | exact statevector or density-matrix simulation from committed code |

Offline readiness must produce:

- observable list and basis grouping;
- circuit count before transpilation;
- predicted ideal values;
- noise-model sensitivity if available;
- explicit statement whether full tomography is unnecessary.

## Optional Hardware Scope

If QPU execution is later approved, use a minimal falsification block:

| Field | Value |
|-------|-------|
| `n` | `4` |
| Families | one DLA parity depth, one FIM pair if still scientifically needed |
| States | maximum `3` prepared states |
| Measurement settings | maximum `18` basis settings per state/family |
| Repetitions | `3` per setting |
| Shots | `2048` |
| Readout states | prepared states plus `0000` and `1111` |
| Readout shots | `8192` |

Ceiling:

- default DLA-only block: `<= 200` circuits;
- DLA plus FIM block: `<= 380` circuits;
- IBM-reported QPU-time ceiling: `15` minutes for DLA-only, `25` minutes if
  the FIM block is explicitly approved.

Do not run full state tomography if reduced measurements answer the preregistered
question.

## Live Readiness Gates

Before any hardware submission:

- confirm that exact classical predictions and observable definitions are
  committed;
- generate all measurement circuits from committed code only;
- verify basis labels and qubit ordering against Qiskit little-endian output;
- live-transpile all circuits on the selected backend/layout;
- reject if measurement-basis expansion increases max depth by more than 20 %
  over the source circuit;
- reject if circuit count or shot count exceeds the ceiling;
- record backend, calibration timestamp, circuit count, shot count, basis list,
  depth summary, two-qubit gate summary, and estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Primary diagnostics:

- pairwise Pauli correlators on high-priority logical edges;
- half-chain purity or second-Renyi proxy where estimator assumptions are valid;
- parity survival and exact-state retention reported alongside entanglement
  diagnostics;
- deviation from exact classical reference with confidence intervals.

Required reporting:

- basis grouping and estimator formula;
- readout-correction boundary;
- bootstrap or binomial uncertainty method;
- comparison before and after any exact-state readout correction;
- failure cases where the estimator is too noisy to support interpretation.

## Falsification Rules

The entanglement-structure interpretation is rejected or downgraded if:

- estimated correlators are statistically consistent with readout/calibration
  artefacts;
- the measured entanglement proxy is not distinguishable from the product-state
  or classical-reference null;
- readout correction changes the sign of the promoted comparison;
- the uncertainty interval is larger than the effect being interpreted;
- tomography overhead changes the circuit family enough to invalidate the
  comparison.

If the result is inconclusive, report it as a measurement-cost boundary rather
than a physics confirmation.

## Output Artefacts

Expected paths after offline readiness:

- `data/phase3_entanglement_tomography/entanglement_tomography_readiness_<date>.json`;
- `data/phase3_entanglement_tomography/entanglement_observable_rows_<date>.csv`;
- `docs/campaigns/phase3_entanglement_tomography_readiness_<date>.md`.

Expected paths after approved hardware execution:

- `data/phase3_entanglement_tomography/entanglement_tomography_counts_<backend>_<timestamp>.json`;
- `data/phase3_entanglement_tomography/entanglement_tomography_summary_<date>.json`;
- `data/phase3_entanglement_tomography/entanglement_tomography_rows_<date>.csv`;
- `docs/campaigns/phase3_entanglement_tomography_manifest_<date>.md`.

Each artefact must include observable definitions, basis settings, backend,
layout, calibration metadata, raw counts where applicable, SHA256 hashes,
estimator formulas, confidence intervals, and reproduction commands.

## Submission Boundary

This preregistration is complete. Hardware execution remains blocked until the
offline readiness artefacts, backend selection, budget confirmation, and
explicit approval are completed in a separate task.
