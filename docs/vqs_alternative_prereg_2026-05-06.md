<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — VQS Alternative Preregistration -->

# Variational Quantum Simulation Alternative Preregistration

Date: 2026-05-06

This preregistration defines an offline-first comparison between Trotterized
Kuramoto-XY evolution and a variational quantum simulation (VQS) alternative. It
does not submit IBM jobs, reserve backend time, or authorise QPU spend.

## Scientific Question

Can a topology-informed variational quantum simulation ansatz approximate the
same promoted Kuramoto-XY observables with lower compiled depth or lower
two-qubit gate burden than Trotter circuits?

## Claim Boundary

Supported after successful execution and analysis:

- small-system resource/fidelity comparison between VQS and Trotter circuits;
- identification of regimes where VQS is a useful hardware-depth proxy;
- negative evidence if VQS cannot preserve the promoted observables.

Blocked even after a positive result:

- exact time evolution beyond the tested observables;
- backend-general depth advantage;
- quantum advantage;
- replacement of Trotter data already used in papers without a new
  preregistered analysis;
- extrapolation to larger `n` without tensor-network or exact-reference checks.

## Candidate VQS Modes

Allowed candidates:

- real-time variational evolution using a topology-informed ansatz;
- McLachlan-style projected dynamics where metric and force terms are recorded;
- time-sliced variational refits against exact small-system states;
- structured ansatz snapshots trained to match selected observables rather than
  full statevectors.

Disallowed shortcuts:

- fitting to hardware outcome data before preregistering the circuit;
- changing the target observable after seeing results;
- comparing VQS at lower accuracy against Trotter at higher accuracy without
  reporting the accuracy gap;
- claiming physical dynamics from an ansatz that only matches one scalar.

## Offline Readiness Matrix

Default no-QPU readiness scope:

| Field | Value |
|-------|-------|
| `n` | `4, 6, 8` where references are tractable |
| Families | DLA parity A+G, popcount controls, FIM `lambda=0` vs `lambda=4` |
| Target times/depths | match promoted Trotter depths or evolution times |
| Ansatz families | topology-informed, EfficientSU2, TwoLocal baseline |
| Seeds | at least `5` fixed optimization seeds |
| Optimizer budget | fixed and recorded before execution |
| References | exact statevector for `n=4`, exact/sparse/TN reference where feasible for `n=6,8` |

Readiness outputs:

- compiled depth and two-qubit gate count for each candidate;
- target-observable error versus reference;
- state fidelity where feasible;
- optimization failure rate;
- parameter count;
- transpilation seed sensitivity.

## Promotion Gates

A VQS candidate may be promoted only if:

- the target-observable error is no worse than the Trotter comparator within the
  preregistered tolerance;
- median compiled depth is at least 25 % lower than the Trotter comparator, or
  median two-qubit gate count is at least 25 % lower;
- optimization succeeds for at least 80 % of fixed seeds;
- the result is not dominated by one lucky initialization;
- claim text reports both resource gain and approximation error.

Default tolerances:

- parity survival or leakage observable error: `<= 0.02` absolute;
- exact-state retention error: `<= 0.02` absolute;
- FIM sector-survival error: `<= 0.03` absolute;
- state fidelity target where full reference exists: `>= 0.98`.

## Optional Hardware Scope

If offline gates pass and QPU execution is separately approved, use the smallest
block that can test whether the lower-depth VQS circuit actually improves
hardware retention/leakage:

| Field | Value |
|-------|-------|
| `n` | `4` |
| Families | one DLA parity pair; optional one FIM pair |
| Circuits | Trotter comparator and promoted VQS candidate |
| Depths/times | one promoted signal point and one stress point |
| Repetitions | `6` per circuit |
| Shots | `4096` |
| Readout states | prepared states plus `0000` and `1111` |
| Readout shots | `8192` |

Circuit ceiling:

- DLA-only block: `<= 120` circuits;
- DLA plus FIM block: `<= 220` circuits.

IBM-reported QPU-time ceiling:

- DLA-only: `10` minutes;
- DLA plus FIM: `18` minutes.

## Live Readiness Gates

Before any hardware submission:

- regenerate reference values and VQS parameters from committed artefacts only;
- live-transpile Trotter and VQS circuits with identical backend/layout and seed
  rules;
- reject if the VQS candidate fails promotion gates after live transpilation;
- reject if VQS improves nominal depth but worsens calibration-weighted
  two-qubit error load;
- record backend, calibration timestamp, target observables, trained parameter
  artefact hash, circuit count, shot count, depth summary, two-qubit gate
  summary, and estimated QPU minutes;
- get explicit approval immediately before submission.

## Analysis Plan

Offline primary endpoints:

- target-observable error versus reference;
- resource delta versus Trotter;
- success rate across seeds;
- parameter count and optimizer evaluations;
- calibration-weighted error-load proxy where backend metadata exists.

Hardware endpoints after separate approval:

- parity leakage and exact-state retention by method;
- readout-corrected comparison where exact-state or full-basis calibration
  exists;
- comparison against the offline predicted observable error.

Required reporting:

- VQS is approximate unless equivalence is explicitly proven;
- lower depth is useful only if target observables remain within tolerance;
- negative VQS results are reported as ansatz/cost boundaries.

## Falsification Rules

The VQS alternative is rejected or downgraded if:

- it does not meet target-observable tolerance;
- resource gains disappear after live transpilation;
- optimization is unstable across seeds;
- VQS matches one scalar while failing parity, retention, or sector survival;
- optional hardware counts show worse retention/leakage despite lower depth.

## Output Artefacts

Expected paths after offline readiness:

- `data/phase3_vqs_alternative/vqs_readiness_<date>.json`;
- `data/phase3_vqs_alternative/vqs_candidate_rows_<date>.csv`;
- `data/phase3_vqs_alternative/vqs_resource_rows_<date>.csv`;
- `docs/phase3_vqs_alternative_readiness_<date>.md`.

Expected paths after approved hardware execution:

- `data/phase3_vqs_alternative/vqs_counts_<backend>_<timestamp>.json`;
- `data/phase3_vqs_alternative/vqs_summary_<date>.json`;
- `docs/phase3_vqs_alternative_manifest_<date>.md`.

Every artefact must include target observables, ansatz identity, parameter
hashes, optimizer settings, seeds, backend target where applicable, depth/gate
summaries, raw counts where applicable, SHA256 hashes, and reproduction
commands.

## Submission Boundary

This preregistration is complete. Hardware execution remains blocked until
offline readiness artefacts, backend selection, budget confirmation, and
explicit approval are completed in a separate task.
