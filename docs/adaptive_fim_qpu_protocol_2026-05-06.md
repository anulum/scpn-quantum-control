# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- adaptive FIM QPU protocol boundary

# Adaptive FIM QPU Protocol Boundary

Date: 2026-05-06

This document defines the approval boundary for any future adaptive
`lambda_fim` hardware campaign. It is a preregistration and readiness artefact,
not an IBM submission record and not evidence that adaptive FIM feedback has
been validated on hardware.

## Current evidence boundary

The repeated SCPN/FIM IBM follow-up on `ibm_kingston` falsified the simple
claim that the tested digital Trotter implementation with `lambda_fim = 4`
improves hardware coherence. The adaptive protocol therefore starts from the
opposite operational assumption: if leakage rises or exact-state retention
falls, the controller should reduce `lambda_fim`, not reward larger feedback.

Safe framing:

- The controller is a deterministic batch-level rule for selecting the next
  static `lambda_fim` value.
- The controller consumes committed leakage and retention witnesses.
- The controller does not run mid-circuit feedback.
- The controller does not submit QPU jobs.
- Any adaptive hardware claim requires a separately approved run, raw counts,
  integrity hashes, and analysis artefacts.

Blocked framing:

- Do not claim real-time adaptive feedback.
- Do not claim FIM coherence protection.
- Do not claim platform-general behaviour from a single backend or calibration
  window.

## Implemented software boundary

The implemented controller lives in
`src/scpn_quantum_control/analysis/adaptive_fim_feedback.py`.

Inputs:

- current `lambda_fim`,
- leakage probability in `[0, 1]`,
- exact-state retention probability in `[0, 1]`,
- optional depth,
- optional shot count,
- `AdaptiveFIMConfig`.

Modes:

- `leakage_suppression`: reduce `lambda_fim` when leakage exceeds target.
- `retention_recovery`: reduce `lambda_fim` when retention falls below target.

The rule is clipped to `[lambda_min, lambda_max]` and supports a deadband. This
keeps the hardware follow-up conservative after the negative FIM result.

## Candidate adaptive campaign

This campaign is not authorised by this document. It is the minimum design that
would make an adaptive FIM follow-up interpretable if approved later.

| Field | Candidate value |
| --- | --- |
| Backend class | IBM Heron r2 or equivalent calibrated gate-model backend |
| Initial backend | `ibm_kingston` only if calibration and queue state are acceptable |
| Qubits | n=4 |
| States | same representative magnetisation-sector set as the repeated FIM follow-up |
| Depths | `{2, 4, 6}` unless live transpilation rejects a depth |
| Lambda grid | batch 0 uses `{0, 1, 4}` from the repeated protocol |
| Adaptive batches | at most 2 follow-up batches after batch 0 |
| Shots | 4096 per measured circuit unless budget gate lowers this |
| Readout | full 16-state basis calibration required in each calibration window |
| Primary witness | magnetisation-sector leakage |
| Secondary witness | exact-state retention |
| Controller mode | `leakage_suppression` unless preregistered otherwise |
| Maximum lambda | 8.0 |
| Minimum lambda | 0.0 |

## QPU budget gate

Before submission, the prepared manifest must include:

- circuit count,
- shot count,
- estimated QPU seconds,
- expected queue class,
- backend name,
- backend calibration timestamp,
- max transpiled depth,
- max two-qubit gate count,
- readout calibration circuit count,
- abort criteria.

Hard budget rule:

- The campaign must have a written QPU-time estimate before submission.
- The estimate must fit within the remaining approved QPU budget.
- If the estimate exceeds the approved budget, reduce optional adaptive batches
  before reducing calibration integrity.

## Live transpilation gate

Each candidate circuit must pass live backend transpilation before submission.

Required metadata:

- physical qubit layout,
- transpiled depth,
- two-qubit gate count,
- measurement mapping,
- backend basis gates,
- optimisation level,
- pass-manager or transpiler version.

Abort criteria:

- any `lambda_fim > 0` arm exceeds twice the repeated-follow-up maximum depth,
- any arm loses the intended measurement register,
- any arm fails backend transpilation,
- any readout calibration circuit cannot be mapped to the same measured qubits,
- backend queue or calibration state makes the run likely to exceed the budget.

## Falsification and promotion rules

The adaptive campaign is considered successful only if it produces an
interpretable bounded result, not only if the effect is positive.

Promotion requires:

- raw count dictionaries,
- job IDs,
- SHA256 hashes,
- exact circuit manifest,
- readout calibration manifest,
- controller input witnesses,
- generated adaptive schedule,
- per-batch leakage and retention tables,
- claim-boundary update.

Positive adaptive claim requires:

- leakage or retention improves relative to the preregistered non-adaptive
  baseline after readout mitigation,
- improvement is not driven by reducing circuit depth or changing layout,
- the controller's chosen `lambda_fim` values are reproducible from the
  committed witness artefact.

Negative or null result claim:

- if the adaptive rule does not improve leakage or retention, report it as a
  bounded negative result for the tested backend, depths, states, and
  controller configuration.

Blocked claims after any single adaptive run:

- no quantum advantage,
- no general FIM protection,
- no backend-general adaptive feedback,
- no real-time control unless mid-circuit feedback is actually implemented and
  validated.

## Required artefact names

Future approved runs should use these names or document a replacement:

- `data/scpn_fim_hamiltonian/adaptive_fim_candidate_manifest_YYYY-MM-DD.json`
- `data/scpn_fim_hamiltonian/adaptive_fim_live_readiness_YYYY-MM-DD.json`
- `data/scpn_fim_hamiltonian/adaptive_fim_raw_counts_YYYY-MM-DD_JOBID.json`
- `data/scpn_fim_hamiltonian/adaptive_fim_analysis_YYYY-MM-DD_JOBID.json`
- `docs/adaptive_fim_claim_boundary_YYYY-MM-DD.md`

## Current status

Status: protocol designed, not authorised for QPU submission.

Next required action before any hardware use: generate a non-submitting
candidate manifest with circuit count, shot count, live transpilation metadata,
readout calibration plan, and QPU-time estimate.
