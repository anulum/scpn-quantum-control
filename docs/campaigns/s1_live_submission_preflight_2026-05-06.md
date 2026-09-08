<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S1 Live Submission Preflight -->

# S1 Live Submission Preflight

Date created: 2026-05-06

This checklist is mandatory before wiring any live provider submitter for the S1
hybrid feedback job. It is deliberately manual because the final decision spends
real QPU budget and changes the evidentiary status of the project.

## Non-negotiable Rule

No S1 live job may be submitted unless every gate below is complete and recorded
in a new private audit record. A dry-run package, capability probe, or valid credential is
not sufficient by itself.

## Required Artefacts

- [ ] `scpn-bench s1-feedback-ready` was run immediately before submission.
- [ ] `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json` exists.
- [ ] `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.md` exists.
- [ ] `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json` exists.
- [ ] `data/s1_feedback_loop/s1_feedback_analysis_summary_2026-05-06.json` exists.
- [ ] The hardware-job dossier embedded in the preregistration manifest was reviewed.
- [ ] The raw-count analysis schema was reviewed against the planned provider output.

## Scientific Gates

- [ ] Purpose is unchanged: feedback-vs-matched-open-loop steering of `R_live`.
- [ ] Hypothesis is unchanged from the dossier.
- [ ] Falsification condition is unchanged from the dossier.
- [ ] Matched open-loop control arm is included.
- [ ] Positive, null, negative, and contradictory decision branches are accepted.
- [ ] Claim boundary is accepted: no quantum advantage, no backend-independent claim,
      no sub-microsecond feedback claim unless provider-side logic supports it.

## Provider Capability Gates

- [ ] Live backend metadata was captured without submitting a job.
- [ ] Declared capacities are positive Python integers; optional snapshot
      limits may be undeclared (`None`). Capability, simulator and submission
      policy flags are booleans, not truthy strings or integers.
- [ ] Capability probe reports `ready` for the selected backend.
- [ ] Backend supports required qubit count.
- [ ] Backend supports required shot count.
- [ ] Backend supports required circuit-batch count.
- [ ] Backend supports cross-shot batches, including single-circuit S1 packages.
- [ ] Backend supports mid-circuit measurement.
- [ ] Backend supports conditional reset.
- [ ] Backend supports conditional control.
- [ ] Live transpilation was performed without submission.
- [ ] Live transpiled depth and operation counts were recorded.
- [ ] Transpiled payload still fits the scientific claim boundary.

## Budget Gates

- [ ] Remaining QPU budget was checked.
- [ ] Reconcile any previous provider overrun. The scheduler records returned
      usage and job identity before raising on an exceeded limit and refuses
      follow-up commands against the exhausted approval. This in-memory ledger
      does not undo spend or reconcile interrupted/provider-exception jobs.
- [ ] Requested S1 execution estimate was recorded.
- [ ] Execution time itself is positive; queue/calibration overhead cannot
      make a zero-execution reservation ready.
- [ ] Package rounds, circuit count, shots and repetitions are positive Python
      integers, not booleans or fractional values. Reservation components and
      their total are finite and non-negative; preparation rejects overflow.
- [ ] Queue/calibration uncertainty was recorded separately from execution estimate.
- [ ] Approval record `max_qpu_seconds` is greater than or equal to the planned spend.
- [ ] The limit is finite and non-negative; NaN and infinity are invalid.
- [ ] Approval is the boolean `True`, not a string or integer. Invalid approval
      types fail at record construction; boolean `False` refuses dispatch.
- [ ] Approval record provider matches selected backend provider.
- [ ] Approval record package hash matches the exact preregistration manifest.
- [ ] The scheduler snapshots the manifest at construction and gives each
      provider call a detached copy. Changing the package requires a new
      scheduler and matching approval; editing an inspected copy has no effect.

## Data and Reproducibility Gates

- [ ] Raw-count output path was prepared.
- [ ] Job metadata path was prepared.
- [ ] SHA256 hashing plan was prepared.
- [ ] Analysis command was recorded:
      `python scripts/analyse_s1_feedback_hardware.py <raw-count-package.json>`.
- [ ] Failure handling was recorded: cancelled/failed jobs remain auditable until
      classified as superseded or resolved.

## Approval Record

The live scheduler may only be wired after creating a `HardwareApprovalRecord`
with:

- `approved=True`;
- non-empty `approval_id`;
- named `approver`;
- exact preregistration package hash;
- selected provider;
- approved QPU-second ceiling;
- session-log reference in `notes`.

### Dispatch and recovery boundary

`FeedbackResult` rejects fractional/boolean shot counts and non-finite/boolean
control metrics. It snapshots the provider's counts and metrics at construction;
retain raw provider evidence separately. Its mappings remain editable, so the
proportional observer also validates its selected metric immediately before use
and rejects corrupted data without changing its control parameter. Do not repair
invalid measurements by coercing counts or replacing NaN with a synthetic value.

`FeedbackRunner` accepts only exact boolean approval and scheduler hardware flags.
It rechecks approval when `run()` starts and before every dispatch, including
after observer callbacks; revocation blocks the next command. Malformed or absent
hardware identity is rejected instead of assumed to mean simulation. The existing
explicit `require_hardware_approval=False` opt-out disables only this runner-level
check; it is not provider approval or authority to operate hardware. S1 live runs
must retain the default approval requirement and use the approved scheduler below.

`ApprovalGatedFeedbackHardwareScheduler` allows one in-flight call per instance;
concurrent and reentrant calls fail immediately without reaching the provider.
Its provider, callback, approval and descriptor are construction-time settings.
Provider callbacks receive detached command and manifest copies.

A provider exception, cancellation or malformed result leaves an unknown
outcome. The attempt remains in `submissions` with `result_qpu_seconds=None`
(not zero); `requires_reconciliation=True` blocks every subsequent dispatch.
The original exception propagates. `spent_qpu_seconds` includes only known
reported usage; it is not the total bill when an attempt is unresolved.
Reported overruns remain charged and recorded even when submission raises.

This wrapper is in-memory, not a persistent or multi-process budget ledger.
Do not retry through a fresh instance or reuse the original allowance after
restart. Archive available attempt records and query the provider's job and
usage records outside the scheduler. Resolve uncertain consumption and obtain
an explicit approval for the remaining budget before constructing a replacement.
There is intentionally no automatic retry or reconciliation-reset API.
These guards do not certify real-provider execution or command membership in a
scientific protocol; the campaign's preregistration checks remain mandatory.

## Stop Conditions

Stop and do not submit if any of the following occurs:

- backend capability status is `blocked` or `unknown`;
- live transpilation changes the circuit beyond the preregistered claim boundary;
- matched open-loop control cannot be submitted in the same campaign;
- QPU budget approval is missing or lower than the planned spend;
- provider dry-run payload and live backend metadata disagree;
- raw-count archival path is not ready;
- private audit record cannot be written.

## Post-run Requirement

Immediately after completion:

- write raw counts before analysis;
- write job IDs and backend metadata;
- hash raw-count files;
- run the preregistered analysis script;
- update the S1 private audit record;
- do not alter the hypothesis or decision tree after seeing results.
