<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S1 Hybrid Feedback Loop -->

# S1 Hybrid Classical-Quantum Feedback Loop

Date activated: 2026-05-06

This note records the first implementation slice for strategic item S1. It is a
safe cross-shot feedback foundation, not a live IBM submission and not an
intra-shot real-time feedback claim.

## Implemented Surface

`src/scpn_quantum_control/hardware/feedback_loop.py` provides:

| API | Role |
|-----|------|
| `FeedbackLoopConfig` | Step, latency, QPU-budget, and approval guard. |
| `FeedbackCommand` | One scheduler command with optional estimated QPU seconds. |
| `FeedbackResult` | Counts, metrics, optional job ID, measured QPU seconds, metadata. |
| `FeedbackScheduler` | Protocol boundary for simulator, mock, or approved hardware schedulers. |
| `FeedbackObserver` | Protocol boundary for classical observers. |
| `FeedbackRunner` | Bounded cross-shot feedback loop with latency and QPU accounting. |
| `ProportionalMetricObserver` | Reference observer for metric-target proportional updates. |
| `RealtimeControllerScheduler` | Simulator scheduler wrapping `RealtimeSyncFeedbackController` with zero QPU spend. |

The realtime-controller scheduler accepts mapping payloads. `coupling_scale`
or `value` sets the next cross-shot coupling multiplier through the controller's
validated bounds, while `seed` controls deterministic finite-shot sampling. It
returns the controller's readout counts, live and statevector order-parameter
metrics, feedback action, coupling scales, correction angle, and an explicit
`qpu_seconds = 0.0` result.

## Scientific Boundary

This is cross-shot or cross-circuit feedback. Python and IBM Runtime round-trip
latency are not suitable for sub-microsecond intra-shot control. Intra-shot
feedback must be implemented as provider-side dynamic-circuit logic and then
wrapped by a separately approved scheduler.

The runner does not:

- create IBM Runtime sessions;
- read credentials;
- submit provider jobs by itself;
- bypass QPU budget gates;
- claim hardware convergence;
- claim adaptive FIM or DLA protection has been validated.

## Safety Gate

Hardware schedulers must expose `is_hardware=True`. `FeedbackRunner` rejects
such schedulers unless `hardware_approved=True` is passed explicitly. The
configuration also enforces maximum steps, total latency, per-step latency, and
QPU-second ceilings.

## Tests

`tests/test_feedback_loop.py` covers:

- rejection of unapproved hardware schedulers;
- convergence and step-record creation;
- QPU budget rejection before submission;
- QPU budget rejection after result accounting;
- proportional observer clipping and missing-metric errors.
- realtime-controller simulator scheduling with deterministic seeds;
- invalid realtime scheduler payload rejection.

## Latency Benchmark Command

The no-QPU S1 latency benchmark is regenerated with:

```bash
scpn-bench s1-feedback
```

or directly:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_s1_feedback_loop.py
```

It writes `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json`
and the corresponding CSV. The benchmark covers `FeedbackRunner` plus
`RealtimeControllerScheduler` wrapping `RealtimeSyncFeedbackController`; it does
not include IBM Runtime session creation, queue time, provider round-trip
latency, or QPU execution.

## Submission-Readiness Package

`src/scpn_quantum_control/hardware/feedback_submission.py` prepares a
provider-neutral, no-submission S1 package. It builds the monitored dynamic
circuit from `RealtimeSyncFeedbackController`, summarises qubits, classical
bits, depth, operations, conditional controls, and measurement requirements,
then evaluates platform capabilities against that payload.

The readiness layer currently distinguishes:

- IBM Heron-style dynamic-circuit backends;
- generic gate-based dynamic-circuit providers;
- neutral-atom analogue XY targets;
- continuous-variable analogue targets;
- local statevector simulation.

The corrected dynamic-circuit payload is ready for providers that declare
mid-circuit measurement, conditional rotations, cross-shot batches, and
sufficient qubits. Monitor resets are unconditional in the IBM-valid S1
payload; reset operations inside conditional blocks are not required.
Analogue-native targets are intentionally marked `manual_review` for this
payload because they are scientifically relevant for native/open-loop XY
follow-up, but they do not execute the same mid-circuit
measurement-and-conditional-rotation circuit without a separate analogue
feedback formulation.

Default preregistration helper:

```python
import numpy as np
from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_submission import (
    build_s1_feedback_submission_package,
)

controller = RealtimeSyncFeedbackController(
    np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
    np.array([0.1, 0.4], dtype=np.float64),
)
package = build_s1_feedback_submission_package(
    controller,
    n_rounds=3,
    shots_per_circuit=1024,
    repetitions=12,
    estimated_seconds_per_circuit=1.0,
)
print(package.to_dict())
```

This is the object we can use to state QPU need clearly: circuits,
shots-per-circuit, repetitions, estimated execution seconds, supported
platforms, blocked/manual-review platforms, and claim boundary.

The default S1 budget reserves two arms: the monitored feedback circuit and a
matched open-loop control. This is necessary because the job is only
interpretable if feedback is compared against the same oscillator family, shots,
repetitions, and layout target without feedback action.

The preregistration export is regenerated with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/export_s1_feedback_preregistration.py
```

It writes:

- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json`;
- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.md`.

The manifest includes no-submit provider dry-runs for:

- IBM Runtime dynamic-circuit execution;
- provider-neutral OpenQASM 3 style gate execution;
- analogue-native review, explicitly requiring a separate native-feedback
  formulation before any analogue submission.

## Approval-gated Hardware Scheduler

`src/scpn_quantum_control/hardware/feedback_hardware_scheduler.py` defines the
hardware boundary for eventual submission. It is deliberately fail-closed:

- a provider submitter must be injected by the caller;
- the scheduler does not read credentials;
- the scheduler does not create provider sessions;
- the approval record must set `approved=True`;
- the approval provider must match the scheduler provider;
- the approval package hash must match the preregistered manifest;
- estimated and reported QPU seconds must stay inside the approved budget.

This means S1 can be technically prepared for IBM or another provider without
creating an accidental submission path. A live job still requires explicit QPU
approval and a provider submitter wired for the selected backend.

## No-submit Capability Probes

`src/scpn_quantum_control/hardware/feedback_capability_probe.py` evaluates
backend metadata snapshots against the S1 preregistered package without
submitting jobs. It checks:

- qubit count;
- shot limit;
- circuit-batch limit;
- mid-circuit measurement support;
- conditional-control support;
- conditional-reset support;
- cross-shot batch support.

The preregistration manifest includes template probe decisions for a dynamic
gate backend and an analogue-native review target. Live provider adapters can
later feed real backend metadata into the same probe before any approval-gated
submission is considered.

`src/scpn_quantum_control/hardware/feedback_provider_metadata.py` provides the
no-submit metadata adapters for that step:

- `snapshot_from_generic_metadata(...)` for provider-neutral metadata records;
- `snapshot_from_qiskit_backend(...)` for Qiskit-style backend objects.

These adapters only inspect local metadata already supplied by the caller. They
do not fetch credentials, open provider sessions, or submit jobs.

## One-command Readiness Bundle

The complete no-QPU S1 readiness bundle is regenerated with:

```bash
scpn-bench s1-feedback-ready
```

This runs:

- `scripts/benchmark_s1_feedback_loop.py`;
- `scripts/export_s1_feedback_preregistration.py`;
- `scripts/analyse_s1_feedback_hardware.py` on the synthetic fixture.

The command regenerates latency artefacts, preregistration JSON/Markdown,
provider dry-runs, capability examples, and the synthetic-analysis summary
without credentials or hardware submission.

## Live Submission Preflight

`docs/campaigns/s1_feedback_readiness_index_2026-05-06.md` summarises the full no-QPU
readiness state, artefact inventory, commands, current preregistered job shape,
claim boundary, and remaining live-submission blockers.

`docs/campaigns/s1_live_submission_preflight_2026-05-06.md` is mandatory before any live
provider submitter is wired. It records required artefacts, scientific gates,
provider capability gates, budget gates, reproducibility gates, approval-record
requirements, stop conditions, and post-run requirements.

The key rule is simple: valid credentials, dry-runs, and capability probes are
not enough. A live S1 job requires the completed preflight checklist, a matching
`HardwareApprovalRecord`, and a new private audit record before submission.

## IBM Metadata Probe Command

`scripts/probe_s1_ibm_metadata.py` writes a no-submit S1 capability decision for
IBM/Qiskit-style metadata. Offline template usage:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_ibm_metadata.py \
  --metadata-json data/s1_feedback_loop/s1_ibm_metadata_template_2026-05-06.json
```

Already-authenticated runtime usage:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_ibm_metadata.py \
  --backend <backend-name> \
  --instance <optional-instance>
```

The script does not accept credential strings, does not submit jobs, and writes
`hardware_submission=false` into the output JSON. The `--backend` path loads
backend metadata through either saved Qiskit Runtime authentication or the local
credentials vault path; secret values are not printed or written into the
artefact.

## Generic Gate Metadata Probe Command

`scripts/probe_s1_generic_gate_metadata.py` writes a no-submit capability
decision for non-IBM gate-based targets from provider-neutral metadata:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_generic_gate_metadata.py \
  data/s1_feedback_loop/s1_generic_gate_metadata_template_2026-05-06.json
```

The command performs no network access and writes both
`hardware_submission=false` and `network_access=false` into the output JSON.

## Raw-count Analysis Harness

`scripts/analyse_s1_feedback_hardware.py` defines the preregistered analysis for
future S1 raw-count packages. It requires a JSON object with:

- `experiment_id`;
- `target_r`;
- optional `job_ids`;
- an `arms` list containing `feedback` and `matched_open_loop_control`;
- per-arm records with `r_live` and raw `counts`.

The generated summary reports per-arm total shots, mean live order parameter,
mean target error, final live order parameter, feedback-minus-control mean R,
absolute and relative target-error improvement, a positive/null-or-negative
decision, and the claim boundary. The analysis script exists before the live
run so the result cannot be interpreted post hoc.

Synthetic rehearsal fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analyse_s1_feedback_hardware.py \
  data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json
```

The fixture is not hardware data. It exists only to prove the schema and
analysis path before QPU time is spent.

### S1 Raw-count Schema

```json
{
  "experiment_id": "string",
  "target_r": 0.72,
  "job_ids": ["optional-provider-job-id"],
  "arms": [
    {
      "label": "feedback",
      "records": [
        {
          "r_live": 0.62,
          "counts": {"000": 180, "001": 76}
        }
      ]
    },
    {
      "label": "matched_open_loop_control",
      "records": [
        {
          "r_live": 0.54,
          "counts": {"000": 148, "001": 108}
        }
      ]
    }
  ]
}
```

`counts` are raw bitstring count dictionaries. `r_live` is the preregistered
finite-shot synchronisation observable for the same record. Each live package
must preserve provider job IDs and raw counts before derived analysis is run.

## Hardware Job Dossier Requirement

Every submission-ready hardware job must carry a dossier generated from
`HardwareJobDossier`. The required sections are:

- purpose;
- hypothesis;
- falsification condition;
- expected observables;
- circuit/package summary;
- QPU budget;
- platform fit;
- risks and confounds;
- decision tree for positive, null, negative, and contradictory outcomes;
- paper impact;
- follow-up avenue;
- possibilities opened;
- claim boundary;
- reproducibility package.

The S1 readiness package embeds this dossier directly in `package.to_dict()`.
This requirement applies to IBM, non-IBM gate-based, analogue, CV, and simulator
preparation. Jobs without a dossier are not submission-ready.

## Next S1 Work

Remaining S1 implementation should proceed in this order:

1. Add a provider-submitter implementation only after the live-submission
   preflight has been completed and explicitly approved.
