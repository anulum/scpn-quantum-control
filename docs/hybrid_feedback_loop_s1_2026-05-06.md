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

The dynamic-circuit payload is ready for providers that declare mid-circuit
measurement, conditional reset, conditional rotations, cross-shot batches, and
sufficient qubits. Analogue-native targets are intentionally marked
`manual_review` for this payload because they are scientifically relevant for
native/open-loop XY follow-up, but they do not execute the same mid-circuit
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

1. Export a JSON preregistration manifest from the S1 submission package.
2. Add provider-specific dry-run adapters for IBM Runtime and at least one
   non-IBM target; do not submit hardware.
3. Add an approval-gated hardware scheduler only after a preregistered QPU budget and
   explicit approval.
