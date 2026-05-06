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

## Next S1 Work

Remaining S1 implementation should proceed in this order:

1. Add a simulator scheduler that wraps `RealtimeSyncFeedbackController` without
   QPU access.
2. Add a latency benchmark command and document it in
   `docs/pipeline_performance.md`.
3. Add an IBM Dynamic Circuits design payload only; do not submit hardware.
4. Add an approval-gated IBM scheduler only after a preregistered QPU budget and
   explicit approval.
