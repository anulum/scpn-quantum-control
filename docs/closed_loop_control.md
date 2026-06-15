# Closed-Loop Control Analysis

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.control.closed_loop_analysis` turns the response of the
measurement-feedback synchronisation controller
(`control.realtime_feedback.RealtimeSyncFeedbackController`) into a
control-theoretic verdict and gates any non-simulation execution behind an
explicit policy.

The controller measures a finite-shot order-parameter estimate each round and
adjusts the coupling scale toward a set-point. This module reads the resulting
trajectory — the exact statevector order parameter as the controlled output, the
sampled estimate as the feedback signal — and reports how well the loop tracked
the set-point.

It is a local **software-in-the-loop** assessment: not provider-prepared
dynamic-circuit evidence and not live closed-loop QPU evidence. Hardware
execution is refused fail-closed without an explicit live ticket.

## Response classification

`analyse_closed_loop_response` returns a `ResponseClass` and a
`ControlPerformance`:

| verdict | meaning |
|---|---|
| `converged` | the response settles inside the tolerance band with small steady-state error |
| `limit_cycle` | the response oscillates persistently around the set-point without settling |
| `diverged` | the steady-state error is worse than the early-window error |
| `unsettled` | still transient at the end of the horizon |

`ControlPerformance` carries the steady-state error, settling round, overshoot
(relative to the initial gap), integral absolute error, late-window oscillation
amplitude, and trailing error sign-change count.

```python
from scpn_quantum_control.control import analyse_closed_loop_response

verdict, performance = analyse_closed_loop_response(response, target=0.75, tolerance=0.05)
```

## Execution policy

`evaluate_closed_loop_policy` is fail-closed: it authorises a simulation run by
default, and authorises hardware only when the policy sets `allow_hardware`,
carries a non-empty `live_ticket`, and the requested backend is on the
allow-list. The round budget is enforced in both modes.

```python
from scpn_quantum_control.control import ClosedLoopExecutionPolicy, evaluate_closed_loop_policy

policy = ClosedLoopExecutionPolicy()                       # simulation-only
decision = evaluate_closed_loop_policy(policy, requested_rounds=32)
decision.mode      # ExecutionMode.SIMULATION
```

## Replayable run

`run_closed_loop_control` runs the controller under the policy, classifies the
response, and returns a deterministic `ClosedLoopControlEvidence` record (same
seed → same trajectory) with a claim boundary in its provenance.

```python
from scpn_quantum_control.control import run_closed_loop_control

evidence = run_closed_loop_control(controller, n_rounds=32, seed=0)
evidence.classification      # ResponseClass verdict
evidence.performance         # control-theoretic metrics
evidence.decision.mode       # simulation unless a live ticket authorises hardware
```

## Claim boundary

The evidence is software-in-the-loop only: the controlled output is the exact
statevector order parameter and the feedback signal is the finite-shot estimate.
It is not provider-prepared dynamic-circuit evidence, not a latency measurement,
and not live closed-loop QPU evidence.
