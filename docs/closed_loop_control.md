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
The package-level [Control Scope Boundary](control_scope.md) also excludes
generic pulse-shape optimisation, provider-native pulse calibration, hardware
drift compensation, and lab-instrument control from this module.

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

## Latency budget

`measure_closed_loop_latency_budget` adds a no-submit latency gate around the
same controller contract. By default it measures each local feedback round with
`time.perf_counter_ns`; CI and replay fixtures may pass
`observed_round_latencies_s` to validate a stored profile without relying on
workstation timing.

```python
from scpn_quantum_control.control import (
    ClosedLoopLatencyBudget,
    measure_closed_loop_latency_budget,
)

budget = ClosedLoopLatencyBudget(
    max_round_latency_s=0.050,
    p95_round_latency_s=0.040,
    p99_round_latency_s=0.045,
    max_total_latency_s=1.500,
)
latency = measure_closed_loop_latency_budget(controller, 32, budget=budget, seed=0)
latency.passes
latency.to_dict()
```

The report records per-round samples, max/p95/p99/total latency, policy
authorisation, response classification, and blockers. A failed policy decision
or latency-budget breach is evidence, not a provider submission.

## Publication package

`build_closed_loop_publication_package` creates a structured scaffold for the
future paper/control campaign. It separates three evidence classes:

| evidence class | status boundary |
|---|---|
| `software_in_loop_simulation` | available from the local controller and latency gate |
| `provider_prepared_dynamic_circuit` | placeholder until backend capability and provider-preparation artefacts exist |
| `live_closed_loop_qpu` | blocked until a live ticket, allow-listed backend, raw counts, calibration snapshot, job IDs, and replay harness exist |

```python
from scpn_quantum_control.control import build_closed_loop_publication_package

package = build_closed_loop_publication_package(latency_report=latency)
package.to_dict()
package.to_markdown()
```

## Claim boundary

The evidence is software-in-the-loop only: the controlled output is the exact
statevector order parameter and the feedback signal is the finite-shot estimate.
The latency gate measures local controller/simulator wall-clock time only. It
is not provider-prepared dynamic-circuit evidence and not live closed-loop QPU
evidence.
