# Compose existing control/* stack product

Versioned **ownership map** and **typed, executable adapter ports** over ambient production
`control/*` modules so quantum-classical co-design does **not** invent a second stack.
Refuse evaluate/run without `ClosedLoopExecutionPolicy`. Never invent-green PCS.

Modules: `scpn_quantum_control.control_stack_compose_product` and
`scpn_quantum_control.control_stack_runtime_adapters`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `control_stack_compose_product.v1` |
| Runtime-adapter result schema | `control_stack_runtime_adapters.v2` |
| Ownership | Ambient `control/*`, cosimulation, hardware feedback ports |
| Unknown module/port | Fail closed |
| Evaluate without policy | Refuse |
| PCS invent-green | Refuse |
| `realtime_runtime` rewrite | Forbidden |
| Live hardware | Ambient hardware-safe execution policy + ticket |

## Ownership catalogue (ambient, no rewrite)

| module_id | path role |
|---|---|
| `realtime_feedback` | sync feedback controller |
| `realtime_runtime` | production runtime/SLA (no rewrite) |
| `closed_loop_analysis` | ExecutionPolicy + telemetry |
| `qaoa_mpc` | executable abstract QAOA-MPC adapter; pulse execution delegated to the optional pulse-execution boundary |
| `adaptive_branching` | readiness tables |
| `cosimulation_quantum_classical` | executable local partition bridge + mapped telemetry |
| `hardware_feedback_dryrun` | hardware-side adapter port |
| `execution_policy_gate` | policy compose gate |

## Quick start

```python
from scpn_quantum_control.control_stack_compose_product import (
    assert_control_stack_compose_product_integrity,
    build_control_stack_compose_product_registry,
    decide_control_compose_path,
    materialise_demo_closed_loop_telemetry_probe,
)

reg = assert_control_stack_compose_product_integrity(
    build_control_stack_compose_product_registry()
)
assert reg["invent_green_pcs_policy"] is False

path = decide_control_compose_path("realtime_feedback", policy_present=False)
assert path.allowed is False  # need ClosedLoopExecutionPolicy

probe = materialise_demo_closed_loop_telemetry_probe()
assert probe.mode == "simulation"
assert probe.invent_green_pcs is False
```

Executable adapters require an actual policy object, not a Boolean assertion:

```python
import numpy as np

from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control_stack_runtime_adapters import run_qaoa_mpc_adapter

controller = QAOA_MPC(
    np.array([[1.0]]), np.array([0.5]), horizon=1, p_layers=1
)
result = run_qaoa_mpc_adapter(
    controller, policy=ClosedLoopExecutionPolicy(), seed=7
)
assert result.decision.mode.value == "simulation"
```

### Verified constrained co-design proposals

`ConstrainedCodesignSession` holds the last accepted plan of a constrained
co-design problem: an objective, inequality constraints `g(plan, d) <= 0`, and
the box `[lower, upper]` that is the tested domain. An optimiser is adapted
through `ConstrainedOptimiserPort.optimize(start) -> OptimiserOutcome`.

A proposal is adopted only when all of the following hold:
- the optimiser converged within `evaluation_budget`;
- the plan lies in the domain;
- the session recomputes every constraint for the nominal case and for each
  held-out disturbance (which the optimiser never sees) and finds it within
  `tolerance`.

Each `ConstrainedProposal` records the recomputed objective, every constraint
value, the domain, the held-out disturbances checked and the tolerance. The
tolerance is in the constraints' own units and defaults to 0, so any slack
(for example for an optimiser that lands on an active constraint to rounding)
is an explicit caller choice for that constraint's scale. Every case must
return the nominal number of constraint values. The evaluation budget is
checked against the count the optimiser port reports. A port that raises
`NotImplementedError` or `RecursionError` fails the call instead of producing
an outcome. Its status is one of `CODESIGN_PROPOSAL_STATUSES`.
`infeasible`, `nonconvergent`, `out_of_domain` and `over_budget` are outcomes,
not exceptions: the previous accepted plan stays in force. Each proposal is
first authorised by `ClosedLoopExecutionPolicy` in simulation mode. A proposal
never actuates hardware (`hardware_actuation` is always `False`). An accepted
plan claims feasibility only in the checked cases and inside the tested
domain.

```python
import numpy as np

from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.control_stack_runtime_adapters import (
    ConstrainedCodesignSession,
    OptimiserOutcome,
)


class Halfway:
    def optimize(self, start):
        return OptimiserOutcome(np.array([0.4, 0.4]), True, 1)


session = ConstrainedCodesignSession(
    lambda plan: float(np.sum((plan - 0.9) ** 2)),
    lambda plan, disturbance: np.array([plan[0] + plan[1] - 1.0]),
    [0.0, 0.0],
    [1.0, 1.0],
    policy=ClosedLoopExecutionPolicy(),
)
proposal = session.propose(Halfway())
assert proposal.status == "accepted" and session.current_plan == (0.4, 0.4)
```

## Public API contracts

### Ownership and adapter discovery

| API | Contract |
|---|---|
| `list_ownership_module_ids()` | Return stable ambient control-owner identifiers in catalogue order. |
| `list_adapter_port_ids()` | Return stable executable/policy adapter identifiers. |
| `get_ownership_row(module_id)` | Resolve one immutable owner row; blank and unknown identifiers raise `ValueError`. |
| `get_adapter_port(port_id)` | Resolve one typed adapter contract; blank and unknown identifiers fail closed. |
| `iter_ownership_rows(support_posture=None)` | Return every owner or a stable support-posture-filtered tuple. |

`OwnershipRow` validates module identity, ambient ownership, adapter link,
support posture, rewrite prohibition, and claim boundary. `AdapterPortRow`
validates its ambient modules, hardware-safety pointer, execution-policy
requirement, and false invent-green PCS flag. Both records serialize through
`to_dict()`.

### Eligibility and telemetry

| API | Contract |
|---|---|
| `decide_control_compose_path(port_id, ...)` | Require `ClosedLoopExecutionPolicy` presence and return explicit blockers for invented PCS integration or realtime-runtime rewrites. |
| `materialise_closed_loop_telemetry_probe(...)` | Delegate authorization to ambient `evaluate_closed_loop_policy` and return validated simulation/hardware-gated telemetry. |
| `materialise_demo_closed_loop_telemetry_probe()` | Materialise the deterministic simulation-only, one-round demonstration. |

`PathEligibilityDecision` keeps outcome, permission, reason, and blockers
consistent. `MaterialisedClosedLoopTelemetryProbe` validates authorization,
mode, requested rounds, ticket presence, and the false invent-green PCS flag.
The product does not treat a Boolean policy assertion as an executable policy
object; runtime adapters require the actual `ClosedLoopExecutionPolicy`.

### Registry evidence and integrity

| API | Contract |
|---|---|
| `map_control_stack_compose_public_surfaces()` | Emit deterministic product and ambient control-surface descriptors with rewrite boundaries. |
| `build_control_stack_compose_product_registry()` | Build schema-tagged ownership, adapters, surfaces, policy flags, counts, and residual-work evidence. |
| `assert_control_stack_compose_product_integrity(payload=None)` | Reject empty, malformed, blank, duplicate, count-drifted, policy-gate-missing, ownership/port-drifted, invent-green, or rewrite-permissive state. |

These APIs compose the existing control stack. They do not create a second
runtime, execute provider pulses, rewrite realtime ownership, bypass the
hardware-safe execution policy, or promote PCS claims.

## Completed boundaries

- The abstract QAOA-MPC controller is adapted; pulse execution is explicitly
  descoped and fails closed to `pulse-boundary/runtime-adapter`.
- The existing mean-field co-simulation is policy-gated and its
  partition outputs are mapped into compact telemetry without changing ownership.
- Quantum-classical co-design specifies ports over the compose adapters.

All executable adapters are local-simulator only. Even a hardware-authorising
policy is refused because this compose product does not own provider submission.
Serialized adapter results reject stale schemas and altered claim boundaries.

## Related

- Ambient: `control.realtime_feedback`, `control.closed_loop_analysis`, `control.realtime_runtime`
- Hardware-safe execution posture

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
