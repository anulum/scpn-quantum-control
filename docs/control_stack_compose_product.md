# Compose existing control/* stack product (BL-67)

Versioned **ownership map** and **typed, executable adapter ports** over ambient production
`control/*` modules so co-design (BL-33) does **not** invent a second stack.
Refuse evaluate/run without `ClosedLoopExecutionPolicy`. Never invent-green PCS.

Modules: `scpn_quantum_control.control_stack_compose_product` and
`scpn_quantum_control.control_stack_runtime_adapters`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `control_stack_compose_product.v1` |
| Ownership | Ambient `control/*`, cosimulation, hardware feedback ports |
| Unknown module/port | Fail closed |
| Evaluate without policy | Refuse |
| PCS invent-green | Refuse |
| `realtime_runtime` rewrite | Forbidden |
| Live hardware | Ambient policy + ticket (BL-47 compose) |

## Ownership catalogue (ambient, no rewrite)

| module_id | path role |
|---|---|
| `realtime_feedback` | sync feedback controller |
| `realtime_runtime` | production runtime/SLA (no rewrite) |
| `closed_loop_analysis` | ExecutionPolicy + telemetry |
| `qaoa_mpc` | executable abstract QAOA-MPC adapter; pulse execution delegated to BL-58 |
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

## Completed boundaries

- **S67.3** — abstract QAOA-MPC is adapted; pulse execution is explicitly
  descoped and fails closed to `BL-58/S58.4`.
- **S67.4** — the existing mean-field co-simulation is policy-gated and its
  partition outputs are mapped into compact telemetry without changing ownership.
- **S67.6** — BL-33 now specifies ports over BL-67 adapters.

All executable adapters are local-simulator only. Even a hardware-authorising
policy is refused because BL-67 does not own provider submission.

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl67_compose_existing_control_stack.md`
- Ambient: `control.realtime_feedback`, `control.closed_loop_analysis`, `control.realtime_runtime`
- BL-47 hardware-safe execution posture

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
