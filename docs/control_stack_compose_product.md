# Compose existing control/* stack product (BL-67)

Versioned **ownership map** and **typed adapter ports** over ambient production
`control/*` modules so co-design (BL-33) does **not** invent a second stack.
Refuse evaluate/run without `ClosedLoopExecutionPolicy`. Never invent-green PCS.

Module: `scpn_quantum_control.control_stack_compose_product`

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
| `qaoa_mpc` | optional MPC path (BL-58 residual) |
| `adaptive_branching` | readiness tables |
| `cosimulation_quantum_classical` | partition bridge (S67.4 residual) |
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

## Residuals (honest)

- **S67.3** — full QAOA-MPC / pulsed BL-58 compose
- **S67.4** — full cosimulation partition bridge depth
- **S67.6** — amend BL-33 architecture doc to “ports over adapters”

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl67_compose_existing_control_stack.md`
- Ambient: `control.realtime_feedback`, `control.closed_loop_analysis`, `control.realtime_runtime`
- BL-47 hardware-safe execution posture

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
