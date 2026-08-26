# Adjoint reversible simulator replay product

Versioned **reverse-mode adjoint-via-replay product** over ambient
`program_ad_adjoint` generation and executable step replay. Materialised local
scalar demos only; refuses mid-circuit measurement / irreversible invent-green,
Catalyst parity, and hardware adjoint.

Module: `scpn_quantum_control.adjoint_replay_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `adjoint_replay_product.v2` |
| Default surface | `reverse_adjoint_grad` |
| Mid-circuit measurement | Refused |
| Catalyst parity | Refused |
| Hardware adjoint | Refused |
| Blank/unknown surface | Fail closed |
| Full automatic checkpointing | Open capability; not claimed by this product |

Claim boundary:

> Adjoint reversible-replay product surface only; catalogues reversibility
> conditions, checkpoint policies, reverse-mode Program AD gradient, and
> executable adjoint step replay over ambient program_ad_adjoint; materialised
> local scalar demos only; refuses mid-circuit measurement / irreversible ops
> invent-green, Catalyst parity, and hardware adjoint; does not invent full
> automatic checkpointing, open-system reverse, or complete planner-registration
> coverage

## Public API

```python
from scpn_quantum_control.adjoint_replay_product import (
    assert_adjoint_replay_product_integrity,
    build_adjoint_replay_product_registry,
    build_checkpoint_policy,
    decide_adjoint_replay_path,
    list_adjoint_replay_surface_ids,
    materialise_demo_adjoint_replay_probe,
)

assert "reverse_adjoint_grad" in list_adjoint_replay_surface_ids()
reg = assert_adjoint_replay_product_integrity(
    build_adjoint_replay_product_registry()
)

allowed = decide_adjoint_replay_path(has_supported_unitary_ir=True)
assert allowed.allowed is True

refused = decide_adjoint_replay_path(has_mid_circuit_measurement=True)
assert refused.allowed is False

# Worked demo: f(x,y)=x^2+y^2 at [0.5, -0.25] => grad [1.0, -0.5]
probe = materialise_demo_adjoint_replay_probe()
assert abs(probe.adjoint_gradient[0] - 1.0) < 1e-9
assert abs(probe.replay_gradient[1] + 0.5) < 1e-9
assert probe.agreement_max_abs < 1e-9

policy = build_checkpoint_policy(schedule="every_k", interval_k=2)
assert policy.interval_k == 2
```

## Replay surface catalogue

| ID | Kind |
|---|---|
| `reversibility_conditions` | predicates |
| `checkpoint_policy` | checkpoint schedule |
| `reverse_adjoint_grad` | reverse adjoint grad |
| `executable_adjoint_replay` | step-stream replay |
| `irreversible_mid_circuit_refuse` | irreversible refuse |
| `catalyst_hardware_adjoint_refuse` | Catalyst/hardware refuse |

## Worked scalar example

For ``f(x, y) = x^2 + y^2`` at ``[0.5, -0.25]``:

- value = ``0.3125``
- true gradient = ``[1.0, -0.5]``
- ambient reverse adjoint and executable replay agree within ``1e-12``

## Bounded product status

Shipped: surface catalogue, reversibility and checkpoint-policy contracts,
reverse/replay materialised demo, irreversible/Catalyst/hardware refusal, and
documentation/API map.

Open: full checkpointed simulator campaign, extended validation harness,
memory/time report artefacts, planner-registration matrix, extended usage
guide, and open-system reverse.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
