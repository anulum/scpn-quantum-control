# Unsuitable-scenario + anti-silent-wrong registry (BL-53)

This page is the operator-facing guide for the **negative-space governance**
product under BL-53: a versioned catalogue of scenarios that must **fail closed**
rather than silently produce wrong gradients.

Related surfaces:

- Multi-ecosystem route matrix (BL-52): [Governed route matrix](governed_route_matrix.md)
- Generated planner/support matrix: [Differentiable Support Matrix](differentiable_support_matrix.md)
- Full API map: [Differentiable API](differentiable_api.md)
- Module: `scpn_quantum_control.unsuitable_scenario_registry`

## Why this exists

Silent-wrong reverse-mode classes (for example DifferentiationInterface.jl
compiled tapes under value-dependent control flow) are a known industry hazard.
SCPN publishes refuse paths as first-class catalogue entries with reasons,
evidence pointers, and deep links to BL-52 route IDs — the opposite of inventing
green support.

## How to read the registry

Every entry has:

| Field | Meaning |
|---|---|
| `scenario_id` | Stable key (`unsuitable:…` or `anti_silent:…`) |
| `kind` | `unsuitable_scenario` or `anti_silent_wrong` |
| `trigger` | Condition that must refuse |
| `expected_outcome` | Refuse class (`raise_value_error`, `permanent_boundary`, …) |
| `expected_error` | Error / boundary token |
| `reason` | Non-empty human-readable refusal reason |
| `related_route_ids` | Optional BL-52 governed route IDs |

There are **no blank entries** and **no green probes**. Unknown scenario IDs
either raise or, under `unknown_policy="boundary"`, synthesise a refuse row
prefixed with `unknown:`.

Claim boundary:

> unsuitable-scenario and anti-silent-wrong registry only; entries document
> explicit refuse paths and competitor failure modes, never invent gradient
> success, hardware execution, or silent-tape recovery claims

## Public API

```python
from scpn_quantum_control.unsuitable_scenario_registry import (
    assert_unsuitable_registry_integrity,
    build_unsuitable_scenario_registry,
    list_unsuitable_scenario_ids,
    probe_unsuitable_scenario,
)

registry = assert_unsuitable_registry_integrity(
    build_unsuitable_scenario_registry()
)

# Known unsuitable scenario — always refuses
result = probe_unsuitable_scenario(
    "unsuitable:complex.objective_without_wirtinger"
)
assert result.refused is True
assert result.selected.reason

# Competitor anti-silent-wrong fixture
anti = probe_unsuitable_scenario(
    "anti_silent:differentiation_interface.compiled_tape"
)
assert anti.selected.kind == "anti_silent_wrong"

# Unknown IDs fail closed
try:
    probe_unsuitable_scenario("no.such.scenario")
except ValueError:
    pass
boundary = probe_unsuitable_scenario(
    "no.such.scenario", unknown_policy="boundary"
)
assert boundary.refused is True

# RL research without preregistration also fails closed
rl = probe_unsuitable_scenario(
    "unsuitable:rl.research_without_preregistration"
)
assert rl.refused is True
```

## Seed catalogue (bounded product)

Unsuitable scenarios (local refuse paths):

- Complex objective without Wirtinger contract
- Hardware gradient without owner ticket
- Rust Program AD dynamic axes on static replay
- Unregistered torch fullgraph compile
- PennyLane hardware-plugin gradients
- RL-adjacent research without preregistration, fixed seeds, and budgets

Anti-silent-wrong / competitor fixtures:

- DifferentiationInterface.jl ReverseDiff compiled-tape silent wrong grads
- Catalyst qjit + vmap over quantum instructions
- Catalyst no-broadcast adaptive finite-shot trainability

## Bounded product status (BL-53)

Shipped in this slice:

- Versioned schema + catalogue (no blanks)
- Pure `probe_unsuitable_scenario` / lookup APIs
- Competitor anti-silent fixtures with citations + BL-52 deep links
- Operator guide (this page)
- Real-surface tests

Still open by design:

- S53.2 executable CI probe runner job
- S53.4 require new gradient methods to declare scenario coverage

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
