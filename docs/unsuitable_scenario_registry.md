# Unsuitable-scenario + anti-silent-wrong registry

This page is the operator-facing guide for the **negative-space governance**
surface: a versioned catalogue of scenarios that must **fail closed** rather
than silently produce wrong gradients.

Related surfaces:

- Multi-ecosystem route matrix: [Governed route matrix](governed_route_matrix.md)
- Generated planner/support matrix: [Differentiable Support Matrix](differentiable_support_matrix.md)
- Full API map: [Differentiable API](differentiable_api.md)
- Module: `scpn_quantum_control.unsuitable_scenario_registry`

## Why this exists

Silent-wrong reverse-mode classes (for example DifferentiationInterface.jl
compiled tapes under value-dependent control flow) are a known industry hazard.
SCPN publishes refuse paths as first-class catalogue entries with reasons,
evidence pointers, and deep links to governed route IDs — the opposite of
inventing green support.

This registry is a decision-support surface, not a gradient executor. Reading,
filtering, serialising, or probing it cannot submit provider work, access a
QPU, mutate a circuit, run a benchmark, or certify that a gradient method is
safe. A refusal explains a known boundary; it does not diagnose arbitrary
caller code.

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
| `related_route_ids` | Optional governed route-matrix identifiers |
| `test_id` | Stable test label expected to exercise the refusal |
| `citation` | Optional literature or competitor evidence label |
| `claim_boundary` | Non-promotional statement attached to every row |

There are **no blank entries** and **no green probes**. Unknown scenario IDs
either raise or, under `unknown_policy="boundary"`, synthesise a refuse row
prefixed with `unknown:`.

Claim boundary:

> unsuitable-scenario and anti-silent-wrong registry only; entries document
> explicit refuse paths and competitor failure modes, never invent gradient
> success, hardware execution, or silent-tape recovery claims

## Record and result layers

`UnsuitableScenarioRecord` is the immutable catalogue layer. Construction
rejects blank identifiers, triggers, errors, reasons, evidence labels, and
route identifiers. It also rejects classification values outside the closed
`ScenarioKind` and `RefuseOutcome` vocabularies. `to_dict()` converts tuple
fields to lists so the result is JSON-ready without weakening the stored
record.

`ScenarioProbeResult` is the immutable decision layer. Its `refused` field is
required to be `True`; attempting to construct a green result raises
`ValueError`. The selected record, operator message, deterministic notes, and
shared claim boundary remain available through `to_dict()`.

The registry payload is the aggregate layer. It carries the schema identifier,
claim boundary, category counts, zero-blank count, and serialised rows. The
catalogue order is stable and is also the order returned by
`list_unsuitable_scenario_ids()` and `iter_unsuitable_scenarios()`.

## Lookup and filtering

Use `get_unsuitable_scenario()` when the identifier must already exist. Blank
or unknown input raises `ValueError`; lookup never manufactures a matching
row. Use `iter_unsuitable_scenarios()` to select by `kind`,
`expected_outcome`, both, or neither. It returns an immutable tuple and
preserves catalogue order.

`probe_unsuitable_scenario()` has two explicit unknown-ID policies:

- `unknown_policy="raise"` is the default and rejects the identifier.
- `unknown_policy="boundary"` returns an always-refused synthetic row whose
  identifier starts with `unknown:`.

Known anti-silent-wrong rows add a classification note. Known rows with route
links add a deterministic `related_route_ids=` note; rows without links do not
invent one.

## Integrity validation

Call `assert_unsuitable_registry_integrity()` with no argument to build and
validate the canonical payload, or pass a payload received through another
local boundary. Validation requires a non-empty scenario list, mapping-shaped
rows, non-empty identifiers, one of the two closed kinds, reasons and expected
errors, a zero `blank_entry_count`, and an exact `scenario_count`.

The validator deliberately does not treat a malformed row as partial success.
It reports the first structural failure and refuses the payload. This makes it
suitable as a pre-serialisation or evidence-ingest check, while schema
validation and provenance checks remain separate caller responsibilities.

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

## Bounded product status

Shipped:

- Versioned schema + catalogue (no blanks)
- Pure `probe_unsuitable_scenario` / lookup APIs
- Competitor anti-silent fixtures with citations and governed-route deep links
- Operator guide (this page)
- Real-surface tests

Outside this surface:

- Executing gradient implementations or provider workloads
- Automatically declaring new gradient methods safe or supported
- Replacing caller-side schema, provenance, or evidence validation

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
