# Whole-program AD product surface

Versioned **frontend → IR → adjoint/replay product** map for whole-program AD:
public journeys, layered architecture map, support badges, and dry-run posture.
Ambient `whole_program_*` / `program_ad_*` workbench modules remain experimental
under the API-stability policy (not a frozen SemVer mega-contract).

Module: `scpn_quantum_control.whole_program_ad_product`

## Rules

| Rule | Behaviour |
|---|---|
| Default journey | `frontend_compile_dry_run` |
| Dry-run | Structured allowed plan; no QPU submission |
| Hardware request | Refused by the no-submit compute policy |
| Unsupported frontend execute | Refused with an unsuitable-scenario pointer |
| Unverified polyglot certificate | Refused while the certificate subset is residual |
| Incomplete edge/WASM route | Refused while edge routing is residual |
| Stability | `experimental_workbench` under the API-stability policy |
| Blank/unknown journey | Fail closed |

Claim boundary:

> Whole-program AD product surface only; catalogues public journeys and layered
> architecture map; ambient whole_program_*/program_ad_* workbench is not a frozen
> SemVer mega-contract; unsupported frontend cases fail closed toward the
> unsuitable-scenario registry; polyglot parity certificates and edge/WASM routing
> remain residual; dry-run journeys refuse unsupported hardware and execution
> claims; does not replace full IR/adjoint engines

## Public API

```python
from scpn_quantum_control.whole_program_ad_product import (
    assert_whole_program_ad_product_integrity,
    build_whole_program_ad_product_registry,
    dry_run_whole_program_ad_journey,
    list_whole_program_ad_journey_ids,
    map_whole_program_ad_architecture_layers,
    map_whole_program_ad_public_surfaces,
)

assert "frontend_compile_dry_run" in list_whole_program_ad_journey_ids()
reg = assert_whole_program_ad_product_integrity(build_whole_program_ad_product_registry())
d = dry_run_whole_program_ad_journey("frontend_compile_dry_run")
assert d.allowed is True
assert d.steps_completed

refused = dry_run_whole_program_ad_journey(
    "frontend_compile_dry_run",
    request_hardware=True,
)
assert refused.allowed is False

layers = map_whole_program_ad_architecture_layers()
assert any(row["layer"] == "frontend" for row in layers)
```

## Closed vocabularies and immutable records

The product surface uses two closed string vocabularies. `SupportBadge`
distinguishes `local_dry_run`, `frontend_boundary`, `parity_boundary`,
`edge_boundary`, and `experimental_workbench`. `JourneyOutcome` contains only
`allowed_dry_run` and `refused`. These labels describe the posture of this
catalogue; they do not certify an underlying compiler, provider, or device.

`WholeProgramADJourney` is the immutable catalogue row. It records a stable
identifier, user-facing title and summary, owning module, support badge,
ordered dry-run steps, architecture layer, residual pointers, stability class,
inventory date, and the shared claim boundary. Construction rejects:

- blank identifiers, titles, summaries, module paths, layers, or dates;
- an unknown support badge or stability class;
- an empty step sequence or a blank step; and
- `allows_hardware=True` on a `local_dry_run` journey.

`WholeProgramADJourneyDecision` is the immutable result returned by the public
dry-run entrypoint. An allowed result must use `allowed_dry_run`, contain no
blockers, and report the catalogue steps it acknowledged. A refused result
must use `refused` and carry at least one non-blank blocker. The records are
frozen and slot-backed; `to_dict()` materialises tuple fields as JSON-ready
lists without mutating the original record.

## Catalogue access

### `list_whole_program_ad_journey_ids()`

Returns every canonical identifier in catalogue order. The result is a tuple,
so callers cannot mutate the catalogue through the returned value.

### `get_whole_program_ad_journey(journey_id)`

Returns the immutable row for a known, non-blank identifier. Leading and
trailing whitespace is ignored for lookup. Blank and unknown identifiers raise
`ValueError`; the function never fabricates a permissive default journey.

```python
from scpn_quantum_control.whole_program_ad_product import (
    get_whole_program_ad_journey,
)

frontend = get_whole_program_ad_journey("frontend_compile_dry_run")
assert frontend.support_badge == "local_dry_run"
assert frontend.allows_hardware is False
```

### `iter_whole_program_ad_journeys(...)`

Returns a stable tuple, optionally filtered by `support_badge`,
`architecture_layer`, or their intersection. Filters do not discover plugins,
load provider state, or widen the closed badge vocabulary.

```python
from scpn_quantum_control.whole_program_ad_product import (
    iter_whole_program_ad_journeys,
)

frontend_rows = iter_whole_program_ad_journeys(
    support_badge="local_dry_run",
    architecture_layer="frontend",
)
assert all(row.architecture_layer == "frontend" for row in frontend_rows)
```

## Dry-run decision policy

`dry_run_whole_program_ad_journey()` acknowledges the selected journey's
declared steps. It does not execute the objective, compile a programme, replay
an adjoint, contact a provider, or submit work to a QPU. Four explicit request
flags protect residual boundaries:

| Request | Decision |
|---|---|
| `request_hardware=True` | Refused through the no-submit compute boundary |
| `request_unsupported_frontend_execute=True` | Refused with the unsuitable-scenario registry pointer |
| `request_polyglot_cert=True` | Refused while the parity-certificate subset remains residual |
| `request_edge_wasm=True` | Refused while edge/WASM routing remains residual |

A boundary journey without its residual-completion flag may still return an
allowed dry-run. That means the caller may inspect the boundary map and its
steps; it does not mean the residual feature is complete. Refused decisions
deduplicate blockers and report no completed steps.

## Public-surface and architecture maps

`map_whole_program_ad_public_surfaces()` emits one row per unique owning module.
Each row identifies its architecture layer, support badge, journey IDs,
API-stability class, and the shared claim boundary. Duplicate module paths are
collapsed in first-catalogue order.

`map_whole_program_ad_architecture_layers()` groups modules into the fixed
`frontend → ir → adjoint → product → residual` order. Ambient IR ownership is
declared separately because not every IR module is a primary journey. An empty
IR ownership set omits the IR row rather than inventing content, and duplicate
module paths within a layer are not repeated.

## Registry construction and integrity

`build_whole_program_ad_product_registry()` returns a schema-tagged mapping
containing the journey rows, public-surface map, architecture map, counts,
default journey, policy note, and claim boundary. It is a deterministic local
inventory with no runtime discovery.

Use `assert_whole_program_ad_product_integrity(payload)` at serialisation or
transport boundaries. Omitting `payload` validates a newly built canonical
registry. The validator rejects:

- an absent, empty, or non-list journey collection;
- non-mapping rows, blank or duplicate identifiers, and unsupported badges;
- empty steps, missing architecture layers, or a missing unsuitable-scenario pointer;
- any journey that claims hardware permission;
- catalogue-set drift or loss of the default journey; and
- inconsistent blank, journey-count, or architecture metadata.

The return value is a normalised dictionary suitable for serialisation. A
`ValueError` means the caller must stop and repair the payload, not fall back to
an assumed journey. `RuntimeError` is reserved for an internally blank,
duplicate, or empty canonical catalogue.

## Operational non-effects

Importing or calling this module does **not**:

- execute frontend compilation, differentiation, adjoint replay, or an
  objective function;
- access credentials, networks, providers, hardware, or QPU queues;
- issue a polyglot parity certificate or complete residual edge/WASM routing;
- promote experimental workbench modules to a stable public contract;
- mutate registries, release metadata, evidence ledgers, or deployment state;
  or
- establish scientific, performance, parity, or product-readiness claims.

Those operations remain owned by their named governed surfaces and evidence
packages.

## Architecture layers

| Layer | Role |
|---|---|
| frontend | `compile_whole_program_frontend`, contracts, semantics |
| ir | result records, effect IR, primitive registry |
| adjoint | adjoint generation / replay dry-run |
| product | `whole_program_value_and_grad` product entry |
| residual | Polyglot parity certificates and edge/WASM routing boundaries |

## Bounded product status

Shipped: layered architecture map · public entrypoints catalogue · unsupported
frontend fail-closed product path through the unsuitable-scenario registry ·
documentation · API-stability pointers.

Open: polyglot parity certificate subset · edge/WASM routing · mass call-site
migration of ambient workbench exports.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
