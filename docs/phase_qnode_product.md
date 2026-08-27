# Phase-QNode product surface

Versioned **primary quantum programming product** map for Phase-QNode: public
journeys, support badges, and dry-run posture. Ambient `phase/qnode_*` workbench
modules remain experimental under BL-97 honesty (not a frozen SemVer mega-contract).

Module: `scpn_quantum_control.phase_qnode_product`

## Rules

| Rule | Behaviour |
|---|---|
| Default journey | `build_differentiate_dry_run` |
| Dry-run | Structured allowed plan; no QPU submission |
| Hardware request | Refused (BL-47/BL-95 no-submit posture) |
| Stability | `experimental_workbench` (BL-97) |
| Blank/unknown journey | Fail closed |

Claim boundary:

> Phase-QNode product surface only; catalogues public journeys and support badges;
> ambient phase/qnode_* workbench is not a frozen SemVer mega-contract (BL-97);
> dry-run journeys refuse invent-green hardware/QPU spend (BL-47/BL-95); does not
> replace full circuit engines

## Public API

```python
from scpn_quantum_control.phase_qnode_product import (
    assert_phase_qnode_product_integrity,
    build_phase_qnode_product_registry,
    dry_run_phase_qnode_journey,
    list_phase_qnode_journey_ids,
    map_phase_qnode_public_surfaces,
)

assert "build_differentiate_dry_run" in list_phase_qnode_journey_ids()
reg = assert_phase_qnode_product_integrity(build_phase_qnode_product_registry())
d = dry_run_phase_qnode_journey("build_differentiate_dry_run")
assert d.allowed is True
assert d.steps_completed

refused = dry_run_phase_qnode_journey(
    "build_differentiate_dry_run",
    request_hardware=True,
)
assert refused.allowed is False
```

## Closed vocabularies and immutable records

`SupportBadge` distinguishes `local_dry_run`, `simulator_local`,
`hardware_boundary`, and `experimental_workbench`. `JourneyOutcome` contains
only `allowed_dry_run` and `refused`. These labels describe catalogue posture,
not provider or hardware validation.

`PhaseQNodeJourney` is a frozen, slot-backed catalogue row carrying identity,
title, summary, owning module, badge, ordered steps, hardware permission,
stability class, inventory date, and the shared claim boundary. Construction
rejects blank fields, unknown badges, empty or blank steps, and any
`local_dry_run` row with hardware permission.

`PhaseQNodeJourneyDecision` binds a journey to an allowed or refused outcome.
Allowed decisions use `allowed_dry_run` with no blockers; refused decisions use
`refused` with at least one non-blank blocker. Both records expose JSON-ready
`to_dict()` mappings without mutating their tuple-backed state.

## Catalogue access and dry-run policy

`list_phase_qnode_journey_ids()` returns the stable canonical order.
`get_phase_qnode_journey()` rejects blank and unknown IDs instead of fabricating
a default. `iter_phase_qnode_journeys()` optionally filters by badge and returns
an immutable tuple.

`dry_run_phase_qnode_journey()` acknowledges the catalogue steps as a plan. A
hardware request, or a journey that would claim hardware permission, returns a
refused decision under BL-47/BL-95. An allowed result means only that the local
plan can be inspected; it does not build a circuit, differentiate an objective,
submit a QPU job, or spend provider credit.

## Public-surface map and registry integrity

`map_phase_qnode_public_surfaces()` emits one row per unique owning module with
its badge, stability class, journey IDs, role, and claim boundary. Duplicate
module paths are collapsed in catalogue order.

`build_phase_qnode_product_registry()` returns the schema, counts, default
journey, public map, catalogue rows, policy note, and zero blank-entry count.
Use `assert_phase_qnode_product_integrity()` at storage and transport
boundaries. It rejects empty/non-mapping rows, blank or duplicate IDs, unknown
badges, empty steps, invented hardware permission, missing default or required
journeys, catalogue drift, and inconsistent count metadata.

An integrity `ValueError` is a stop condition, not permission to assume support.
An internal blank, duplicate, or empty canonical catalogue raises
`RuntimeError` during catalogue-map construction.

## Operational non-effects

This module does not import or execute QNode frameworks, compile circuits,
evaluate gradients, contact providers, read credentials, submit hardware,
mutate registries, publish evidence, or promote workbench exports to a stable
contract. Those actions remain governed by their owning runtime and evidence
surfaces.

## Bounded product status

Shipped: public API map, journey catalogue, product-catalogue badge fields,
stability pointers, and documentation.

Open: full badge CI job, complete ≤15 minute notebook curriculum, and mass
deprecation application across all QNode exports.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
