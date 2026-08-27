# qpu_compute product surface

Fail-closed typed **compute plan** product between algorithms and HALs. Default
posture is **dry-run / no-submit**; would-live and hardware_enabled plans are
refused without inventing QPU spend. Composes `qpu_compute_types` kernels and
the `hardware_safe_execution` no-submit audit posture.

Module: `scpn_quantum_control.qpu_compute_product`

This page documents the product facade. It is not a provider tutorial and it
does not grant authority to submit jobs, enable hardware, spend credits, or
promote simulator output as hardware evidence.

## Contract discovery

Use the catalogue functions when a caller must discover supported choices
without importing private constants:

| Function | Contract |
|---|---|
| `list_plan_kind_ids()` | Returns all stable plan-kind identifiers in catalogue order. |
| `get_plan_kind(plan_kind_id)` | Returns one exact kind; blank and unknown identifiers raise `ValueError`. |
| `iter_plan_kinds(mode=...)` | Returns all kinds, or the stable subset matching one typed mode. |
| `list_supported_kernels()` | Returns the sorted kernel vocabulary owned by `qpu_compute_types`. |
| `list_supported_backend_policies()` | Returns the sorted backend-policy vocabulary owned by `qpu_compute_types`. |

The current plan modes are `dry_run`, `would_live`, and `ticketed_prep`.
Discovery is metadata-only: none of these functions constructs a provider
client or probes a device.

## Value objects

The facade exposes three immutable, slot-backed dataclasses:

- `ComputePlanKind` describes a versioned catalogue row and its defaults.
- `ComputePlanRecord` captures a constructed request before policy validation.
- `ComputePlanDecision` records the fail-closed outcome, blockers, audit id,
  and composed hardware-safety policy id.

Each object validates itself at construction and provides `to_dict()` for a
JSON-ready mapping. The mapping is a transport representation, not an approval
token. In particular, `allowed=True` means only that bounded planning may
continue; it never means that a QPU job ran.

## Rules

| Rule | Behaviour |
|---|---|
| Default kind | `dry_run_simulator` (`simulator_statevector`, hardware off) |
| Would-live / hardware_enabled | Refused on product surface |
| Ticketed prep | Requires non-empty ticket; plan-only (no live submit) |
| Kernels | `sync_witness`, `dla_parity`, `sync_dla` from `qpu_compute_types` |
| Blank/unknown plan kind | Fail closed |
| Audit | Secret-free; composes hardware-safety audit fields |

## Constructing a plan

`construct_compute_plan()` resolves a catalogue kind, applies documented
defaults, normalises the ticket and policy strings, and validates the kernel,
shot count, and value-object invariants. It does not decide whether execution
is allowed.

```python
from scpn_quantum_control.qpu_compute_product import construct_compute_plan

plan = construct_compute_plan(
    "dry_run_simulator",
    kernel="sync_witness",
    shots=256,
)
assert plan.mode == "dry_run"
assert plan.backend_policy == "simulator_statevector"
assert plan.hardware_enabled is False
assert plan.no_submit is True
```

Invalid kernels, blank identifiers, non-positive shot counts, and malformed
ticket values raise `ValueError`. An unsupported but non-blank backend policy
can be represented in the intermediate record so that the dry-run validator
can return a structured refusal with an auditable blocker.

## Validating a dry run

`dry_run_compute_plan()` constructs the record and composes the
hardware-safe policy. Outcomes are deliberately asymmetric:

| Request | Result |
|---|---|
| Default `dry_run_simulator` | `allowed_plan`, provided kernel and backend policy are supported. |
| `live_would_submit` | Refused; the product surface remains no-submit. |
| Any `hardware_enabled=True` override | Refused. |
| `ticketed_prep_plan` without a ticket | Refused. |
| `ticketed_prep_plan` with a non-empty ticket | Planning allowed; still no live submission. |
| Unsupported backend policy | Refused with a structured blocker. |

Blockers are de-duplicated while preserving their first-seen order. The
deterministic `audit_id` identifies the plan shape and outcome; it is not a
provider job id.

## Audit records

`audit_compute_plan_decision()` emits a secret-free dictionary under schema
`qpu_compute_product_audit.v2`. It includes the decision, claim boundary,
`contains_secrets=False`, and—when a hardware-safety policy id is present—a nested
hardware-safe audit record. Ticketed audit composition uses a fixed placeholder
and never copies the caller's ticket into the audit payload.

Treat audit dictionaries as planning evidence. They do not prove provider
availability, queue admission, device calibration, job submission, execution,
or scientific validity.

## Registry and integrity

`build_qpu_compute_product_registry()` produces the complete serialisable
catalogue under schema `qpu_compute_product.v2`. The registry reports plan-kind,
dry-run, and no-submit counts plus the supported kernel and backend vocabularies.

Always pass stored or transported registries through
`assert_qpu_compute_product_integrity()`. Validation fails closed on:

- a stale or unknown registry schema;
- a missing or empty `plan_kinds` list;
- non-mapping, blank, duplicate, missing, or extra catalogue rows;
- invalid modes or non-boolean `no_submit` values;
- any row that relaxes the product-wide no-submit contract;
- drift in `plan_kind_count` or `blank_entry_count`; or
- a default row that enables hardware or disables no-submit.

Claim boundary:

> qpu_compute product only; default posture is dry-run / no-submit; would_live
> and hardware_enabled plans are refused without owner gate; composes
> qpu_compute_types kernels and hardware-safe audit posture; never
> executes QPU jobs or invents hardware results

## Public API

```python
from scpn_quantum_control.qpu_compute_product import (
    assert_qpu_compute_product_integrity,
    audit_compute_plan_decision,
    build_qpu_compute_product_registry,
    dry_run_compute_plan,
    list_plan_kind_ids,
)

assert "dry_run_simulator" in list_plan_kind_ids()
reg = assert_qpu_compute_product_integrity(build_qpu_compute_product_registry())
d = dry_run_compute_plan("dry_run_simulator", kernel="sync_dla", shots=64)
assert d.allowed is True
assert d.outcome == "allowed_plan"

live = dry_run_compute_plan("live_would_submit")
assert live.allowed is False
audit = audit_compute_plan_decision(live)
assert audit["contains_secrets"] is False
```

The public constants `QPU_COMPUTE_PRODUCT_SCHEMA` and
`QPU_COMPUTE_PRODUCT_CLAIM_BOUNDARY`, type aliases `PlanMode` and
`ValidationOutcome`, all three value objects, and every discovery,
construction, validation, audit, registry, and integrity function are exported
through `__all__`.

## Failure handling

- Catch `ValueError` for caller-controlled identifiers, dimensions, and
  transported registry drift.
- Treat a returned `refused` decision as the expected policy result, not as an
  exception to bypass.
- Do not retry a refusal against a live provider. Change the plan only through
  the separately governed owner/hardware workflow.
- Treat `RuntimeError` from internal catalogue construction as repository
  corruption requiring maintainer repair.

## Operational non-effects

Calling this module does not read credentials, access the network, initialise a
provider SDK, inspect live rate tables, reserve capacity, submit or cancel jobs,
mutate a registry, write an evidence pack, or spend credits. It also does not
replace the low-level simulator runtime or the provider-neutral HAL. Those
boundaries require their own contracts and approvals.

## Bounded product status

Shipped: typed inventory, public plan-kind discovery and validation, the
dry-run decision path, composed hardware-safety audits, and this operator guide.

Open: mass algorithm call-site migration onto this layer · live HAL wiring ·
full runtime simulator integration in the product façade (existing
`qpu_compute_runtime` remains the low-level simulator path).

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
