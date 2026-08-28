# Governed multi-ecosystem route matrix

This page is the operator-facing guide for the **fail-closed multi-ecosystem
route matrix**. It answers: *which differentiable route IDs exist, what their
closure status is, and why alternatives were rejected* — without inventing
green support for blank cells.

Related surfaces:

- Generated transform / planner matrix: [Differentiable Support Matrix](differentiable_support_matrix.md)
- Full API map: [Differentiable API](differentiable_api.md)
- Module: `scpn_quantum_control.governed_route_matrix`

## How to read the matrix

Every catalogue cell has exactly one of:

| Status | Meaning |
|---|---|
| `supported` | Bounded local or documented adapter evidence exists. Not a hardware/provider/performance claim. |
| `permanent_boundary` | Explicitly refused (scientific, safety, or competitor-documented limitation). Never silently approximated. |
| `implementation_path` | Known gap with a real path; still not green. Callers must not treat it as supported. |

There are **no blank cells**. Unknown route IDs either raise or, under
`unknown_policy="boundary"`, resolve to a synthetic `permanent_boundary` row
prefixed with `unknown:`.

Claim boundary attached to every row:

> governed multi-ecosystem route matrix only; supported rows are local
> conformance or documented adapter evidence, permanent_boundary and
> implementation_path rows are fail-closed and never silently promoted to
> provider, hardware, compiler-performance, or category-leadership claims

## Public API

```python
from scpn_quantum_control.governed_route_matrix import (
    RouteCapability,
    assert_no_blank_matrix_cells,
    build_governed_route_matrix,
    explain_route,
    get_governed_route,
    list_governed_route_ids,
)

# Full serialisable matrix (schema governed_route_matrix.v1)
matrix = assert_no_blank_matrix_cells(build_governed_route_matrix())

# Explain one route under a capability context
explanation = explain_route(
    "transform:native.grad_vmap",
    RouteCapability(ecosystem="native", method="grad"),
)
assert explanation.selected.closure_status == "supported"
assert explanation.rejected  # deterministic rejected alternatives

# Adapter implementation-path cell (not invent-green)
impl = explain_route(
    "adapter:jax.provider_arbitrary_simulator",
    {"ecosystem": "jax"},
)
assert impl.selected.closure_status == "implementation_path"

# Unknown IDs fail closed
try:
    explain_route("no.such.route")
except ValueError:
    pass
boundary = explain_route("no.such.route", unknown_policy="boundary")
assert boundary.selected.closure_status == "permanent_boundary"
```

### Immutable records and capability context

`RouteCapability` normalises the caller's ecosystem and method to lower-case
labels and carries the finite-shot and hardware-policy flags. Blank ecosystem
or method values raise `ValueError`; they are never interpreted as permission.

`GovernedRouteRecord` is the frozen catalogue cell. It validates the closed
family and closure-status vocabularies, non-blank identity and summary, and
non-blank evidence and alternative pointers. Supported rows cannot carry a
closure reason. Every `permanent_boundary` or `implementation_path` row must
carry one, so a non-green cell always explains why it is closed.

`RouteExplanation` binds the requested ID, normalised capability, selected
record, rejected alternatives, and deterministic operator notes. Its
`to_dict()` method materialises tuple fields as JSON-ready lists while leaving
the slot-backed records immutable.

### Catalogue lookup and filtering

`list_governed_route_ids()` returns the stable catalogue order.
`get_governed_route(route_id)` returns the exact immutable row and rejects a
blank or unknown ID. `iter_governed_routes()` optionally filters by family,
closure status, or their intersection without discovery or network access.

```python
from scpn_quantum_control.governed_route_matrix import iter_governed_routes

adapter_gaps = iter_governed_routes(
    family="adapter",
    closure_status="implementation_path",
)
assert all(row.family == "adapter" for row in adapter_gaps)
```

### Explanation policy

`explain_route()` accepts a `RouteCapability`, a mapping, or no capability. A
missing capability becomes the bounded `native` / `auto` default. Other input
types raise `TypeError`. The `unknown_policy` vocabulary is closed to `raise`
and `boundary`; any other value raises `ValueError`.

Hardware permission never upgrades `provider:hardware.gradient_live` to
supported. The explanation only records whether hardware is disabled or that
owner-ticket evidence remains required. Likewise, `finite_shot=True` adds an
operator note for a supported transform but does not silently redirect it to a
finite-shot planner. Rejected alternatives are resolved from the catalogue;
missing pointers are skipped rather than converted into invented records.

### Matrix integrity

`build_governed_route_matrix()` derives every count from the canonical rows and
sets `blank_cell_count` to zero. Pass a transported payload to
`assert_no_blank_matrix_cells()` before consumption. It rejects non-list or
empty route collections, non-mapping rows, blank IDs, unsupported statuses,
duplicate cells, catalogue drift, and inconsistent per-status or blank-cell
counts. Omitting the payload validates a freshly built canonical matrix.

An integrity exception is a stop condition. It is not permission to fall back
to a guessed route or to promote an implementation path.

### Route ID taxonomy

Route identifiers use `family:ecosystem.or.surface` keys:

| Family | Examples |
|---|---|
| `transform` | `transform:native.grad_vmap`, `transform:unsupported.complex_objective` |
| `adapter` | `adapter:jax.value_and_grad_local`, `adapter:torch.func_local`, `adapter:pennylane.local_default_qubit`, `adapter:l16.local_indicator` |
| `compiler` | `compiler:mlir_enzyme.bounded_kernels`, `compiler:catalyst.qjit_vmap` |
| `rust` | `rust:program_ad.static_registry_replay`, `rust:program_ad.dynamic_axes` |
| `provider` | `provider:hardware.gradient_live` |
| `competitor_boundary` | `competitor:differentiation_interface.silent_wrong_grads`, `competitor:catalyst.no_broadcast_adaptive_shots` |

BL-70 adds two explicit SSGF latent-geometry transform cells:

- `transform:ssgf.latent_finite_difference` is supported for bounded local
  simulation of the complete nonlinear `z -> softplus(W) -> H -> C` path.
- `transform:ssgf.latent_parameter_shift` is a permanent boundary. A circuit
  parameter-shift is not directly `dC/dz` through the nonlinear latent map.

BL-85 adds two explicit L16 director adapter cells:

- `adapter:l16.local_indicator` supports bounded local exact-simulator
  indicator evaluation and heuristic BL-33 safety routing.
- `adapter:l16.autonomous_hardware_control` is a permanent boundary. The
  weighted composite is not a Lyapunov, PCS, or stability certificate and
  cannot authorise hardware or plant actuation.

## Competitor boundary fixtures

Competitor rows are first-class catalogue cells, not prose footnotes:

- **DifferentiationInterface.jl / ReverseDiff compiled tapes** — documented silent
  wrong-gradient class; SCPN refuses the same silent degradation pattern.
- **Catalyst qjit + vmap / no-broadcast adaptive finite-shot** — documented
  batching / trainability boundaries, recorded as `permanent_boundary` rather
  than missing rows.

## What this surface does *not* do

- Execute gradients or submit hardware jobs.
- Import or execute adapter, compiler, Rust, provider, or competitor runtimes.
- Read credentials, contact networks, mutate the catalogue, or write evidence.
- Promote performance, category-leadership, or universal-transform claims.
- Fill blank cells by inventing backends (that is forbidden; use
  `permanent_boundary` or `implementation_path` instead).
- Replace the generated planner/support-matrix page — this is the unified
  route-ID product layer that composes those assets.

## Bounded product status

Shipped in this slice:

- Unified route taxonomy + catalogue (no blanks)
- `explain_route` pure API with rejected alternatives
- Competitor boundary fixtures
- Operator how-to-read guide (this page)
- Real-surface tests and fail-closed unknown/blank behaviour

Still open by design (not falsely completed):

- Continuous CI drift gate vs capability manifest (S52.4)
- Scheduled refresh job (S52.5)
- Notebook / Sync Challenge baseline wiring to route IDs (S52.7)

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
