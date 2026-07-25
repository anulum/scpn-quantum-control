# Governed multi-ecosystem route matrix (BL-52)

This page is the operator-facing guide for the **fail-closed multi-ecosystem
route matrix** productised under BL-52. It answers: *which differentiable route
IDs exist, what their closure status is, and why alternatives were rejected* —
without inventing green support for blank cells.

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

### Route ID taxonomy

Route identifiers use `family:ecosystem.or.surface` keys:

| Family | Examples |
|---|---|
| `transform` | `transform:native.grad_vmap`, `transform:unsupported.complex_objective` |
| `adapter` | `adapter:jax.value_and_grad_local`, `adapter:torch.func_local`, `adapter:pennylane.local_default_qubit` |
| `compiler` | `compiler:mlir_enzyme.bounded_kernels`, `compiler:catalyst.qjit_vmap` |
| `rust` | `rust:program_ad.static_registry_replay`, `rust:program_ad.dynamic_axes` |
| `provider` | `provider:hardware.gradient_live` |
| `competitor_boundary` | `competitor:differentiation_interface.silent_wrong_grads`, `competitor:catalyst.no_broadcast_adaptive_shots` |

BL-70 adds two explicit SSGF latent-geometry transform cells:

- `transform:ssgf.latent_finite_difference` is supported for bounded local
  simulation of the complete nonlinear `z -> softplus(W) -> H -> C` path.
- `transform:ssgf.latent_parameter_shift` is a permanent boundary. A circuit
  parameter-shift is not directly `dC/dz` through the nonlinear latent map.

## Competitor boundary fixtures

Competitor rows are first-class catalogue cells, not prose footnotes:

- **DifferentiationInterface.jl / ReverseDiff compiled tapes** — documented silent
  wrong-gradient class; SCPN refuses the same silent degradation pattern.
- **Catalyst qjit + vmap / no-broadcast adaptive finite-shot** — documented
  batching / trainability boundaries, recorded as `permanent_boundary` rather
  than missing rows.

## What this surface does *not* do

- Execute gradients or submit hardware jobs.
- Promote performance, category-leadership, or universal-transform claims.
- Fill blank cells by inventing backends (that is forbidden; use
  `permanent_boundary` or `implementation_path` instead).
- Replace the generated planner/support-matrix page — this is the unified
  route-ID product layer that composes those assets.

## Bounded product status (BL-52)

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
