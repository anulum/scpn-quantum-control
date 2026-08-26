# QIR / CUDA-Q compiler boundary product

**First-class external-compiler boundary register** — not a marketing tick list.
Status enum: `supported` | `adapter` | `implementation_path` | `permanent_boundary`.

Module: `scpn_quantum_control.compiler_boundary_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `compiler_boundary_product.v2` |
| Unknown compiler | Fail closed |
| CUDA-Q full runtime invent-green | Refuse |
| QIR provider submit invent-green | Refuse |
| permanent_boundary import/export | Refused |
| QIR / Catalyst import-export | Validate-only when allowed |

## Catalogue

| compiler_id | status |
|---|---|
| `mlir_enzyme_in_tree` | adapter (ambient LLVM/JIT claim gate) |
| `catalyst_external` | implementation_path |
| `qir` | implementation_path (validate-only) |
| `cudaq` | permanent_boundary (no owner GPU programme) |
| `tensor_network_future` | permanent_boundary |

## Quick start

```python
from scpn_quantum_control.compiler_boundary_product import (
    assert_compiler_boundary_product_integrity,
    build_compiler_boundary_product_registry,
    decide_compiler_path,
    materialise_demo_compiler_boundary_probe,
)

reg = assert_compiler_boundary_product_integrity(
    build_compiler_boundary_product_registry()
)
assert reg["invent_green_runtime_policy"] is False

assert decide_compiler_path("qir", request_import_export=True).allowed is True
assert decide_compiler_path("cudaq", invent_green_full_runtime=True).allowed is False

probe = materialise_demo_compiler_boundary_probe()
assert probe.invent_green_cudaq_runtime is False
assert probe.catalyst_promotion_ready is False
```

## API reference

All public objects are exported by `scpn_quantum_control.compiler_boundary_product`.
The registry and every serialised row use JSON-ready dictionaries, lists,
strings, booleans, and integers.

### Types and constants

| API | Contract |
|---|---|
| `BoundaryStatus` | Literal status: `supported`, `adapter`, `implementation_path`, or `permanent_boundary`. |
| `SupportPosture` | Literal evidence posture: `local_research`, `live_hardware_gated`, `policy_only`, or `metadata_only`. |
| `PathDecisionOutcome` | Literal decision outcome: `allowed` or `refused`. |
| `COMPILER_BOUNDARY_PRODUCT_SCHEMA` | Stable serialisation schema identifier, currently `compiler_boundary_product.v2`. |
| `COMPILER_BOUNDARY_CLAIM_BOUNDARY` | Shared non-promotional boundary copied into rows, decisions, probes, and registries. |

### Data models

#### `CompilerBoundaryRow`

Frozen, slotted catalogue row. Construction validates all identifiers and
pointers, rejects unknown status or posture values, and forbids invent-green
runtime claims. A `permanent_boundary` row cannot allow import/export.
`to_dict()` returns the complete JSON-ready row without mutating it.

#### `PathEligibilityDecision`

Frozen, slotted result from `decide_compiler_path()`. The `allowed` boolean and
`outcome` literal must agree. Allowed decisions cannot carry blockers; refused
decisions require at least one non-blank blocker. `to_dict()` preserves those
fields for evidence bundles.

#### `MaterialisedCompilerBoundaryProbe`

Frozen, slotted snapshot of the ambient Catalyst comparison and LLVM/JIT claim
gate. Both invent-green flags are permanently false. Construction rejects
blank status, boundary, or demo labels; `to_dict()` provides the serialisable
probe.

### Catalogue access

| Function | Parameters | Returns | Failure behaviour |
|---|---|---|---|
| `list_compiler_ids()` | None | Compiler ids in canonical order. | No failure for the built-in non-empty catalogue. |
| `get_compiler_boundary(compiler_id)` | Non-blank compiler id. | Matching `CompilerBoundaryRow`. | Raises `ValueError` for blank or unknown ids. |
| `iter_compiler_boundaries(*, status=None, support_posture=None)` | Optional literal filters. | Stable tuple satisfying both filters. | Returns an empty tuple when no row matches. |

```python
from scpn_quantum_control.compiler_boundary_product import (
    get_compiler_boundary,
    iter_compiler_boundaries,
    list_compiler_ids,
)

assert list_compiler_ids()[0] == "mlir_enzyme_in_tree"
assert get_compiler_boundary("qir").import_export_allowed is True
assert all(
    row.import_export_allowed is False
    for row in iter_compiler_boundaries(status="permanent_boundary")
)
```

### Decisions and probes

`decide_compiler_path(compiler_id, *, request_import_export=False,
invent_green_full_runtime=False, invent_green_provider_submit=False)` returns a
`PathEligibilityDecision`. It refuses full-runtime invention, provider/QIR
submission invention, and import/export on rows that do not allow it. It does
not submit jobs or initialize provider SDKs.

`materialise_compiler_boundary_probe(*, catalyst_runner_status="runtime_gap")`
accepts `dependency_gap`, `runtime_gap`, `correctness_gap`, or `success`. It
composes the ambient comparison with the LLVM/JIT claim boundary and returns a
finite local probe. `materialise_demo_compiler_boundary_probe()` fixes the
status to `runtime_gap` for deterministic examples and tests.

### Registry and integrity

`map_compiler_boundary_public_surfaces()` returns deterministic module/symbol
metadata. `build_compiler_boundary_product_registry()` assembles the schema,
claim boundary, policy flags, public surfaces, and canonical rows.
`assert_compiler_boundary_product_integrity(payload=None)` validates a supplied
registry or builds the canonical one. It raises `ValueError` for stale schemas,
unexpected top-level keys, claim or policy drift, malformed or duplicate rows,
canonical-row drift, missing QIR/CUDA-Q boundaries, inconsistent counts, blanks,
or invent-green policy flags.

```python
from scpn_quantum_control.compiler_boundary_product import (
    assert_compiler_boundary_product_integrity,
)

validated = assert_compiler_boundary_product_integrity()
assert validated["compiler_count"] == len(validated["compilers"])
assert validated["blank_entry_count"] == 0
```

## Safety and side effects

- Catalogue access, decisions, probes, and registry validation are local and
  deterministic for the supplied inputs.
- No API in this module performs provider submission, credential lookup,
  hardware execution, network I/O, or filesystem writes.
- A `success` Catalyst runner status is an input observation, not release,
  hardware, provider, or scientific-promotion evidence.
- Callers must preserve governed-route and separate authorization
  gates before any external compiler or provider action.

## Residuals (honest)

- Governed-route identifiers and competitive-watch feeds still require automation
- The Rust LLVM/JIT decision must cite this register
- Full CUDA-Q runtime remains out of scope without owner GPU programme

## Related

- Ambient: `compiler.mlir_llvm_jit_claim_gate`, `benchmarks.differentiable_catalyst_comparison`

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
