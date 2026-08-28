# Compile & dense resource budget gate

Fail-closed **resource budgets** for sparse Pauli/compile construction and dense
Hilbert-space allocations. Product catalogue + probe over the low-level guards
in `compile_budget` and `dense_budget` (same estimate formulas; no silent
diverging math).

Module: `scpn_quantum_control.resource_budget_gate`

## Rules

| Rule | Behaviour |
|---|---|
| Catalogue | `compile_pauli_*` and `dense_hilbert_*` dimensions with explicit GiB caps |
| Estimate | Composes `estimate_pauli_operator` / `estimate_dense_allocation` |
| Within budget | Structured `allowed` decision with GiB fields |
| Exceed budget | Structured `refused` decision; `enforce_resource_budget` raises `ResourceBudgetExceededError` |
| Unknown/blank id | Fail closed |

Claim boundary:

> resource budget gate only; estimates compose compile_budget/dense_budget
> formulas with explicit GiB caps; exceed-budget is refused fail-closed; does not
> claim production OOM immunity or invent host-RAM green capacity

## Public API

```python
from scpn_quantum_control.resource_budget_gate import (
    assert_resource_budget_integrity,
    build_resource_budget_registry,
    check_resource_budget,
    enforce_resource_budget,
    estimate_resource_budget,
    list_budget_dimension_ids,
)

assert "compile_pauli_default" in list_budget_dimension_ids()
reg = assert_resource_budget_integrity(build_resource_budget_registry())
est = estimate_resource_budget("dense_hilbert_default", n_qubits=2)
assert est.within_budget is True

ok = check_resource_budget("compile_pauli_default", n_qubits=2)
assert ok.allowed is True

refused = check_resource_budget(
    "compile_pauli_tight", n_qubits=8, max_gib=1e-9
)
assert refused.allowed is False
```

## API reference

All public objects are exported by `scpn_quantum_control.resource_budget_gate`.
Every `to_dict()` result and registry payload contains JSON-ready primitives.

### Types and constants

| API | Contract |
|---|---|
| `BudgetFamily` | Literal family: `compile_pauli` or `dense_hilbert`. |
| `CheckOutcome` | Literal decision outcome: `allowed` or `refused`. |
| `RESOURCE_BUDGET_GATE_SCHEMA` | Stable registry schema identifier, currently `resource_budget_gate.v1`. |
| `RESOURCE_BUDGET_GATE_CLAIM_BOUNDARY` | Shared non-promotional boundary copied into dimensions, estimates, decisions, errors, and registries. |

### Data models and error

#### `BudgetDimension`

Frozen, slotted catalogue row containing `budget_id`, family, summary,
positive default GiB cap, low-level estimator label, inventory date, and claim
boundary. Construction rejects blanks, unknown families, and non-positive
caps. `to_dict()` returns the complete dimension.

#### `ResourceBudgetEstimate`

Frozen, slotted estimate with byte/GiB requirements, the active cap,
`within_budget`, and family-specific `detail`. Construction validates the
family, positive qubit count, byte fields, and exact consistency of
`within_budget == (bytes_required <= budget_bytes)`. `to_dict()` copies the
detail mapping into a JSON-ready dictionary.

#### `ResourceBudgetDecision`

Frozen, slotted outcome from `check_resource_budget()`. The `allowed` boolean
and `outcome` literal must agree. Allowed decisions have no blockers; refused
decisions require non-blank blockers. `to_dict()` serialises the decision.

#### `ResourceBudgetExceededError`

`MemoryError` subclass raised only by `enforce_resource_budget()` for an
over-budget request. It exposes `budget_id`, `n_qubits`, `bytes_required`, and
`budget_bytes`; `to_dict()` adds the stable error name and claim boundary.

### Catalogue access

| Function | Parameters | Returns | Failure behavior |
|---|---|---|---|
| `list_budget_dimension_ids()` | None | Dimension ids in canonical order. | No failure for the built-in non-empty catalogue. |
| `get_budget_dimension(budget_id)` | Non-blank dimension id. | Matching `BudgetDimension`. | Raises `ValueError` for blank or unknown ids. |
| `iter_budget_dimensions(*, family=None)` | Optional `BudgetFamily` filter. | Stable tuple of matching rows. | Returns an empty tuple when no row matches. |

The catalogue currently contains default and tight variants for both families.
Default caps come from `compile_budget.DEFAULT_PAULI_BUDGET_CAP_GIB` and
`dense_budget.DEFAULT_DENSE_BUDGET_CAP_GIB`; tight rows use `0.001` GiB for
deterministic refusal testing.

### Estimation

`estimate_resource_budget(budget_id, *, n_qubits, max_gib=None,
include_zz=False, dense_rank=2, dense_object_count=1)` dispatches by family:

- `compile_pauli` composes `estimate_pauli_operator`; `detail` includes
  `term_count`, `label_chars`, `include_zz`, and the low-level function name.
- `dense_hilbert` composes `estimate_dense_allocation`; `detail` includes
  dimension, shape, dtype, object count, rank, and the low-level function name.

The explicit `max_gib` override replaces the catalogue cap for that call. The
function raises `TypeError` when `n_qubits` is not an integer and `ValueError`
for unknown dimensions, `n_qubits < 1`, or a non-positive cap.

```python
from scpn_quantum_control.resource_budget_gate import estimate_resource_budget

estimate = estimate_resource_budget(
    "dense_hilbert_default",
    n_qubits=4,
    dense_rank=2,
    dense_object_count=3,
)
assert estimate.detail["object_count"] == 3
assert estimate.bytes_required <= estimate.budget_bytes
```

### Check and enforce

`check_resource_budget()` accepts the same estimator parameters and always
returns a structured decision for a valid request. Within-budget requests are
`allowed`; over-budget requests are `refused` with a non-empty blocker.

`enforce_resource_budget()` also accepts the same parameters. It returns the
estimate when allowed and raises `ResourceBudgetExceededError` when refused.
Use `check_resource_budget()` when callers need a non-throwing policy result;
use `enforce_resource_budget()` immediately before a guarded allocation.

```python
from scpn_quantum_control.resource_budget_gate import (
    ResourceBudgetExceededError,
    enforce_resource_budget,
)

try:
    enforce_resource_budget(
        "dense_hilbert_tight",
        n_qubits=8,
        max_gib=1e-9,
    )
except ResourceBudgetExceededError as error:
    assert error.bytes_required > error.budget_bytes
```

### Registry and integrity

`build_resource_budget_registry()` assembles the schema, GiB constant, claim
boundary, family counts, canonical dimensions, and residual policy note.
`assert_resource_budget_integrity(payload=None)` validates a supplied registry
or builds the canonical one. It raises `ValueError` for missing or malformed
rows, blanks, invalid caps, duplicates, catalogue drift, a missing family, or
inconsistent counts.

## Safety and side effects

- The APIs estimate memory; they do not reserve, allocate, or free it.
- No API discovers host RAM, claims OOM immunity, mutates global caps, performs
  network I/O, accesses credentials, or runs a compiler/provider/hardware job.
- An allowed decision means only that the composed estimate fits the explicit
  cap. It is not runtime, performance, production-capacity, or release proof.
- Callers remain responsible for the open enforcement hooks at every
  compiler and linear-algebra allocation boundary.

## Bounded product status

Shipped: budget dimensions · exceed → typed refuse · documentation product
surface.

Open: full enforce hooks in every compiler/linalg call site · studio
execute preflight wiring beyond this catalogue.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
