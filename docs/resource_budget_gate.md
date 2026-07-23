# Compile & dense resource budget gate (BL-94 / W5)

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

## Bounded product status

Shipped: S94.0 budget dimensions · S94.2 exceed → typed refuse · S94.3 docs
product surface.

Open: S94.1 full enforce hooks in every compiler/linalg call site · studio
execute preflight wiring beyond this catalogue.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
