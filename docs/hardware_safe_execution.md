# Hardware-safe gradient execution product (BL-47 / Axis 5)

Fail-closed **no-submit default** execution policy surface for hardware-adjacent
gradient planning. Dry-run plans estimate shot budgets and cost-model status
without provider submission; enforce refuses would-submit and over-budget paths.

Module: `scpn_quantum_control.hardware_safe_execution`  
Complements `phase.hardware_gradient_policy` (preparation decisions) with a
versioned public catalogue + probe product.

## Rules

| Rule | Behaviour |
|---|---|
| Default policy | `default_no_submit` with `no_submit=True` |
| Would-submit | Always refused on this product surface |
| Dry-run | Allowed when within shot/param budgets |
| Cost model | `unavailable` / `blocked` / `rate_table` (fixture only; no invent vendor rates) |
| Ticketed prep | Requires `owner_allow_submit` + non-empty ticket; still no live submit |
| Unknown/blank policy | Fail closed |

Claim boundary:

> hardware-safe execution product only; no-submit is the default; dry-run plans
> estimate shots/cost status without provider submission; enforce refuses
> would-submit and over-budget without owner-gated allow; this surface never
> executes QPU jobs or invents hardware results

## Public API

```python
from scpn_quantum_control.hardware_safe_execution import (
    assert_hardware_safe_execution_integrity,
    build_audit_record,
    build_hardware_safe_execution_registry,
    default_execution_policy,
    dry_run_execution_plan,
    enforce_execution_request,
)

assert default_execution_policy().no_submit is True
reg = assert_hardware_safe_execution_integrity(
    build_hardware_safe_execution_registry()
)
plan = dry_run_execution_plan("default_no_submit", n_params=2, shots_per_evaluation=64)
assert plan.outcome == "allowed_plan"

submit = enforce_execution_request(
    "default_no_submit",
    mode="would_submit",
    n_params=1,
    shots_per_evaluation=64,
)
assert submit.allowed is False
audit = build_audit_record(submit)
assert audit.to_dict()["contains_secrets"] is False
```

## Bounded product status

Shipped: S47.0–S47.1 policy catalogue · S47.2 shot budgets · S47.3 cost status ·
S47.4 dry-run planner · S47.5 audit records · S47.6 enforce wrapper · docs.

Open: S47.7 BL-33/BL-09 integration · S47.8 BL-29 ticket package template ·
multi-file `execution_policy/` package split · live rate tables (owner data).

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
