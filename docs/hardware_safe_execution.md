# Hardware-safe gradient execution product

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

The registry schema is `hardware_safe_execution.v1`. Catalogue order and policy
identifiers are stable. Every policy defines positive per-evaluation, total,
parameter, and shift-term ceilings. A policy cannot set `no_submit=False`
without explicit `owner_allow_submit=True`, and the product still never submits
a job.

Claim boundary:

> hardware-safe execution product only; no-submit is the default; dry-run plans
> estimate shots/cost status without provider submission; enforce refuses
> would-submit and over-budget without owner-gated allow; this surface never
> executes QPU jobs or invents hardware results

## Policy catalogue

`list_execution_policy_ids()` returns canonical identifiers in stable order.
`get_execution_policy(policy_id)` trims outer whitespace and raises
`ValueError` for blank or unknown values. `iter_execution_policies(no_submit)`
optionally filters the immutable catalogue; omitting the filter returns all
policies. `default_execution_policy()` resolves `default_no_submit`.

`ExecutionPolicy` validates its identifier, summary, positive budgets, total
versus per-evaluation shots, cost status, rate, owner gate, and inventory date.
`cost_model_status` has three honest states:

| Status | Meaning |
|---|---|
| `unavailable` | No rate table is present; estimated cost is `None` |
| `blocked` | Cost evaluation is deliberately unavailable for this policy |
| `rate_table` | The policy carries an explicit non-negative fixture rate |

A non-rate-table policy must carry a zero rate. The built-in ticketed-prep row
uses a zero-cost fixture only; it is not a vendor price claim.

## Dry-run planning

`dry_run_execution_plan(...)` estimates two-sided parameter-shift work as
`2 * n_params * shift_terms` evaluations and multiplies by shots per
evaluation. When shots are omitted, the policy maximum is used. Non-positive
parameters, shifts, or shots raise `ValueError`.

The returned immutable `DryRunPlan` records dimensions, evaluations, total
shots, optional cost, submit intent, outcome, reason, and blockers. It refuses
requests that exceed any policy ceiling. Submit intent also adds no-submit and
owner-gate blockers. An allowed plan has no blockers; a refused or blocked plan
must explain at least one blocker.

## Enforcement modes

`enforce_execution_request(...)` accepts exactly three modes:

| Mode | Decision |
|---|---|
| `dry_run` | Allowed only when the dry-run plan is within every budget |
| `would_submit` | Always refused because this surface never submits |
| `ticketed_prep` | Allowed for plan preparation only when the policy owner gate, non-empty ticket, and every budget pass |

An allowed ticketed-prep decision is not submission authority. Ticket labels
are identifiers, not credentials, and the function does not use them to contact
a provider. Duplicate blocker text is removed while preserving order.

`EnforceDecision` includes the policy, mode, allowed/outcome state, total shots,
reason, blockers, and a deterministic audit identifier. Value-object validation
rejects inconsistent allowed/outcome/blocker combinations and negative totals.

## Audit records and registry integrity

`build_audit_record(decision)` copies a decision into an immutable
`AuditRecord`. Its JSON-ready mapping always reports `contains_secrets=False`;
no ticket value is copied into the record.

`build_hardware_safe_execution_registry()` serialises the schema, claim
boundary, counts, default identifier, note, and every policy.
`assert_hardware_safe_execution_integrity(payload=None)` rejects:

- an absent, empty, or non-list policy collection;
- non-mapping, blank, duplicate, or unknown policy rows;
- a missing or submit-enabled `default_no_submit` row;
- non-boolean no-submit values or non-positive budgets;
- canonical policy-set drift; and
- inconsistent blank or policy counts.

The validator returns a shallow dictionary copy. It proves registry structure
and safety posture, not provider availability or successful execution.

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
assert plan.estimated_total_shots == 256

submit = enforce_execution_request(
    "default_no_submit",
    mode="would_submit",
    n_params=1,
    shots_per_evaluation=64,
)
assert submit.allowed is False
audit = build_audit_record(submit)
assert audit.to_dict()["contains_secrets"] is False

prep = enforce_execution_request(
    "owner_ticketed_prep",
    mode="ticketed_prep",
    n_params=1,
    shots_per_evaluation=64,
    live_execution_ticket="owner-ticket-reference",
)
assert prep.allowed is True
```

Importing the module, listing policies, planning, enforcing, building an audit
record, or validating the registry does not access a network, inspect provider
credentials, submit a circuit, spend funds, execute feedback control, mutate a
rate table, or promote hardware readiness.

## Bounded product status

Shipped: policy catalogue · shot budgets · cost status · dry-run planner ·
secret-free audit records · enforcement wrapper · public docs.

Open: co-design/planner integration · provider ticket-package template ·
multi-file `execution_policy/` package split · live rate tables (owner data).

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
