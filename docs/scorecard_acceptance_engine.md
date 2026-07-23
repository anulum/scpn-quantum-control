# Scorecard acceptance engine (BL-56 / W1)

Fail-closed promotion surface for the **eleven differentiable baseline-scorecard
categories**. Honest ``behind_baseline`` inventory is the default until claim-ledger
and external-comparison evidence packages exist.

Module: `scpn_quantum_control.scorecard_acceptance_engine`  
Composes category ids from `differentiable_baseline_scorecard` (does not rewrite
the committed scorecard artefact).

## Rules

| Status | Meaning |
|---|---|
| `behind_baseline` | Honest inventory; requires blockers |
| `at_baseline` / `exceeds_baseline` | Only via promote with required evidence |
| `not_comparable` | Requires impossibility evidence labels |

Promotion to ready statuses requires evidence ids covering:

* `claim_ledger_promoted_row`
* `external_baseline_comparison`

and forbids unbounded promotional language (“category of its own”, “state-of-the-art”, …).

Claim boundary:

> scorecard acceptance engine only; behind_baseline is an honest inventory
> state; promote refuses invent-green at_baseline/exceeds_baseline without
> required evidence digests and language-safe claims

## Public API

```python
from scpn_quantum_control.scorecard_acceptance_engine import (
    assert_scorecard_acceptance_integrity,
    build_scorecard_acceptance_registry,
    list_scorecard_category_ids,
    promote_scorecard_category,
)

reg = assert_scorecard_acceptance_integrity(build_scorecard_acceptance_registry())
assert reg["behind_baseline_count"] == 11
assert len(list_scorecard_category_ids()) == 11

# Invent-green without evidence → refuse
d = promote_scorecard_category(
    "benchmark_promotion",
    target_status="exceeds_baseline",
)
assert d.allowed is False
assert d.missing_evidence
```

## Bounded product status

Shipped: S56.0–S56.3 catalogue + fail-closed promote + honest inventory + docs.

Open: S56.4 regenerator/CI drift · S56.5 hermetic commands · S56.6 dashboard ·
S56.7 BL-61 feeds.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
