# Studio executive + coverage frontier product

Versioned **executive verb catalogue** with governed-route pointers and a
**coverage-frontier** score (honesty × answer-rate). Never invent-green full
coverage while refuse rates are hidden.

Module: `scpn_quantum_control.studio_executive_product`

This page documents a bounded executive catalogue and an honesty-aware
coverage metric. It does not dispatch a verb, grant approval, redesign Studio,
or convert a coverage score into evidence of route completeness.

## Contract discovery

| Function | Contract |
|---|---|
| `list_executive_verb_ids()` | Returns all nine stable verbs in catalogue order. |
| `get_executive_verb(verb_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_executive_verbs(...)` | Returns all rows or filters by support posture. |
| `map_studio_executive_public_surfaces()` | Identifies the product, ambient verb spine, and coverage-frontier owner. |

When the optional Studio platform is importable, the catalogue reads ambient
verb contracts. Otherwise it uses the explicit product-local fallback with the
same nine bounded verbs. Discovery itself executes no verb or hardware route.

## Public value objects

- `ExecutiveVerbRow` records governed-route and unsuitable-scenario pointers,
  support posture,
  approval/live-hardware flags, backends, and the shared claim boundary.
- `PathEligibilityDecision` records an allowed/refused outcome, reason, ordered
  blockers, and non-promotional boundary.
- `MaterialisedCoverageFrontierProbe` records ledger counts, answer/honesty
  rates, frontier score, off-frontier state, and the mandatory false
  invent-green flag.

All records are immutable slot-backed dataclasses with validated construction
and JSON-ready `to_dict()` mappings.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `studio_executive_product.v2` |
| Default verb | `differentiate` |
| Unknown verb | Fail closed |
| Unsupported route invent-green | Refused by governed-route and unsuitable-scenario policy |
| Full coverage invent-green | Refused |
| Live execute without approval | Refused |

## Executive path decisions

`decide_executive_path()` validates the exact verb, then refuses unsupported
route claims, invented full coverage, and approval-gated verbs without explicit
approval. Multiple blockers are de-duplicated in first-seen order.

An allowed decision says only that the product policy admits that route. It
does not execute the route, authorize a provider, spend shots, or certify an
output.

Claim boundary:

> Studio executive + coverage frontier product surface only; catalogues
> executive verbs with governed-route pointers and materialises honesty×answer-rate
> coverage-frontier probes; invent_green_full_coverage=false when boundary
> abstentions exist; refuses invent-green unsupported routes and hidden refuse
> rates; does not claim full reproduction-kit export or Studio UI redesign

## Public API

```python
from scpn_quantum_control.studio_executive_product import (
    assert_studio_executive_product_integrity,
    build_studio_executive_product_registry,
    decide_executive_path,
    list_executive_verb_ids,
    materialise_demo_coverage_frontier_probe,
)

assert "differentiate" in list_executive_verb_ids()
reg = assert_studio_executive_product_integrity(
    build_studio_executive_product_registry()
)
probe = materialise_demo_coverage_frontier_probe()
assert probe.invent_green_full_coverage is False
assert abs(probe.frontier_score - 0.24) < 1e-9  # 0.8 * 0.3

assert decide_executive_path("execute").allowed is False  # needs approval
assert decide_executive_path("execute", approval_present=True).allowed is True
```

## Executive verbs

`compile` · `simulate` · `analyse` · `validate` · `benchmark` · `replay` ·
`differentiate` · `mitigate` · `execute` (gated live hardware)

Every verb carries governed-route and unsuitable-scenario pointers. Only
`execute` may advertise live hardware, and it must require
approval. The remaining verbs are local research routes in this product.

## Coverage frontier

Demo partition: total=10, answered=3, honest abstentions=5, improvable=2 →
answer_rate=0.3, honesty_rate=0.8, frontier_score=0.24, off_frontier=True.

`compute_coverage_frontier_score()` requires a positive total, non-negative
improvable count, and a partition that does not exceed the total. It computes:

- `answer_rate = answered_confident / total_claims`;
- `honesty_rate = (answered_confident + honest_abstentions) / total_claims`;
- `frontier_score = honesty_rate * answer_rate`.

The result always sets `invent_green_full_coverage=False`. Honest abstention is
bookkeeping, not an answered claim, and therefore cannot inflate answer rate.

## Registry integrity

`build_studio_executive_product_registry()` emits schema
`studio_executive_product.v2`, the full verb catalogue, public surface map,
default id, counts, policy note, and shared claim boundary.

Always validate transported or stored payloads through
`assert_studio_executive_product_integrity()`. It rejects missing, empty,
non-list, non-mapping, blank, duplicate, missing, or extra rows; missing route
pointers or backend lists; live hardware on any verb except `execute`; loss of
the `differentiate` or `execute` sentinels; count drift; and any relaxed
invent-green policy.

## Failure handling and operational non-effects

Treat `ValueError` as a caller-contract, route-policy, score-partition, or
transported registry failure. Treat `RuntimeError` from catalogue construction
as repository or optional-platform contract corruption.

This product performs no network access, credential lookup, provider or QPU
discovery, verb execution, approval mutation, hardware submission, spend,
benchmark, evidence export, reproduction-kit generation, Studio UI mutation, result
promotion, or publication.

## Bounded product status

Shipped: verb-to-route catalogue and inventory, fail-closed verb paths,
coverage-frontier metric and tests, and product documentation.

Open: evidence-bundle export through the reproduction kit.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
