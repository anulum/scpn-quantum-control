# Studio executive + coverage frontier product (BL-62)

Versioned **executive verb catalogue** with BL-52 route pointers and a
**coverage-frontier** score (honesty × answer-rate). Never invent-green full
coverage while refuse rates are hidden.

Module: `scpn_quantum_control.studio_executive_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `studio_executive_product.v1` |
| Default verb | `differentiate` |
| Unknown verb | Fail closed |
| Unsupported route invent-green | Refused (BL-52/53) |
| Full coverage invent-green | Refused |
| Live execute without approval | Refused |

Claim boundary:

> Studio executive + coverage frontier product surface only; catalogues
> executive verbs with BL-52 route pointers and materialises honesty×answer-rate
> coverage-frontier probes; invent_green_full_coverage=false when boundary
> abstentions exist; refuses invent-green unsupported routes and hidden refuse
> rates; does not claim full BL-55 kit export or Studio UI redesign (S62.4 residual)

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

## Verbs (S62.0)

`compile` · `simulate` · `analyse` · `validate` · `benchmark` · `replay` ·
`differentiate` · `mitigate` · `execute` (gated live hardware)

## Coverage frontier (S62.3)

Demo partition: total=10, answered=3, honest abstentions=5, improvable=2 →
answer_rate=0.3, honesty_rate=0.8, frontier_score=0.24, off_frontier=True.

## Bounded product status

Shipped: S62.0 verb→route catalogue · S62.1 inventory · S62.2 fail-closed verb
paths · S62.3 coverage frontier metric + tests · S62.5 product docs.

Open residual: S62.4 evidence bundle export → BL-55 kit.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
