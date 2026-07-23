# stable_core experiment model product (BL-81 / P1)

Versioned **public experiment model** over durable `Problem` / `Backend` /
`Experiment` / `Result` contracts: schema policy, JSON envelope round-trip,
and digest helpers. Ambient `stable_core` remains the narrow durable
SemVer-intent surface under BL-97 honesty.

Module: `scpn_quantum_control.stable_core_product`

## Rules

| Rule | Behaviour |
|---|---|
| Model schema | `stable_core.experiment_model.v1` |
| Product schema | `stable_core_product.v1` |
| Silent field drop | Refused |
| Blank/unknown schema or contract | Fail closed |
| Demo path | Classical-reference, no hardware submission |
| Stability | `stable_core` (BL-97 durable intent) |
| Substrate pointers | BL-55 hermetic · BL-56 scorecard |

Claim boundary:

> stable_core product surface only; versioned schema policy and JSON
> round-trip/digest helpers over Problem/Backend/Experiment/Result; narrow
> durable SemVer-intent surface under BL-97; substrate for BL-55 hermetic kits
> and BL-56 scorecards; does not migrate all challenge/scorecard adapters
> (S81.3 residual); does not invent-green hardware submission or claim full
> historical field compatibility matrix

## Public API

```python
from scpn_quantum_control.stable_core_product import (
    assert_stable_core_product_integrity,
    build_demo_experiment,
    build_stable_core_product_registry,
    list_stable_core_contract_ids,
    round_trip_experiment,
    schema_version_policy,
)

assert "experiment_contract" in list_stable_core_contract_ids()
reg = assert_stable_core_product_integrity(build_stable_core_product_registry())
policy = schema_version_policy()
assert policy["silent_field_drop_allowed"] is False

exp = build_demo_experiment()
rt = round_trip_experiment(exp)
assert rt.matched is True
assert rt.digest_sha256
```

## Bounded product status

Shipped: S81.0 schema version policy · S81.1 public docs/API map · S81.2 JSON
round-trip + digest helpers · partial S81.4 compatibility via fail-closed
round-trip field-loss detection · BL-97/55/56 pointers.

Open: S81.3 mass challenge/scorecard adapter migration onto stable_core types ·
full historical field compatibility matrix beyond envelope v1.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
