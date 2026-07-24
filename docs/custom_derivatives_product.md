# Custom / registered derivatives product (BL-92 / P1)

Versioned **safe extension surface** for third-party and domain JVP/VJP rules:
registration contract, public register/query helpers, fail-closed duplicates,
and a documented scaled-linear example rule.

Module: `scpn_quantum_control.custom_derivatives_product`

Composes ambient `CustomDerivativeRegistry` / `CustomDerivativeRule` — does not
rewrite the full transform algebra stack.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `custom_derivatives_product.v1` |
| Default contract | `registration_contract` |
| Identity format | `namespace:name@version` |
| Default product namespace | `scpn.product.custom_derivatives` |
| Duplicate without overwrite | Fail closed |
| Blank/unknown contract or identity | Fail closed |
| Isolated registry | Product register creates isolated registry unless caller passes one |
| Transform-algebra CI | Residual S92.2 (not invent-green) |
| BL-46 metamorphic automation | Residual S92.4 (not invent-green) |

Claim boundary:

> Custom derivatives product surface only; versioned registration contract and
> fail-closed register/query over ambient CustomDerivativeRegistry; does not
> invent-green full transform-algebra CI (BL-03/52) or mass rule migration;
> residual S92.2 transform-algebra interaction tests and S92.4 full BL-46
> metamorphic automation open honestly

## Public API

```python
from scpn_quantum_control.custom_derivatives_product import (
    assert_custom_derivatives_product_integrity,
    build_custom_derivatives_product_registry,
    build_example_scaled_linear_rule,
    list_custom_derivative_contract_ids,
    new_product_registry,
    probe_example_rule_round_trip,
    register_product_custom_rule,
)

assert "registration_contract" in list_custom_derivative_contract_ids()
reg = assert_custom_derivatives_product_integrity(
    build_custom_derivatives_product_registry()
)

registry = new_product_registry()
rule = build_example_scaled_linear_rule(scale=2.0)
result = register_product_custom_rule(
    "scpn.product.custom_derivatives:demo@1",
    rule,
    registry=registry,
)
assert result.registered is True

probe = probe_example_rule_round_trip(scale=2.0)
assert probe["value"] == [2.0, 4.0]
assert probe["jvp"] == [2.0, 2.0]
```

## Bounded product status

Shipped: S92.0 registration contract · S92.1 public register API + fail-closed
duplicates · S92.3 example custom rule + docs · BL-46 residual pointer.

Open: S92.2 full transform-algebra interaction CI (BL-03/52) · S92.4 full
BL-46 metamorphic automation for every new rule.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
