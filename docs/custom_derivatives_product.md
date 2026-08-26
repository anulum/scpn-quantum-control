# Custom / registered derivatives product

Versioned **safe extension surface** for third-party and domain JVP/VJP rules:
registration contract, public register/query helpers, fail-closed duplicates,
and a documented scaled-linear example rule.

Module: `scpn_quantum_control.custom_derivatives_product`

Composes ambient `CustomDerivativeRegistry` / `CustomDerivativeRule` — does not
rewrite the full transform algebra stack.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `custom_derivatives_product.v2` |
| Default contract | `registration_contract` |
| Identity format | `namespace:name@version` |
| Default product namespace | `scpn.product.custom_derivatives` |
| Duplicate without overwrite | Fail closed |
| Blank/unknown contract or identity | Fail closed |
| Isolated registry | Product register creates isolated registry unless caller passes one |
| Transform-algebra interaction coverage | Residual boundary (not invent-green) |
| Per-rule metamorphic verification | Residual boundary (not invent-green) |

Schema v2 replaces opaque residual-policy markers with the descriptive values
`transform-algebra-interaction-coverage` and
`custom-rule-metamorphic-verification`. Consumers of v1 policy payloads must
regenerate them; no compatibility alias retains the opaque values.

Claim boundary:

> Custom derivatives product surface only; versioned registration contract and
> fail-closed register/query over ambient CustomDerivativeRegistry; does not
> invent-green complete transform-algebra and governed route-matrix CI or mass
> rule migration; transform-algebra interaction coverage and per-rule
> metamorphic verification remain open honestly

## Public API

```python
from scpn_quantum_control.custom_derivatives_product import (
    assert_custom_derivatives_product_integrity,
    build_custom_derivatives_product_registry,
    build_example_scaled_linear_rule,
    list_custom_derivative_contract_ids,
    list_product_registered_identities,
    new_product_registry,
    probe_example_rule_round_trip,
    register_product_custom_rule,
    require_product_custom_rule,
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

## Public API contracts

### Contract discovery and policy

| API | Contract |
|---|---|
| `list_custom_derivative_contract_ids()` | Return stable catalogue identifiers without importing or executing a derivative rule. |
| `get_custom_derivative_contract(contract_id)` | Resolve one immutable contract row; blank and unknown identifiers raise `ValueError`. |
| `iter_custom_derivative_contracts(kind=None)` | Return the complete catalogue or a stable kind-filtered tuple. |
| `registration_contract_policy()` | Return the versioned identity, duplicate, rule-presence, and residual-work policy as JSON-ready data. |
| `parse_product_identity(identity)` | Preserve a `PrimitiveIdentity` or parse `namespace:name@version`; blank or malformed input fails closed. |

`CustomDerivativeContractRow` validates every catalogue identifier, kind,
title, module/symbol pointer, stability class, and inventory date when it is
constructed. `RegistrationResult` represents successful registration only;
its identity and rule name must be non-empty and `registered` must be true.
Both records provide `to_dict()` for JSON-ready evidence payloads.

### Rule construction and registry lifecycle

| API | Contract |
|---|---|
| `build_example_scaled_linear_rule(scale=2.0, name="scaled_linear")` | Build the documented `y = scale * x` rule with exact JVP and VJP; reject blank names and zero or non-finite scales. |
| `new_product_registry()` | Return a fresh isolated registry; it never mutates the ambient process registry. |
| `register_product_custom_rule(identity, rule, *, overwrite=False, registry=None)` | Validate and bind a rule. Omitted `registry` creates an isolated registry; duplicates fail unless overwrite is explicit. |
| `require_product_custom_rule(identity, *, registry)` | Return the exact registered rule or raise when the identity/registry is invalid or missing. |
| `list_product_registered_identities(*, registry)` | Return canonical identity keys in sorted order. |
| `probe_example_rule_round_trip(scale=2.0, values=None, tangent=None)` | Register and execute the example through ambient `value_and_custom_jvp`; reject shape, value, or JVP disagreement. |

Callers that need persistence must create and retain a registry explicitly:

```python
registry = new_product_registry()
identity = "example.team:scaled_linear@1"
rule = build_example_scaled_linear_rule(scale=3.0, name="scale_by_three")
register_product_custom_rule(identity, rule, registry=registry)

assert list_product_registered_identities(registry=registry) == (identity,)
assert require_product_custom_rule(identity, registry=registry) is rule
```

### Registry evidence and integrity

| API | Contract |
|---|---|
| `map_custom_derivatives_public_surfaces()` | Emit deterministic module descriptors with contract identifiers, kind, stability, and claim boundary. |
| `build_custom_derivatives_product_registry()` | Build the schema-tagged contract, policy, public-surface, count, and residual-work evidence payload. |
| `assert_custom_derivatives_product_integrity(payload=None)` | Reject empty, non-mapping, blank, invalid-kind, duplicate, count-drifted, default-missing, contract-drifted, or permissive-policy state. |

These surfaces describe and validate the bounded registration product. They do
not claim full transform-algebra matrix CI, mass rule migration, provider or
hardware evidence, or complete per-rule metamorphic verification.

## Bounded product status

Shipped: versioned registration contract · public register API with fail-closed
duplicates · documented example custom rule · metamorphic-verification boundary
pointer.

Open: complete transform-algebra interaction CI across the transform algebra and
governed route matrix · complete metamorphic automation for every new rule.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
