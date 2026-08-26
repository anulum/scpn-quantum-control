# stable_core experiment model product

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

### Contract discovery

| API | Contract |
|---|---|
| `list_stable_core_contract_ids()` | Return contract identifiers in stable catalogue order. |
| `get_stable_core_contract(contract_id)` | Resolve one identifier and reject blank or unknown values. |
| `iter_stable_core_contracts(kind=...)` | Return the complete catalogue or an immutable kind-filtered view. |
| `map_stable_core_public_surfaces()` | Emit deterministic rows linking each contract to its ambient public symbol. |

### Schema and envelope operations

| API | Contract |
|---|---|
| `schema_version_policy()` | Declare the one supported model schema and the no-silent-drop policy. |
| `validate_model_schema_version(version)` | Return a supported normalised version; reject blank or unknown versions. |
| `wrap_model_envelope(kind, body, schema_version=...)` | Bind a non-empty contract body to its kind, version, and claim boundary. |
| `unwrap_model_envelope(envelope)` | Validate the version, kind, and body before returning them. |
| `canonical_json_bytes(payload)` | Produce deterministic UTF-8 JSON with sorted keys and compact separators. |
| `digest_stable_core_payload(payload)` | Produce the lowercase SHA-256 digest of the canonical JSON bytes. |

### Model conversion and round-trip proof

| Model | From mapping | To envelope | From envelope |
|---|---|---|---|
| `Problem` | `problem_from_dict()` | `serialise_problem()` | `deserialise_problem()` |
| `Backend` | `backend_from_dict()` | `serialise_backend()` | `deserialise_backend()` |
| `Experiment` | `experiment_from_dict()` | `serialise_experiment()` | `deserialise_experiment()` |
| `Result` | `result_from_dict()` | `serialise_result()` | `deserialise_result()` |

`round_trip_problem()` and `round_trip_experiment()` compare canonical payloads,
raise if any field changes or disappears, and return a
`StableCoreRoundTripResult` containing the verified payload and digest.
`build_demo_experiment()` is deterministic and uses the classical-reference
backend; it never submits hardware work.

### Registry integrity

`build_stable_core_product_registry()` emits the schema-tagged catalogue,
policy, public-surface map, and bounded claim text.
`assert_stable_core_product_integrity()` rejects empty catalogues, blank or
duplicate identifiers, invalid kinds, missing symbols, count drift, a missing
default experiment contract, and any policy that permits silent field drops.

## Bounded product status

Shipped: S81.0 schema version policy · S81.1 public docs/API map · S81.2 JSON
round-trip + digest helpers · partial S81.4 compatibility via fail-closed
round-trip field-loss detection · BL-97/55/56 pointers.

Open: S81.3 mass challenge/scorecard adapter migration onto stable_core types ·
full historical field compatibility matrix beyond envelope v1.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
