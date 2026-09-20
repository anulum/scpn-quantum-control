# Stable-core experiment model

Versioned **public experiment model** over durable `Problem` / `Backend` /
`Experiment` / `Result` contracts: schema policy, JSON envelope round-trip,
and digest helpers. Ambient `stable_core` remains the narrow durable
SemVer-intent surface in the public API stability programme.

Module: `scpn_quantum_control.stable_core_product`

## Rules

| Rule | Behaviour |
|---|---|
| Model schema | `stable_core.experiment_model.v2` |
| Product schema | `stable_core_product.v2` |
| Silent field drop | Refused |
| Blank/unknown schema or contract | Fail closed |
| Demo path | Classical-reference, no hardware submission |
| Stability | `stable_core` (narrow durable SemVer intent) |
| Substrate pointers | Hermetic reproduction kits · scorecard acceptance |

Claim boundary:

> stable_core product surface only; versioned schema policy and JSON
> round-trip/digest helpers over Problem/Backend/Experiment/Result; narrow
> durable SemVer-intent surface in the public API stability programme;
> substrate for hermetic reproduction kits and scorecard acceptance; challenge
> and scorecard adapter migration is incomplete; does not invent-green
> hardware submission or claim full historical field compatibility

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

Versioned wrapping and unwrapping require the complete `to_dict()` field set
for each model, including nested experiment problems and backends. Missing or
unexpected fields are rejected rather than defaulted or discarded. A problem's
`n_qubits` must be a positive integer (not a boolean) matching its matrix and
frequency dimensions. This enforces the existing v2 layout without adding fields
or changing the schema version.

Use `metadata` for application-specific extensions such as units and requested
versus effective settings. The direct `*_from_dict()` convenience constructors
retain their existing defaults; they are not versioned envelope validators.
These field checks do not establish physical unit correctness or provider fidelity.

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

## Scientific-semantics companion

`stable_core.experiment_model.v2` is unchanged by this companion. A record
keeps its exact bytes, its digest and its existing readers; the companion is a
separate `scientific_semantics.v1` document that references the raw record by
digest and adds the metadata the raw envelope never carried — units, dtypes and
shapes per field, parameter order and tangent convention, requested against
effective settings with their provenance, fidelity components, and an explicit
claim boundary.

A record without a companion stays fully readable. What a missing or refused
companion withholds is *qualification*, never raw custody, and a refusal never
rewrites, coerces or substitutes a value.

`validate_semantic_binding(companion, raw_record)` returns a
`SemanticBinding` carrying `raw_readable`, `raw_digest`, the qualified
`ScientificSemantics` or `None`, and every `SemanticRefusal` that fired. Each
refusal names its rule, its field path and the measured evidence that
contradicted the declaration. Qualification is fail-closed: any refusal
withholds qualification and forbids persisting a qualified record.

### What the reader measures rather than assumes

Producer identity, dtype and shape are not read from a table. The reader
resolves the adapter the companion declares, invokes it on the deserialised
record, and measures what it actually produced. A companion claiming `int32`
for a `float64` matrix, a `[2]` vector for a `(2, 2)` matrix, or a same-named
class from a different module is refused against that measurement. Identity is
module-qualified and is never matched by bare class name.

Physical units are the one thing no existing contract records, so they come
from `DECLARED_FIELD_UNITS`, where every entry carries the in-repo reference
that declares it. A field whose unit is not declared anywhere is refused rather
than accepted on the companion's own label.

`DECLARED_PARAMETER_ORDER` is marked `basis="contract_choice"` because no owner
in this repository declares a canonical parameter ordering. It is a fixed
contract, not a measurement, and it says so; changing it is a contract change.

### Operations that refuse by default

| Entry point | Refuses when |
|---|---|
| `validate_semantic_binding()` | any declared field, identity, convention or setting contradicts measured or declared evidence |
| `apply_semantic_transform()` | the request names no accepted transform in `supported_transform_composition` |
| `aggregate_fidelity_components()` | an aggregation is requested without a recorded justification |
| `qualify_native_modality()` | the native result does not carry the requested quantity |

An empty `supported_transform_composition` means no transform support exists,
not that every transform is free. A standard error and a confidence radius
derived from the same covariance are two descriptions of one uncertainty, so
summing them is refused and the components are preserved separately. A backend
profile advertising `supports_statevector` is a declaration about the backend,
not evidence about a result that came back with counts only; amplitudes are
never inferred or padded.

`capture_semantic_record()` deep-copies both documents and fixes their digests
at capture time, so later mutation of either source cannot reach the snapshot.

## Bounded product status

Shipped: model and product schema version policy · public documentation and API
map · JSON round-trip and digest helpers · fail-closed contract, envelope, and
registry drift detection · public stability, hermetic reproduction, and
scorecard acceptance pointers.

Open: broad challenge and scorecard adapter migration onto stable-core types ·
full historical field compatibility matrix beyond envelope v2 · companion
binding across the remaining producer families beyond the frozen custody
corpus · a declared canonical parameter order owned by each producer rather
than fixed as a contract choice in the semantics owner.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
