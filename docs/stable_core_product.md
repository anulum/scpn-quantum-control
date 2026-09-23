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
| `canonical_json_bytes(payload)` | Produce deterministic UTF-8 JSON with sorted keys and compact separators; refuse NaN and infinities. |
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

`read_experiment_with_semantics(envelope, companion)` is the experimental
stable-core consumer of this separation. It returns the native v2 `Experiment`
and a separate `SemanticBinding`; a missing, unsupported or contradictory
companion cannot change the experiment. An invalid raw v2 envelope still raises
through the original reader. The new consumer does not expand the stable-core
v2 wire schema or turn a qualified plan into a hardware observation.
An experiment companion always stays at the planning stage. Attaching a real
but unrelated HAL workload or result cannot qualify observed settings for that
plan; observed job metadata belongs to the typed result reader.
The HAL result carries an observed shot total, while the submitted workload's
requested shots are unavailable from that result alone and remain explicitly
unqualified.

`read_result_with_semantics(envelope, companion)` gives the same separation for
a native v2 `Result`. Its first qualified result class is a local,
caller-supplied synthetic stochastic derivative. The raw result names the
exact digest of the typed `StochasticGradientResult`; the companion binds its
native parameter order, trainability, gradient layout, standard error and
confidence radius to that actual object. Both uncertainty descriptions retain
their original covariance and are never summed. The dimensionless unit is an
explicit synthetic caller declaration (`origin: caller_declared_synthetic`);
native parameter units, calibration,
hardware execution and a physical-unit claim remain unavailable. A result
with missing or substituted native evidence stays readable but unqualified.

`validate_semantic_binding(companion, raw_record)` returns a
`SemanticBinding` carrying `raw_readable`, `raw_digest`, the qualified
`ScientificSemantics` or `None`, and every `SemanticRefusal` that fired. Each
refusal names its rule, its field path and the measured evidence that
contradicted the declaration. Qualification is fail-closed: any refusal
withholds qualification and forbids persisting a qualified record.
Raw readability is established by the full versioned envelope reader, including
its body and schema checks. A previously measured producer observation may be
reused only for the same raw digest, adapter and source field; a mismatch refuses
qualification even when the replacement raw record is otherwise valid.
Required semantic sections must remain explicitly present, including empty
fidelity evidence, unavailable items and a null calibration reference where
those facts are absent. Dropping a section cannot silently erase a qualification
condition.
The v1 reader also refuses unreviewed top-level claims, extra record or backend
reference fields, extra source-binding or measurement-mapping fields, extra
setting-origin fields and unreferenced source records. Referenced planner
records cannot carry unchecked source-header claims either.
Requested and effective setting sections must be mappings, and origin keys
must match their declared setting keys exactly.
The result-only fidelity-unit declaration cannot be attached to an experiment
plan as an unchecked claim.
The experiment consumer binds its backend identity, raw digest and planning
stage to the actual v2 experiment. It refuses a claimed observed modality,
calibration or nonempty fidelity component until an independent native owner can
be checked; their absence remains explicit and does not alter raw evidence.
The plan-level field eligibility mask requires exact booleans. A `false` entry
requires a separately captured native derivative request; the plan alone
cannot establish that a parameter was frozen.
The claim boundary uses accepted v1 no-execution wording, and unavailable
evidence remains explicit. Planning settings cannot be relabelled as observed;
the experiment reader refuses an observation-stage setting even if an unrelated
HAL result is attached.
The reader invokes only its admitted pure stable-core adapters. Companion
adapter text cannot direct an arbitrary Python import or execution.
For a setting sourced from the local gradient-method planner, a matching hash
of the retained plan is insufficient: the reader reruns that bounded,
non-executing planner on the retained inputs and compares the complete output.
A changed effective shot count with a recomputed self-hash is refused.
The separate HAL result reader binds its observed shot total to the actual typed
job result. That result does not retain the submitted workload's requested
shots, so this companion leaves the requested value empty and names it as
unavailable. It does not claim that a cloud provider applied every requested
device setting.

### What the reader measures rather than assumes

Producer identity, dtype and shape are not read from a table. The reader
resolves the adapter the companion declares, invokes it on the deserialised
record, and measures what it actually produced. A companion claiming `int32`
for a `float64` matrix, a `[2]` vector for a `(2, 2)` matrix, or a same-named
class from a different module is refused against that measurement. Identity is
module-qualified and is never matched by bare class name.

The Program-AD primitive contract, differentiable parameter, deterministic
and stochastic gradient results, HAL profile, workload, job handle and count
result, Studio execution plan and Phase-QNode classical Fisher result each
expose a detached `to_semantic_source()`
projection of their actual native metadata.
The parameter projection retains name and trainability without inventing a
value or unit. Gradient projections retain order, mask and separate uncertainty
arrays without inventing caller units. HAL projections keep route capabilities
as declarations, requested shots and programme digest as request facts, and job identity and
counts as result facts; they do not infer effective shots, statevector data or
hardware attestation. The Studio plan projection keeps its verb contract,
requested parameters and no-submit boundary; it is not execution evidence. A
projection alone is not a qualified cross-family companion.
Count-based modalities remain unqualified until a native producer supplies a
checkable bit/measurement mapping; a companion's mapping label or bit list
alone cannot establish measured wiring. The raw count record stays readable.

`native_semantic_binding.capture_native_source()` and
`validate_native_source_record()` provide the experimental custody bridge for
those native owner types. The Studio plan is resolved only when Studio is
installed, so core readers do not acquire an optional Studio dependency. Each
retains a source-specific version, complete detached projection, owner identity
and SHA-256; validation compares the
retained content to the actual typed object, so recomputing a hash over a
substituted copy does not establish its origin. This bridge proves source
custody only. It does not qualify missing units, count bit order, uncertainty
aggregation, provider attestation or scientific acceptance.
The public `read_result_with_semantics()` also binds local Phase-QNode Fisher
results. The typed producer retains the full reference matrix, finite-shot
estimate, standard error, confidence radius, sampling model and count record.
Observed bitstring replay qualifies only with its native versioned bit mapping
and raw counts; expected-count analysis has no observed-count mapping. Both
uncertainty components retain their finite-shot delta-method assumptions. The
unchanged v2 result carries a checked scalar trace and source digest, while the
native result owns the full evidence. Neither route is acquired QPU data or a
physical-unit claim.
For a typed `QuantumJobResult`, `read_result_with_semantics()` qualifies only
job identity, backend, completion status and observed shot-count metadata. It
keeps the HAL counts in the source record, but the HAL owner supplies no bit-wire
mapping; the companion therefore marks count semantics unavailable and must
not present a count map. A backend profile's statevector capability also does
not turn a counts-only result into amplitudes or hardware attestation.
When a companion retains one of these native records in `source_records`,
`validate_semantic_binding()` requires the actual typed object in
`native_sources` under the same reference. The experimental
`read_experiment_with_semantics()` consumer forwards that mapping. Missing
owners, changed source versions, altered contents and rehashed substitutions
refuse qualification while the raw experiment remains readable.

Physical units are the one thing no existing contract records, so they come
from `DECLARED_FIELD_UNITS`, where every entry carries the in-repo reference
that declares it. A field whose unit is not declared anywhere is refused rather
than accepted on the companion's own label.

`DECLARED_PARAMETER_ORDER` is marked `basis="contract_choice"` because no owner
in this repository declares a canonical parameter ordering. It is a fixed
contract, not a measurement, and it says so; changing it is a contract change.

### Operations that refuse by default

Snapshot capture, transform refusal, component aggregation decisions and native
modality checks live in `semantic_operations`; the experimental
`semantic_record` API continues to re-export them for existing callers.

| Entry point | Refuses when |
|---|---|
| `validate_semantic_binding()` | any declared field, identity, convention or setting contradicts measured or declared evidence |
| `read_result_with_semantics()` | the result's digest, native owner, objective, units or uncertainty components cannot be bound |
| `apply_semantic_transform()` | no independently verified converter owns the requested transform, even if the companion lists it |
| `aggregate_fidelity_components()` | an aggregation is requested without a recorded justification |
| `qualify_native_modality()` | the native result does not carry the requested quantity |

An empty `supported_transform_composition` means no transform support exists.
A reference listed only by the companion is not independent transform authority;
this reader refuses both until a verified converter owner exists. A standard
error and a confidence radius
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
