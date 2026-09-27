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

## Independent local conformance

`IndependentConformanceProtocol` in
`scpn_quantum_control.metamorphic_ad_verification` declares a scalar estimand,
domain, oracle class and source, exact SHA-256 identities of the oracle, input,
product source and dataset, comparator version, runtime, and a numerical budget
*before* accepting a comparison. Positive finite integer budgets normalize to
float; booleans and malformed budgets refuse before identity calculation.
`evaluate_independent_scalar_conformance`
compares both the actual primal and gradient to separately supplied oracle
values. `require_current_conformance` refuses a failed result or one whose
protocol identity changed; it also rechecks both residuals against the
predeclared budget rather than trusting a stored pass flag. An installed
optional comparator or a registered
metamorphic law is not itself executed numerical evidence.
The scalar evaluator accepts analytic or numerical oracle classes only;
metamorphic relations, empirical sampling and formal proof claims require
their separate evaluators and cannot become green through scalar agreement.

The focused local example runs the public central finite-difference owner on
`x³` at `x=2`, then compares its result with the analytic values `8` and `12`
derived in a separate test owner. A changed source, oracle, dataset, comparator
version, or numerical budget invalidates that result. The protocol digest records
provenance; it does not prove that a caller's claimed oracle is independent.
The same focused test owner also compares actual public Kuramoto trajectories
with two-oscillator analytic solutions, a delayed zero-force trajectory and a
fixed-noise step with direct arithmetic, a Program AD inverse-matrix gradient
with a hand-derived diagonal formula, and a non-diagonal near-singular 2×2
solve gradient checked against an independent adjugate identity. Forward and
reverse gradient agreement permits only float64-scale roundoff; a materially
changed gradient still fails. Each result qualifies only its
named input and budget. These checks do not qualify external comparator builds,
physical hardware, broad algorithm families or performance.

A read-only two-oscillator example sends its rational one-step objective through
Studio `differentiate`, computes a constrained proposal outside Studio, and
evaluates that proposal with the public Kuramoto flow under a held-out frequency
shift. The Studio reproduction script replays the sealed differentiation result;
the proposal and held-out evaluation are not a unified Studio action or replay
contract. Changing the sealed plan parameters invalidates its record digest.

The optional Quimb MPS owner uses spin‑½ operators, so its same-input XY
trajectory is not a Pauli-normalised comparator; its model and initial state
must be matched before a cross-method parity claim.

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

## Declared execution-memory admission

The resource-budget facade exposes `ExecutionBuffer`, `ExecutionMemoryPlan`,
`MemoryCapacity`, `check_execution_memory` and `require_execution_memory`.
A plan sums named forward, intermediate, adjoint and dense-output buffers using
fixed-width dtype sizes, then multiplies by declared concurrency. Impossible
native sizes refuse before exponential dimensions or allocations are created.
The existing catalogue estimator retains its wire format and now checks dense
addressability before building the estimated shape.

`check_execution_memory` projects a plan against an explicitly supplied capacity
observation. `require_execution_memory` reads the actual host and visible cgroup
headroom; a requested cap cannot override their ceiling. The default memory
fraction remains 30%, calculated with integer bytes. Unknown host availability
refuses. A device request requires the device-owning caller's observed available
bytes; this API does not discover or validate a GPU device.

```python
from scpn_quantum_control.resource_budget_gate import (
    ExecutionBuffer,
    ExecutionMemoryPlan,
    require_execution_memory,
)

plan = ExecutionMemoryPlan((
    ExecutionBuffer.hilbert("state", "forward", 4),
    ExecutionBuffer.hilbert("temporary", "intermediate", 4, count=2),
    ExecutionBuffer("tape", "adjoint", (8, 16), "float64"),
    ExecutionBuffer.hilbert("output", "dense_output", 4, rank=2),
), concurrency=2)
decision = require_execution_memory(plan, max_gib=0.01)
```

`bridge.knm_hamiltonian.knm_to_dense_matrix` checks its requested output,
matrix intermediates and input-conversion buffers before materialisation.
Input finite-validation masks are constructed only inside the admitted owned
scope, after its lifecycle checkpoint. Their sequential boolean storage fits
within the declared float64 conversion workspace. Capacity refusal, an expired
deadline or an already cancelled request therefore precedes these array masks.
It accepts explicit `backend="python"` or `backend="rust"`; the latter refuses
missing native support and unsupported anisotropy. The default `auto` selection
retains the optional native route and Python fallback when native support is
absent. Malformed native output is refused, rather than replaced with Python.

This is snapshot admission for declared buffers. It does not reserve memory,
account for undeclared third-party workspaces, enforce all forward/AD paths,
or establish concurrent worker isolation or production OOM immunity. Numerical
and hardware claim classes are unchanged.

`QuantumKuramotoSolver.run` now applies its existing
`max_statevector_gib` option to declared forward state, live state/measurement
intermediates, output arrays and the Python time list. An impossible step count
refuses before history construction. Accepted trajectories retain their previous
numerical fields and report `memory_declared_bytes` and `memory_budget_bytes` in
metadata.

`whole_program_value_and_grad`, `whole_program_grad`, `program_adjoint_grad` and
`program_adjoint_value_and_grad` accept `max_execution_gib`. Initial tangent-basis
storage is admitted before its allocation. Each retained numeric tangent is
checked against that context's admitted snapshot before copying into the IR
tape. Retained trace-record and alias storage is also declared before graph
mutation. Arbitrary user allocations, frontend/serialization and additional
primitive workspaces, allocator overhead and cross-worker reservations remain
outside the fully qualified scope. The derivative convention and existing result wire
are unchanged.

## Owned local memory reservations

Charge changes reread host availability and each owner's controller snapshot.
A successful resize updates its observable decision and keeps the tighter prior
cap; refusal preserves the previous decision and charge. Handoff refreshes both
owners before transferring their live declaration. Closing the transferred source
does not release the destination's charge. Caller-supplied device capacity remains
a snapshot; these scopes do not reserve OS pages or coordinate other processes.


`reserve_execution_memory` wraps real live-capacity admission in a
process/thread-owned scope. A locked ledger compares the sum of active declared
charges with their tightest active cap. `resize` admits a complete replacement
plan atomically; refusal preserves the previous charge. Exiting the scope
releases the charge on success or an escaping exception, including interruption.

Dense Hamiltonian export, forward trajectories and whole-program AD now use
this scope. AD numeric tape growth updates its charge before copying a retained
tangent. These APIs also accept `deadline_monotonic` and a caller-owned
`threading.Event` as `cancelled`; the public adjoint wrappers forward them.
Checkpoints refuse elapsed deadlines, observed cancellation, closed scopes
and access from another process or thread.

Whole-program AD creates its initial owned reservation before source/bytecode
frontend inspection. An inadmissible initial tape, observed cancellation or an
already elapsed deadline refuses before that inspection can populate the source
cache. Lifecycle is rechecked after frontend compilation and the subsequent
source read. Frontend AST, source-cache, bytecode and compiler-container allocations
are not yet fully included in the declared memory plan. Inspection is cooperative: these checks
do not interrupt an inspection already in progress.

Reservations coordinate cooperating calls within one process. They do not
reserve OS pages, retain charges for returned arrays after the operation exits,
or coordinate independent processes. Checkpoints do not forcibly terminate
a blocked native call or an arbitrary Python loop without checkpoints. The
current browser binding is synchronous; this API does not claim browser-worker
disposal or install a new worker transport.

The gradient-only AD facade forwards the same memory cap, deadline and cancellation
policy. `program_adjoint_gradient` admits retained and copied numeric gradients in
a new owned scope and checks cancellation/deadline after copying. The objective
adjoint facades forward the policy through this final copy as well. This does not
yet account for all retained trace metadata or executable adjoint replay workspaces.

`program_adjoint_replay_gradient` accepts the same cap, deadline and cancellation
keywords. Replay reserves declared cotangent, gradient and Python container
workspaces before materialising them, checks lifecycle at each reverse step and
after gradient validation, then releases its charge. Container sizing uses the
current interpreter's object sizes and bounded entry counts; this declaration
does not constitute a measurement of allocator overhead or retained trace data.

Nested reservations retain the lifecycle policy of every active parent. A child
cannot disable an ancestor's cancellation or extend its deadline. This also
covers replay invoked by result-constructor validation inside an active AD scope.
Child disposal restores its prior context; forked children start with no inherited
active scope and still reject reservation objects owned by the parent process.

Trace admission also declares retained IR, SSA, effect, alias, control and phi
record storage using the actual dataclass schemas and current interpreter sizes.
A node or array-view alias batch must enlarge its reservation before mutating the
graph. This covers alias growth even when no tangent node is added. Shared fields
are conservatively charged separately; frontend/serialization workspaces and
general allocator overhead still require separate qualification.

Program IR admits its record projection and exact compact ASCII JSON size before
allocating encoding storage. Encoding fills a fixed-size byte buffer in chunks
and checks lifecycle between chunks and after decoding. The declaration includes
the byte buffer, a temporary encoded chunk and output string; canonical sorted
keys and compact separators remain unchanged. General allocator and encoder
iterator overhead still require separate qualification.

Runtime line tracing admits its copied input and each distinct event before
retention, checks lifecycle on every Python line, and restores the previous
tracer in a finally block. Trace data transfers atomically into the enclosing
AD reservation and stays charged during IR and adjoint construction. A handoff
cannot drop declared live bytes or bypass any active scope's tighter allowance.
Source-cache and general interpreter overhead still need separate qualification.

Adjoint generation admits declared gradient, contribution and step/index storage
before building its dictionaries or result arrays. It checks lifecycle during
reverse traversal and step generation, then transfers the declaration into the
enclosing AD scope. Previously admitted IR encoding and trace buffers remain in
the destination plan. Primitive-specific numerical workspaces and allocator
overhead still require separate qualification.


Whole-program AD admits parameter-input and conversion buffers before creating
NumPy data arrays. It accepts plain one-dimensional real numeric ndarrays,
lists, tuples and ranges; custom array protocols and subclasses refuse because
their conversion allocations cannot be inferred safely. A private bounded-size
snapshot prevents sequence growth from expanding the subsequent conversion.
Numeric array inputs are copied into an admitted float64 snapshot. The declaration
includes conversion arrays, finite mask, snapshot pointer storage and container
headers, and remains charged during the AD operation. Oversized range metadata
refuses before enumeration. Callers must not concurrently resize or alter the
storage of an ndarray during a native copy; this scope does not lock caller-owned
NumPy storage. Native conversion internals and general allocator overhead still
require separate qualification.


Frontend bytecode and report digests admit the declared canonical JSON chunk
storage before encoding, then feed ASCII chunks directly into SHA-256. They
check the encoded size and lifecycle along the stream, retaining the existing
sorted-key, compact JSON digest convention. The compiler still builds its
record projection before this scope; source loading, AST/parser storage,
projection containers and encoder internals remain separate qualification work.
This streaming digest scope does not claim complete frontend memory coverage.


Source inspection observes a regular file and admits raw-byte, decoding and
line-storage declarations before their respective allocations. The reader opens
the file without blocking on a substituted pipe, checks descriptor identity,
reads only the observed byte count in bounded chunks, then probes one extra byte
and rechecks identity. Growth, truncation, replacement or unavailable input refuses.
Lifecycle checks surround reading, decoding and line materialisation. Decoding
uses Python source encoding detection and universal newlines.

The frontend locates callable blocks in those admitted lines, including standard
decorator unwrapping, lambdas, methods and lexical class names. It does not load or
replace global linecache entries. After extraction, path identity, type, size and
nanosecond modification/change timestamps must still match the observation.
Admission observes extracted line-object sizes and declares Unicode
join/dedent/strip and split storage before constructing those copies.

These are declared buffers rather than a measured allocator peak. Codec internals,
class AST/parser storage, report projections and runtime-specific metadata still
require qualification. Identity checks do not prove an immutable snapshot against
all concurrent writers. Complete frontend memory coverage is not claimed.


Sequence admission reads only the built-in container length before checking
memory and lifecycle policy. It declares the current NumPy float64/long-double
width allowance without walking the elements. Scalar validation then occurs
inside the bounded snapshot loop with cooperative checkpoints. Interrupted
entry therefore does not first traverse the whole sequence; wider or unsupported
scalar storage still refuses before conversion.


The source declaration returned to whole-program AD transfers atomically into
its enclosing reservation. Source/read/copy buffers remain in later tape,
IR and adjoint plans until that operation exits, without a release/reacquire
gap. The declaration conservatively retains temporary source buffers too.
Standalone compiler AST/report construction still needs complete retained
frontend-storage qualification. Installed Program AD replay additionally charges
its declared retained float64 primals, adjoints and parameter gradients before
numerical value-map construction. Gradient replay checks unary and binary
output-shape metadata against the source or inferred broadcast shape. This
retained declaration is supplemented by source-derived workspace declarations
for native multi-dot, matrix-power, bounded pseudoinverse, Gaussian/LU/solve and
compact convolution/correlation, interpolation, diag/diagflat, cumulative
scans/differences, static-grid trapezoid, product/moment reductions and bounded
2x2 spectral, order-statistic and compact stencil replay. These
include matrix-chain products, reverse prefix powers, projector squares and
projector-construction temporaries before numeric replay. Cumulative declarations
also cover source/reverse buffers and simultaneously live shape, coordinate,
prefix-index or difference-term storage; their shared borrowed metadata parser
refuses invalid source counts and output selections before materialization.
Trapezoid admission uses borrowed SSA source/target shapes and validates grid
labels without allocating the grid. It includes shaped source/cotangent copies,
coordinates, integration buffers and conservative reverse-accumulation storage.
Actual finite segment-width checks remain at the numerical kernel. Product and
moment admission additionally includes group storage and Vec headers, current
group values/contributions and conservative reverse accumulation. Static moment
correction denominators use the actual kernel rule before group allocation;
numerical singularity checks remain at their original owners. Selector and
order-statistic admission includes indexed groups, outer Vec headers, separate
validation/index-order scratch and two-slot selected reverse output. Axis/q
metadata uses its original parser before group construction; strict-order
refusal and interpolation formulas remain numerical responsibilities. Stencil
layout validation scans shape and spacing labels without allocating their
vectors, then declares source/reverse buffers, optional coordinate grid and
shape/target coordinates before materialization. Coefficient storage remains
fixed on the stack; numerical overflow still refuses at its original owner. Sequential effects
share the maximum declared kernel workspace under the same owner. Other
primitive temporaries, metadata-map capacity and returned JSON retention remain
unqualified; these declarations are not measured allocator-peak evidence.

Native Program AD metadata parsing now validates the full JSON without a
retained generic DOM and selects borrowed fields before typed decoding. It
preserves last-key-wins duplicate handling, escaped strings and positional
records. Borrowed raw-token grammar, fixed-buffer finite-number checks and
primitive integer/boolean decoding avoid serde's optional bigint conversion.
A quote-aware depth scan rejects excessive nesting before raw parsing. Explicit
type guards also keep malformed fields out of that conversion path.
A separate owned metadata policy admits serde scratch and fallible
string/vector growth; PyO3 charges metadata and numerical declarations to the
same reservation. Numeric-only policies retain their existing callback contract.
Standalone callers still need an explicit host policy, and these declarations
do not measure allocator overhead or guarantee that vendor allocations cannot
abort. Public parser and installed-wheel regressions cover refusal, recovery,
schema preservation and nested owner restoration; current runtime qualification
remains separate.

Native replay separately admits effect-ordering buffers, owned symbol bytes,
indexed parameter labels and returned label-vector headers. Flattened
parameter-target records are admitted before numeric replay and reserved once;
appends refuse any capacity outside that declaration. These metadata requests
inherit parent policy without changing numeric-only callback declarations.
Public forward and gradient regressions include long symbols, observed budget
boundaries, array metadata refusal before numeric admission and real recovery.
Full payload preflight, map allocator overhead and runtime qualification remain
required; these declarations are not a measured allocator peak.

The installed Rust dense and sparse XY Hamiltonian entries also use shared
memory reservations before output vectors are allocated. Native dimension,
element and byte arithmetic refuses overflow; sparse storage counts diagonal
and nonzero-pair triplets before reserving vector capacity. Allocation failures
become Python memory errors. Owner deadlines/cancellation are checked before
allocation, during basis iteration and before return; failure disposes the
native scope. Inputs must remain unchanged throughout construction.

These entries require the control-package policy to be importable. Native
output charges add to an enclosing plan, and can conservatively tighten its
effective allowance. Both native construction and the Python dense exporter
reject nonfinite constructed values even when individual inputs were finite.
Installed-engine parity and measured performance belong to the exact candidate
source; historical speedup figures do not qualify this admission overhead.

The native XY selector accepts exactly zero anisotropy. A nonzero XXZ request
uses the Python path in automatic mode; explicit Rust selection refuses it.
After the existing sparsity filter selects Pauli terms, simplification removes
only exact zero coefficients, preserving small selected interactions. Native
frequency construction uses the same canonical cutoff as the Python mapper.

Finite coupling symmetrisation halves each operand before addition, so a
representable Pauli coefficient does not overflow during averaging. Pauli
exports reject nonfinite inputs and overflowed XXZ coefficients before
constructing the operator. A representable Pauli coefficient can still yield
an unrepresentable dense matrix; dense output retains its separate finite-value
check.

Traced `diag` vector construction and `diagflat` admit their complete output
pointer storage before materialisation, including fixed source list, constructor
tuple and retained list. An enormous offset cannot bypass admission by producing
few derivative nodes. These paths check cancellation/deadlines during diagonal
placement and retain the conservative container peak declaration through result
creation. This declaration does not measure Python allocator internals or
unrelated numerical workspace.

Traced broadcasting also admits output pointer containers before repetition,
plus source and materialised broadcast index buffers on ranked-array routes.
Empty broadcast outputs retain their existing empty-array semantics.

Traced repeat and tile calculate target shapes before allocating output indices,
then admit numeric index buffers, boxed indices and pointer containers through
the existing owner. Zero-sized targets preserve empty-view semantics without
creating intermediate repeated arrays.

Constant-padding shape validation computes dimensions from static widths and
validates constant broadcasting without constructing output arrays. Numeric
layout construction separately reserves source and padded index/constant
buffers before calling NumPy. This is declared buffer admission; returned
derivative-rule state and interpreter/trace-object overhead require their
separate lifetime accounting.

Traced padding owns its output pointer containers, numeric layout, constant
scalar records and per-cell tangent vectors before building the padded field.
Record declarations use interpreter object/field/array-header sizes and the
actual static constant labels; they are not measured allocator peaks. The fixed
source list avoids growth beyond its admitted slots, lifecycle checks run while
cells are built, and charges remain with the main owner through result creation.
Nested numeric-layout admission can conservatively charge overlapping buffers.

Insertion shape validation derives dimensions and constant broadcasting from
static metadata before constructing source/output indices. Numeric insertion
layout separately reserves its source, output and marker buffers. Scalar and
length-one vector selectors retain their distinct NumPy broadcasting rules.

Traced insertion also admits constant scalar records, per-cell tangent vectors
and output pointer containers before creating inserted cells. Those declarations
remain charged through result construction, with lifecycle checks during cell
creation and disposal on objective failure. Interpreter metadata sizes are
schema declarations; allocator peaks and full primitive workspace qualification
require separate measurements. Numeric layout admission can conservatively
charge overlapping buffers.

Direct padding and insertion value/JVP/VJP callbacks readmit their declared
float64 layout, output, index-selection and boolean-mask buffers against live
capacity at each invocation. A limit changed after rule construction therefore
also governs transform execution. Plain ndarray operand-size errors are checked
before dtype conversion. Transform scopes release charges on success or error;
returned arrays and idle cached rule layouts are caller-held, rather than
permanently charged to this process ledger. Opaque operand protocols and general
allocator overhead still require separate qualification.

Direct getitem, take, take-along-axis and delete value/JVP/VJP callbacks use the
same live transform admission and cooperative lifecycle checks. Take layout
construction declares source indices, output indices, selector storage and
non-axis coordinate vectors before materialising the NumPy selection. Output
size is derived from index insertion or axis broadcasting metadata; registry
shape validation also admits its numeric layout before NumPy checks bounds.
This does not yet qualify all general getitem/delete layout or opaque selector
conversion workspaces.

Traced take and take-along-axis selections retain declared numeric selection,
output pointer, boxed local-index and source-lineage container storage through
result construction. Fixed output slots and cooperative per-cell checks precede
trace-array construction. Take resolves source lineage once for its view rather
than rebuilding a source-size tuple for every selected element. Declarations
use actual interpreter index/header sizes; they are not measured allocator peaks.

Deletion layout construction and registry shape validation admit source-sized
index/output upper bounds and required NumPy boolean-mask/selector workspaces
before materialising the layout. Traced deletion also retains its admitted
output pointer containers through result construction, using fixed slots and
per-cell lifecycle checks. Scalar, one-element integer-array and unit-step slice deletion do not declare a
source-axis-sized mask that NumPy does not construct. These conservative buffer
declarations do not measure allocator peaks or qualify opaque selector conversion.

Getitem layout admission counts output slots from basic slice lengths and the
broadcasted advanced-index shape. Boolean masks contribute their selected count
and nonzero-coordinate workspace. Array selectors are copied under their own
admission before counting, and numerical selection uses that immutable snapshot;
original ellipsis/newaxis placement and NumPy result ordering remain intact.
Direct layout and registry shape consumers admit source/output storage before
selection. Traced getitem retains numeric, pointer, boxed-index and lineage
container declarations through result construction. General Python selector
conversion and interpreter overhead remain separate qualification work.

Getitem selector snapshot ownership now spans metadata counting and actual
selection, including error paths and trace-buffer handoff. The caller's mutable
boolean mask can change after the snapshot without changing the admitted or
selected slots. Snapshot and numeric-layout scopes can conservatively overlap
in their declared selector charges; neither scope measures allocator peaks.

Static indexing-rule source dimensions are checked against native integer
addressability, and flattened int64 index storage is checked before selector
copying or layout construction. An unaddressable dimension still refuses when
another dimension is zero; scalar and ordinary zero-extent shapes keep their
existing semantics. Native-address failures use the same allocation-refusal
category as execution capacity failures, before Python range-length or NumPy
shape conversion can overflow.


Direct matrix-power value/JVP/VJP callbacks declare source conversions, output
and multiplication workspaces before NumPy materialisation. Derivatives also
admit the retained `abs(power)` matrix list, so a large positive or negative
exponent can refuse before creating that list. They inherit the active owner's
cancellation/deadline and checkpoint between power and accumulation operations;
charges dispose on return or failure. The existing product and inverse formulas
are preserved. `tests/test_program_ad_linalg_memory.py` supplies public numeric,
capacity, native-size, singular-input recovery and real parent-cancellation
cases. These declarations do not measure vendor LAPACK scratch, Python container
peaks or returned-array retention; native/WASM parity and runtime qualification
remain separate evidence requirements.


Inverse and vector/matrix RHS solve adjoint generation admit their visible
pullback matrices before constructing numeric inputs. Inverse declares the
matrix, inverse, cotangent, two products and negated output. Solve declares its
matrix and signed/unsigned matrix-adjoint outputs plus RHS, solution, cotangent
and RHS adjoint. Input list/name references and boxed numeric conversion are
included. These scopes inherit the active capacity/deadline/cancellation policy
and check lifecycle after native operations; charges dispose on success or
failure. The public adjoint tests in `tests/test_program_ad_linalg_memory.py`
include independent value/gradient oracles and cap refusal at actual pullback
entry followed by retry. Internal LAPACK workspace and full process peaks still
require separate qualification.


Determinant value/JVP/VJP callbacks declare source conversion, finite-validation
masks, products and outputs before inspecting numeric data. The shared cofactor
callback declares source/output matrices, row-deleted and column-deleted minors,
including old/new minor overlap, and delete masks. It checks inherited lifecycle
policy between minor operations. Trace determinant construction separately
admits the stacked tangent tensor, output tangent, list references and expanded
NumPy view metadata before materialisation. Public registered callbacks and the
adjoint facade retain empty/scalar/matrix determinant semantics and use the
original cofactor implementation. Their tests include huge virtual-buffer
refusal, independent differentials, malformed-input recovery and cancellation
after an actual minor determinant return. Vendor LAPACK workspace and full
allocator/process peaks remain separate qualification requirements.


Inverse registered value/JVP/VJP callbacks declare numeric input conversions,
validation masks, inverse/products and output storage. Determinant and inverse
callbacks reject boolean, complex, object and opaque inputs before conversion,
and reject non-finite inputs inside their admitted validation scope. This agrees
with the finite-input requirement of bounded native replay. Finite supported
formulas and singular inverse errors retain their original meaning.

Inverse trace construction separately admits primal/inverse/product buffers,
the tangent tensor, NumPy stack-view metadata, temporary scalar lists and output
trace-array containers. It checkpoints after the native inverse and between
individual derivative products. Public tests include empty/scalar/nonsymmetric
value/JVP/VJP oracles, refusal of huge virtual inputs, malformed-input recovery,
and real trace-entry capacity refusal before native inverse followed by retry.
These declarations do not qualify vendor LAPACK workspace, full allocator peaks
or performance.


Solve registered and fixed-shape value/JVP/VJP callbacks admit real input
conversion, validation masks, RHS solutions/products and matrix-gradient
buffers before numeric materialisation. Vector and matrix RHS retain their
original solve differentials. Trace solve construction admits numeric and
tangent tensors, per-parameter result arrays and their stacked copy, array
metadata, temporary list/name references and output containers. It checkpoints
after native solves and between parameter/output iterations. Public tests
provide explicit vector/matrix RHS oracles, singular/non-finite recovery and
trace-entry capacity refusal before a native solve followed by retry. These
source declarations do not establish vendor LAPACK workspace bounds, full
allocator peaks, native interruption, parity or performance qualification.


Matrix-power trace construction reserves primal/output matrices, input tangent
tensors, per-parameter JVP results and their stacked copy before materialising
them. Reference lists and NumPy array metadata are included. Actual value/JVP
callbacks reserve their algorithm workspaces under that trace owner; output
containers hand off to the active context. Lifecycle checkpoints follow the
value callback, surround each JVP, and precede output trace construction.
Public composed positive/negative-power tests provide analytical values and
gradients, trace-entry refusal before the value callback followed by recovery,
and cancellation after an actual JVP return. Vendor scratch, full allocator
peaks and cross-backend runtime qualification remain separate evidence.


Direct matrix-power value/JVP/VJP callbacks also admit finite-validation masks
and reject non-finite primal, tangent and cotangent inputs, including exponent
zero. This tightens the raw callback input boundary to agree with native replay;
finite supported formulas and singular negative-power errors are unchanged.


Whole-program AD metadata accepts a plain list or tuple of exact `Parameter`
records with plain string names and boolean trainability. It checks alignment
before copying, admits fixed record/reference/uniqueness storage, then grows the
name declaration before each private record copy. Default generated names are
sized before generation. The fixed-length snapshot detects source-list growth
or shrinkage and preserves independent names/trainable masks. Its charge remains
with the active context through result construction. Opaque metadata sequences
and record/name subclasses refuse before invoking their protocols. This tightens
the resource-boundary input contract; ordinary finite supported calculations
retain their formulas. Public tests cover list/tuple metadata, a nontrainable
coordinate, Unicode names, opaque refusal, over-budget names, malformed alignment
and duplicate-name recovery.


Trace and reverse-adjoint generation declarations include finite-validation
masks used by their result records. The public attached-gradient accessor
rechecks the captured plain one-dimensional float64 layout and parameter count,
admits fixed-size private output and validation-mask storage, and validates
finite copied values before returning them. Mutated captured arrays therefore
cannot silently return nonfinite gradients or reshape/retype the admitted copy.
It detects layout drift at reservation entry and preserves an independent result
copy. Public tests cover post-capture NaN/infinities, shape/dtype/length mutation,
real reservation-entry layout drift, cleanup and finite retry. These declarations
do not measure allocator peaks or establish concurrency safety for external
unsafe mutation of native buffers.


Result validation also declares the frozen-coordinate index and selected-value
buffers. Attached-gradient copies recheck that non-trainable entries remain
zero, with cooperative checks while walking the captured mask. Post-capture
mutation cannot re-enable a frozen coordinate; a public finite retry restores
the original masked gradient. Arithmetic-overflow tests exercise actual value
and tangent rejection through whole-program and adjoint entry points.


Fixed-shape multi-dot rule construction validates dimensions without creating
zero-filled operands. Its value/JVP/VJP callbacks admit numeric conversion,
finite masks, matrix-chain planning tables, possible subchain intermediates,
outputs, operand views/references and VJP basis/retained gradient buffers before
splitting numeric inputs. Size products use Python integers before native
addressability checks. Lifecycle checkpoints surround repeated chain and basis
operations; ordinary product and derivative formulas retain their order.
Public tests provide two-matrix and scalar vector-endpoint analytic differentials,
absurd-shape refusal before numeric operands, malformed/nonfinite recovery and
cancellation after a real native chain return. These conservative declarations
do not measure vendor BLAS or allocator peaks. Backend qualification remains a separate requirement.


Multi-dot trace construction also admits primal arrays and flattened input,
operand tangent stacks and their concatenation, retained per-parameter JVPs and
their stacked copy. It uses the same matrix-chain workspace declarations as the
actual callbacks, with nested derivative owners and cooperative checkpoints.
Array output containers hand off to the context; scalar output directly uses
its node owner. Public facade tests include the expanded vector-matrix-vector
objective and actual trace-entry refusal before a native chain followed by retry.


Diagonal rule construction computes output dimensions without allocating a
coordinate tuple. Actual `diag` and `diagflat` callbacks admit numeric
conversion, finite-validation masks, source/output workspaces, array headers
and diagonal coordinate metadata before numerical construction or pullback.
Rectangular extraction builds only the selected coordinate interval under that
owner. Linear `diag` JVPs retain their primal-independent input convention.
Diagflat shape products use Python integers instead of native-width NumPy
products. Public tests cover offset insertion/extraction differentials,
rectangular sparse pullbacks, nonfinite/malformed recovery and huge factory
metadata followed by callback refusal before numerical or coordinate storage.
Full allocator peaks and backend qualification remain separate evidence.


Both trace reshape normalisation and fixed-shape derivative rules use Python
integer products for known dimensions and final size preservation. Native-width
wrap cannot make a huge layout appear to preserve a small input or make a
nonzero inferred-axis product appear zero. Public facade and direct rule tests
include products congruent to one/zero modulo 2^64, with finite inferred-layout
retry and independent gradients. Their existing remote owner cohort includes
strict typing, documentation and unchanged 100% coverage thresholds.


Trace reduction, cumulative, product, broadcasting and predicate shape counts
use Python integer arithmetic. Predicate containers reject shape products that
would alias their item count after native integer overflow. Numeric `numpy.prod`
continues to compute the original differentiable value product.


Compact trace rules admit primal inputs, tangent inputs, per-coordinate result
arrays, their stacked copy, finite-validation masks and Python container/array
metadata before materialising those buffers. Returned value and tangent storage
remains charged to the trace context through result creation; temporary input
storage is released after evaluation. Each value/JVP callback completion checks
cancellation and deadlines before conversion or the next coordinate. Algorithm
workspace inside each callback remains the responsibility of its numerical owner;
these declarations do not measure vendor allocator scratch.


Cumulative value/JVP/VJP rules admit plain numeric input storage, conversion
copies, finite masks, array metadata and linear source-sized workspaces before
conversion or numerical dispatch. Cumulative inputs and outputs must be finite;
opaque array protocols refuse without conversion. Pullback and product loops
check the active execution lifecycle. Static cumulative factories retain their
existing non-empty-source contract, while a constant non-empty trace can have
zero differentiated parameters. Native kernel compilation and measured parity
remain separate qualification evidence.


Native singular-value replay declares its source, owned matrix, returned U/VT
and spectrum storage, plus reverse contributions, before numeric evaluation.
Shared layout validation rejects malformed metadata and operands before this
admission callback. Decomposition-library scratch and in-kernel interruption
remain unqualified; this declaration alone does not establish a complete solver
memory limit. Public forward/reverse budget, refusal and independent-retry
fixtures are authored in `program_ad_svd_memory.rs` and await native execution.

Replay validation declares its branch, region and phi hash tables before fallible
reservation. Shape, scalar value, numeric value and adjoint maps use the same
checked table bound. The adjoint map reserves its complete target capacity once;
new keys cannot grow beyond that admitted capacity. These are conservative table
layout declarations for the maintained Rust/WASM targets, based on the standard
library hash-table bucket and control-byte layout, rather than allocator peak
measurements. Public branch and gradient regressions exercise refusal before
numeric admission and recovery with unchanged mathematical results.

The browser replay kernel applies a 64 MiB product ceiling to cumulative declared
storage for each call. Owned input/output copies, parser metadata, replay tables
and numerical requests share that charge; an explicit Rust caller may tighten
it with `replay_value_and_gradient_with_memory_budget`. This is a policy ceiling,
not observed browser capacity or a memory reservation. Parent admission remains
binding, allocation failures refuse, and a refused FFI call leaves output bytes
unchanged. Independent subsequent calls start with a fresh charge.
