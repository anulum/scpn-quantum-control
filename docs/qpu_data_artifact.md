# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — QPU Data Artifact Contract

# QPU Data Artifact Contract

The QPU data artifact is the publication gate between source-facing
repositories and `scpn-quantum-control`. It prevents campaign scripts
from inventing coupling matrices locally and makes every QPU submission
traceable back to a source, replay, or curated dataset.

The schema version is:

```text
scpn-quantum-control.qpu-data-artifact.v1
```

## Repository roles

| Repository | Responsibility |
|------------|----------------|
| SC-NeuroCore | Provides source-facing bridge functions and neural/SCPN payloads. |
| Phase Orchestrator | Extracts phases, frequencies, coupling semantics, layer assignments, and provenance. |
| SCPN Quantum Control | Validates the final artifact, compiles circuits, runs hardware jobs, and analyses QPU counts. |

Quantum Control consumes artifacts. It must not generate unnamed
campaign matrices as a fallback path.

Legacy frontier campaign scripts still accept `.npy` parameter files for
operator convenience, but those files are only a transport cache. The
generator in `scripts/frontier_campaign_2026/generate_params.py` fails
closed by default when a source bridge is missing or when a bridge does
not provide `omega`. Deterministic synthetic arrays require the explicit
`--allow-synthetic` flag and write `PARAMETER_PROVENANCE.json` with
`source_mode="synthetic"` entries. Such arrays are smoke-test inputs
only; they are not publication-safe QPU source material.

## SC-NeuroCore canonical bridge

Campaign code may depend on the canonical source-facing bridge namespace:

```python
from scpn_neurocore.bridge import (
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
)
```

The prior unseparated package spelling is not the active tracked
package. Those functions must not silently generate random matrices.
Each loader must return, or point to, an auditable source artifact containing at
least `K_nm`, `omega`, `source_mode`, `source_name`, `normalization`,
`extraction_method`, and either `source_timestamp` or `replay_id` for
publication use.

Synthetic or deterministic smoke payloads are allowed only when the
metadata says so explicitly with `source_mode="synthetic"`,
`"simulation"`, or `"fixture"`. Quantum Control will reject those modes
when `require_publication_safe=True`.

## Required fields

| Field | Meaning |
|-------|---------|
| `domain` | Domain family, e.g. `connectome`, `plasma`, `power-grid`, `scpn`. |
| `source_name` | Human-readable source identifier. |
| `source_mode` | One of `recorded`, `replay`, `curated`, `derived`, `synthetic`, `simulation`, `fixture`. |
| `K_nm` | Coupling matrix following Phase Orchestrator row/column semantics. |
| `omega` | Natural frequency vector. |
| `theta0` | Optional initial phase vector. |
| `layer_assignments` | Optional per-oscillator layer labels. |
| `normalization` | Named normalisation method. Existing schema spelling is kept as an API field. |
| `extraction_method` | Method or pipeline that produced `K_nm` and `omega`. |
| `source_timestamp` | Source acquisition timestamp, when available. |
| `replay_id` | Replay identifier, required when no timestamp exists. |
| `metadata` | Domain-specific non-secret metadata. |
| `hashes` | SHA-256 hashes for numerical arrays. |
| `artifact_sha256` | SHA-256 hash of the serialised payload. |

Loaders treat the hash fields as integrity assertions, not decorative
metadata. If a payload supplies `K_nm_sha256`, `omega_sha256`,
`theta0_sha256`, or `artifact_sha256`, Quantum Control recomputes the
corresponding digest and rejects stale values before the artifact can be
compiled into a circuit. Omitted array digests are filled deterministically for
new in-memory artifacts. The `hashes` map is reserved for recognized numerical
payload digests only; unknown keys and malformed SHA-256 values are rejected.
The top-level `artifact_sha256` must also be a lowercase SHA-256 digest before
payload comparison. Operator notes and non-digest annotations belong in
`metadata`.

Numerical arrays, layer assignments, metadata, and hash maps are
defensive-copied and marked read-only after validation, so callers cannot
mutate a validated artifact in-place and silently invalidate its hashes.
Layer assignments must be supplied as a sequence of non-empty strings; scalar
strings and implicit per-label coercion are rejected before hashing.
Metadata is frozen recursively; serialisation thaws immutable containers back to
JSON-native dictionaries and lists. Metadata must use string keys and
JSON-compatible scalar/list/object values; non-finite floats and opaque Python
objects are rejected before artifact hashing. `metadata` and `hashes` must be
mapping objects at both direct-constructor and loader boundaries; pair-list
coercion is rejected.

`source_timestamp` and `replay_id`, when supplied, must be non-empty strings.
They are trimmed before the whole-artifact digest is computed so equivalent
provenance identifiers do not produce whitespace-dependent artifact identities.
Required identity fields (`domain`, `source_name`, `source_mode`,
`normalization`, and `extraction_method`) follow the same fail-closed string
contract: no implicit `str(...)` coercion, and whitespace is normalised before
hashing. The same contract is enforced for loader input; `from_dict()` does not
coerce non-string identity fields into publication artifact identity strings.
Top-level loader input must be a mapping before schema validation; list or
pair-list payloads are rejected with artifact-contract errors.

## Matrix invariants

Current Kuramoto-XY circuits require:

- `K_nm` is square.
- `K_nm` describes at least one oscillator.
- `K_nm` is finite.
- `K_nm` has zero diagonal.
- `K_nm` is symmetric.
- `K_nm` is non-negative.
- `omega` is finite and has shape `(N,)`.
- `theta0`, if present, is finite and has shape `(N,)`.
- `layer_assignments`, if present, has length `N`.

Diagonal self-couplings must be exactly zero in the stored artifact. Symmetry
checks use absolute tolerance only (`atol=1e-12`, `rtol=0.0`) so large coupling
magnitudes cannot hide directed terms behind relative tolerance. Lag,
directionality, hypergraph terms, and non-reciprocal couplings belong in
metadata or a future specialised schema. They must not be silently forced into
this symmetric Kuramoto-XY contract.
Negative couplings are rejected exactly; preprocessing must not rely on
validator tolerance to carry signed coupling semantics into a non-negative
Kuramoto-XY artifact.
Numerical payloads must be real numeric values before conversion to `float64`;
strings, bytes, booleans, and complex values are rejected instead of being
implicitly coerced into coupling, frequency, or phase values.
Ragged or otherwise non-rectangular numeric containers are rejected with an
artifact-contract error before they can leak backend conversion messages.

## Publication gate

The validator separates source modes into two classes.

Publication-safe modes:

- `recorded`
- `replay`
- `curated`
- `derived`

Smoke-test modes:

- `synthetic`
- `simulation`
- `fixture`

Synthetic artifacts are valid for interface tests, but
`require_publication_safe=True` rejects them. Publication-safe artifacts
also require either `source_timestamp` or `replay_id`.

## Python API

```python
from scpn_quantum_control.bridge import (
    artifact_from_arrays,
    read_qpu_data_artifact,
    validate_qpu_data_artifact,
    write_qpu_data_artifact,
)

artifact = artifact_from_arrays(
    domain="connectome",
    source_name="c_elegans_subnetwork",
    source_mode="curated",
    K_nm=K_nm,
    omega=omega,
    normalization="max coupling to one",
    extraction_method="phase-orchestrator domain compiler",
    replay_id="connectome-v1",
)

validate_qpu_data_artifact(artifact, require_publication_safe=True)
write_qpu_data_artifact("artifact.json", artifact)
loaded = read_qpu_data_artifact("artifact.json")
```

SC-NeuroCore datastream smoke payloads can be adapted with:

```python
from scpn_quantum_control.bridge.qpu_data_artifact import QPUDataArtifact

artifact = QPUDataArtifact.from_scpn_datastream_payload(payload)
assert artifact.is_synthetic
```

That adapter defaults to `source_mode="synthetic"` because the current
datastream is deterministic smoke data, not a recorded source artifact.
The top-level datastream payload must be a mapping before schema inspection.
`seed` is required because it becomes the replay identity for deterministic
smoke payloads.
It routes `knm` and `omega_rad_s` through the same numeric payload validator as
all other artifact constructors; string, boolean, complex, and ragged numeric
coercions are not accepted. `layer_ids` are routed through the same
layer-assignment validator; non-string or blank layer labels are rejected.

## Test coverage

The artifact contract is tested in `tests/test_qpu_data_artifact.py`.
The tests cover:

- real artifact round-trip and hashes
- stale array-hash and artifact-hash tamper rejection
- synthetic artifact rejection by the publication gate
- missing timestamp/replay rejection
- invalid diagonal, negative, or directed `K_nm`
- sub-tolerance non-zero diagonal `K_nm` rejection
- sub-tolerance negative `K_nm` rejection
- large-scale directed `K_nm` rejection without relative-tolerance masking
- empty zero-oscillator `K_nm` rejection
- shape and metadata rejection
- numeric payload rejection for string, boolean, and complex coercion
- SC-NeuroCore datastream top-level mapping rejection
- SC-NeuroCore datastream missing-seed replay identity rejection
- SC-NeuroCore datastream numeric payload coercion rejection
- SC-NeuroCore datastream layer-label coercion rejection
- ragged numeric payload rejection with artifact-contract errors
- SC-NeuroCore datastream adaptation
- schema-version rejection
- defensive-copy/read-only array immutability
- defensive-copy/read-only layer-assignment immutability
- defensive-copy/read-only metadata and hash-map immutability
- recursive metadata freezing with JSON-native serialisation
- metadata rejection for non-string keys, non-finite floats, and opaque values
- hash-map rejection for unknown keys and malformed digest strings
- top-level artifact-hash syntax rejection before payload comparison
- provenance identifier type, blank-value, and whitespace normalisation checks
- required identity-field type and whitespace normalisation checks
- loader rejection for non-string required identity fields
- loader rejection for non-mapping top-level payloads
- constructor and loader rejection for non-mapping metadata/hash containers
- layer-assignment rejection for scalar strings, non-string entries, and blanks

Frontier interfaces that are not implemented yet are guarded in
`tests/test_frontier_interface_guards.py`; those paths fail loudly
instead of returning synthetic scientific values.

## Performance and language path

`QPUDataArtifact` is a validation and serialisation boundary. It is not
a numerical hot loop. Per `docs/language_policy.md`, it is exempt from a
compiled-language path: the work is array validation, JSON serialisation,
and SHA-256 hashing implemented by NumPy, Python's standard library, and
compiled C/Rust internals.

No benchmark is claimed for this module. Its correctness criterion is
schema validation and reproducible hashes, not throughput.
