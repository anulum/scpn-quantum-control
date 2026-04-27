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

## SC-NeuroCore compatibility bridge

Campaign code may depend on the compatibility import surface:

```python
from scpneurocore.bridge import (
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
)
```

Those functions must not silently generate random matrices. Each loader
must return, or point to, an auditable source artifact containing at
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

## Matrix invariants

Current Kuramoto-XY circuits require:

- `K_nm` is square.
- `K_nm` is finite.
- `K_nm` has zero diagonal.
- `K_nm` is symmetric.
- `K_nm` is non-negative.
- `omega` is finite and has shape `(N,)`.
- `theta0`, if present, is finite and has shape `(N,)`.
- `layer_assignments`, if present, has length `N`.

Lag, directionality, hypergraph terms, and non-reciprocal couplings
belong in metadata or a future specialised schema. They must not be
silently forced into this symmetric Kuramoto-XY contract.

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

## Test coverage

The artifact contract is tested in `tests/test_qpu_data_artifact.py`.
The tests cover:

- real artifact round-trip and hashes
- synthetic artifact rejection by the publication gate
- missing timestamp/replay rejection
- invalid diagonal, negative, or directed `K_nm`
- shape and metadata rejection
- SC-NeuroCore datastream adaptation
- schema-version rejection

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
