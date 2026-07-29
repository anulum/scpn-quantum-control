# Bit-exact polyglot parity certificates (BL-49 / P1)

Versioned **externally checkable certificate** product for the Rust Program AD
replay moat: family catalogue (scalar → spectral bounds), certificate schema,
digest build/verify helpers. Digests prove sample bundle identity — they do
**not** claim full NumPy parity.

Module: `scpn_quantum_control.polyglot_parity_certificate`

## Rules

| Rule | Behaviour |
|---|---|
| Certificate schema | `polyglot_parity_certificate.v1` |
| Product schema | `polyglot_parity_certificate_product.v1` |
| Default family | `scalar_interpreter_replay` |
| Sample bit-exact | Supported cert with matching digests, `max_abs_error == 0.0` |
| Boundary / catalogue families | Unsupported; typed blockers; verify refuses invent-green pass |
| Blank/unknown family or schema | Fail closed |
| Full NumPy parity | Never claimed |

Claim boundary:

> Polyglot parity certificate product only; digests prove sample bundle identity
> for published families; does not claim full NumPy parity; unsupported
> Rust/feature paths fail closed with typed blockers; ambient
> program_ad_rust_bridge remains experimental_workbench under BL-97; residual
> CLI (S49.3), committed CI corpus (S49.4), and BL-38 feed (S49.6) open honestly

## Public API

```python
from scpn_quantum_control.polyglot_parity_certificate import (
    assert_polyglot_parity_product_integrity,
    build_polyglot_parity_product_registry,
    build_sample_certificate,
    list_parity_family_ids,
    verify_certificate,
)

assert "scalar_interpreter_replay" in list_parity_family_ids()
reg = assert_polyglot_parity_product_integrity(build_polyglot_parity_product_registry())
cert = build_sample_certificate("scalar_interpreter_replay")
decision = verify_certificate(cert)
assert decision.passed is True
assert decision.observed_max_abs_error == 0.0

boundary = build_sample_certificate("elementwise_primitive_parity")
assert verify_certificate(boundary).passed is False
```

## Families (S49.0)

| Family | Support |
|---|---|
| `scalar_interpreter_replay` | sample_bitexact |
| `value_and_gradient_replay` | sample_bitexact |
| `registry_metadata_mirror` | sample_bitexact |
| `elementwise_primitive_parity` | boundary_unsupported |
| `linalg_primitive_parity` | boundary_unsupported |
| `spectral_bounds_parity` | catalogue_only |

`list_parity_family_ids()` returns these identifiers in deterministic catalogue
order. `get_parity_family(family_id)` performs exact lookup after trimming
outer whitespace; blank and unknown identifiers raise `ValueError` rather than
creating a green result. `iter_parity_families(support=...)` returns immutable
`ParityFamily` records, optionally filtered by `sample_bitexact`,
`boundary_unsupported`, or `catalogue_only`. A filter with no matches returns an
empty tuple.

Each family record contains its identifier, title, summary, support posture,
owning module path, BL-97 stability class, inventory date, and the shared claim
boundary. `ParityFamily.to_dict()` exposes the same fields as a JSON-ready
mapping.

## Canonical payloads and digests

`canonical_json_bytes(payload)` serialises a mapping as compact UTF-8 JSON with
sorted keys and no insignificant whitespace. Non-mapping input raises
`ValueError`; values must still be JSON serialisable, and the underlying JSON
error is not hidden.

`digest_payload(payload)` returns the lowercase SHA-256 hex digest of those
canonical bytes. A digest establishes the identity of the supplied sample
payload only. It is not a signature, provider attestation, benchmark result,
or proof of parity outside that exact payload.

## Certificate construction and parsing

`build_sample_certificate(family_id, sample_id="sample-0")` uses deterministic
input and reference fixtures:

- `sample_bitexact` families receive input, Python-reference, and Rust digests,
  `supported=True`, no blockers, and `max_abs_error=0.0`;
- `boundary_unsupported` families receive input/reference digests and a typed
  feature-support blocker; and
- `catalogue_only` families receive a typed missing-corpus blocker.

Blank or unknown family identifiers and blank sample identifiers raise
`ValueError`. Construction does not execute Rust, compile a kernel, contact a
provider, or read an external corpus; it certifies the module's deterministic
sample fixtures.

`certificate_from_dict(payload)` validates a JSON-compatible mapping and
returns an immutable `PolyglotParityCertificate`. It rejects non-mappings,
unknown families or schemas, missing/blank identifiers, non-string digests,
non-numeric errors, non-boolean support, non-sequence blockers, and blank claim
boundaries. The resulting record additionally enforces 64-character lowercase
hex digests, non-negative errors, supported/blocker consistency, and the
zero-error requirement for supported certificates.

`PolyglotParityCertificate.to_dict()` returns every certificate field,
including blockers and the claim boundary, in a serialisable mapping.

## Verification decisions

`verify_certificate(certificate, expect_supported=None)` accepts either a
validated certificate object or a mapping. It returns a
`CertificateVerifyDecision` with one of three outcomes:

| Outcome | Meaning |
|---|---|
| `passed` | supported sample family; recomputed digests match; error is exactly zero |
| `failed` | schema, expectation, digest, support, or error mismatch |
| `refused` | family is deliberately unsupported or catalogue-only |

For supported families the verifier rebuilds the deterministic sample and
compares every digest, `max_abs_error`, and support flag. Unsupported families
never become `passed`; their blockers are returned, with a fallback blocker if
the certificate omitted one. `expect_supported` is an additional caller
assertion and a mismatch yields `failed`.

The decision record exposes the family/sample identifiers, outcome, pass flag,
human-readable reason, blockers, observed error, and claim boundary.
Verification does not establish full NumPy parity, performance superiority,
hardware equivalence, or compatibility with an unlisted sample corpus.

## Registry and integrity

`map_parity_public_surfaces()` groups catalogue families by owning module and
returns deterministic role, stability, support, family-id, and claim-boundary
rows. `build_polyglot_parity_product_registry()` combines those surfaces with
the product/certificate schemas, counts, default family, canonical family rows,
and policy note.

`assert_polyglot_parity_product_integrity(payload=None)` validates either an
explicit payload or a freshly built registry. It rejects missing/empty family
lists, non-mapping rows, blank or duplicate identifiers, invalid support
postures, missing default family, drift from the canonical family set,
non-zero blank counts, inconsistent family counts, and certificate-schema
drift. It returns a shallow registry mapping after validation.

## Exported schemas and types

`POLYGLOT_PARITY_CERTIFICATE_SCHEMA` identifies certificate mappings;
`POLYGLOT_PARITY_PRODUCT_SCHEMA` identifies product registries; and
`POLYGLOT_PARITY_CLAIM_BOUNDARY` is carried through family, certificate,
decision, surface, and registry records. The exported literal vocabularies are
`FamilySupport` and `VerifyOutcome`; the immutable records are `ParityFamily`,
`PolyglotParityCertificate`, and `CertificateVerifyDecision`.

## Operational boundaries

Importing, listing, building, parsing, or verifying this product does not run a
real Rust/PyO3 parity campaign, mutate the Program AD registry, publish a
certificate, populate the open multi-family CI corpus, feed BL-38, or promote a
compatibility/performance claim. Those actions require their own governed
artefacts and evidence gates.

## Bounded product status

Shipped: S49.0 family list · S49.1 certificate schema + digests · S49.2 product
generator over sample harnesses · S49.5 public docs · fail-closed unsupported
paths.

Open: S49.3 `scpn-bench polyglot-parity-certificates` CLI · S49.4 committed
multi-family sample CI corpus · S49.6 BL-38 decision pack feed.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
