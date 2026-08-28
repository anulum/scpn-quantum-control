# Attested result packs

Strip-resistant **content digests bound to claim axes** — never a self-asserted
“validated” badge. Local unsigned envelopes are first-class; absent keys yield
**UNGRADED**, never silent-validated.

Module: `scpn_quantum_control.attested_result_pack`  
Related: hermetic reproduction digests, hardware result packs (compose).

## Verification statuses

| Status | Meaning |
|---|---|
| `VERIFIED` | Required axes present; content digest matches recomputation |
| `STRIPPED` | Required claim axes missing/empty |
| `FORGED` | Digest present but does not match bound content |
| `UNGRADED` | Missing digest, missing required signature, or unconfigured crypto |

Claim boundary:

> attested result-pack digest contract only; digests bind inputs and claim
> axes, never a self-asserted validated badge; absent keys yield UNGRADED,
> never silent-validated hardware or marketing attestation claims

## Public API

```python
from scpn_quantum_control.attested_result_pack import (
    build_unsigned_envelope,
    default_claim_axes,
    verify_attested_envelope,
    refuse_invent_green_hardware_attestation,
)

env = build_unsigned_envelope(
    claim_id="claim.local.demo",
    claim_axes=default_claim_axes(),
    content={"metric": 0.5, "backend": "statevector"},
)
assert verify_attested_envelope(env).status == "VERIFIED"

# Tamper content → FORGED
from dataclasses import replace
forged = replace(env, content={"metric": 9.9})
assert verify_attested_envelope(forged).status == "FORGED"

# Hardware invent-green without digest → UNGRADED
assert refuse_invent_green_hardware_attestation(
    claim_id="claim.hw", has_content_digest=False
).status == "UNGRADED"
```

## Envelope and verdict records

`AttestedEnvelope` is a frozen, slot-backed record containing the claim ID,
claim axes, structured content, canonical digest, schema, optional signature
token, and shared claim boundary. Construction rejects blank claim IDs,
non-mapping axes or content, and non-empty digests that are not lowercase
64-character SHA-256 hex. `to_dict()` returns plain mappings suitable for JSON
custody without mutating the envelope.

`AttestationVerdict` carries one closed status, a non-blank reason, expected and
observed digests, the claim ID, and the boundary. Invalid status, reason, or
claim ID values fail at construction. The verifier never returns an ambiguous
truthy flag in place of these states.

## Canonical digest and claim-axis binding

`canonical_content_digest(content, claim_axes)` serialises a mapping containing
both inputs with sorted keys, compact separators, UTF-8, and non-finite floats
disabled. Mapping key order therefore does not change the digest, while any
claim-axis change does. Non-mapping or non-JSON-ready values raise rather than
falling back to string representations.

The required verification axes are `claim_unit`, `honesty_axes`, and
`backend_class`. `default_claim_axes()` supplies a bounded local set and
rejects blank units, backend classes, or an empty normalised honesty-axis list.

## Verification decision order

`verify_attested_envelope()` accepts an envelope or its mapping form and applies
the following fail-closed order:

1. Missing or empty required axes produce `STRIPPED`.
2. A missing digest produces `UNGRADED`.
3. Content that cannot be canonically digested produces `UNGRADED`.
4. A digest mismatch produces `FORGED`.
5. A required but missing signature produces `UNGRADED`.
6. A present signature without a configured validating keyring remains
   `UNGRADED`.
7. Only an intact unsigned envelope with complete axes and matching digest is
   `VERIFIED`.

Unsigned `VERIFIED` means structural and digest integrity only. It is not a
cryptographic signature, hardware provenance, scientific validation, or
marketing badge.

## Reports, mapping custody, and hardware refusal

`build_attestation_report()` accepts only a list or tuple of verdict records
and returns counts for all four statuses plus their serialised rows.
`envelope_from_mapping()` recursively materialises nested mappings and
sequences into JSON-ready values before validating the envelope; malformed
axes or content fail closed.

`refuse_invent_green_hardware_attestation()` always returns `UNGRADED`. A
digest-free claim lacks content custody, while a digest alone still lacks a
verified envelope and owner-ticket evidence. Neither path can authorise a QPU
claim.

## Operational non-effects

This module does not read keys or credentials, select a provider, submit
hardware work, sign content, validate an opaque signature, write packs, publish
claims, or mutate evidence. It performs deterministic local structure and
digest checks only.

## Bounded product status

Shipped: S48.0 policy (unsigned-first) · S48.1 canonical digest · S48.2 local
envelope · S48.4 verify statuses · S48.6 strip-resistance battery (tests).

Open: S48.3 optional signed path/keyring · S48.5 BL-32 wiring · S48.7 hermetic
kit verify step.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
