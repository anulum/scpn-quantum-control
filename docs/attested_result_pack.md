# Attested result packs (BL-48)

Strip-resistant **content digests bound to claim axes** — never a self-asserted
“validated” badge. Local unsigned envelopes are first-class; absent keys yield
**UNGRADED**, never silent-validated.

Module: `scpn_quantum_control.attested_result_pack`  
Related: BL-55 hermetic digests, hardware result packs (compose).

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

## Bounded product status

Shipped: S48.0 policy (unsigned-first) · S48.1 canonical digest · S48.2 local
envelope · S48.4 verify statuses · S48.6 strip-resistance battery (tests).

Open: S48.3 optional signed path/keyring · S48.5 BL-32 wiring · S48.7 hermetic
kit verify step.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
