# Hermetic external reproduction kit (BL-55)

Skeptic-facing **command manifest + digest contract** so public numbers can be
rebuilt without trusting narrative docs alone. The **core** kit is local-first
and must not require cloud GPUs or QPUs.

Module: `scpn_quantum_control.hermetic_reproduction_kit`  
Related: BL-46 metamorphic checks, BL-65 no-advantage posture, BL-25 reviewer
commands (compose).

## Rules

| Kind | Meaning |
|---|---|
| `core_local` | Offline/local-first; digests derived from declared fixture payloads |
| `optional_extra` | Optional framework extras; not required for core kit |
| `hardware_gated` | QPU/live paths; owner-ticket gated |
| `refuse_invent_green` | Invented green QPU/cloud digests forbidden |

Claim boundary:

> hermetic reproduction kit contract only; core_local entries are offline/
> local-first command metadata with digests derived from declared fixture
> payloads; hardware_gated and refuse_invent_green rows never invent green
> QPU or cloud-GPU reproduction claims

## Public API

```python
from scpn_quantum_control.hermetic_reproduction_kit import (
    assert_hermetic_kit_integrity,
    build_hermetic_reproduction_kit,
    fixture_payload,
    list_hermetic_kit_entry_ids,
    probe_hermetic_kit_entry,
    verify_digest,
    verify_kit_entry_digests,
)

kit = assert_hermetic_kit_integrity(build_hermetic_reproduction_kit())
assert kit["core_requires_qpu_count"] == 0

# Known core entry + digest match from fixture payload
results = verify_kit_entry_digests("kit:core.transform_algebra_smoke")
assert all(r.matched for r in results)

# Blank / mismatch digests fail closed
blank = verify_digest(label="x", expected_sha256_hex="", content=b"data")
assert blank.refused is True

# Invent-green QPU digests refused
probe = probe_hermetic_kit_entry("kit:refuse.invent_green_qpu_digest")
assert probe["refused"] is True
```

## Bounded product status

Shipped: S55.0 kit contract · S55.1 command/digest schema · S55.4 digest-check mode
(pure) · core local-only catalogue · public docs.

Open: S55.2 generator from BL-25 · S55.3 one-command runner · S55.5 CI job ·
S55.6 external-reviewer docs expansion · S55.7 BL-48/49 verify steps.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
