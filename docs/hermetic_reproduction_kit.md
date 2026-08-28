# Hermetic external reproduction kit

Skeptic-facing **command manifest + digest contract** so public numbers can be
rebuilt without trusting narrative docs alone. The **core** kit is local-first
and must not require cloud GPUs or QPUs.

Module: `scpn_quantum_control.hermetic_reproduction_kit`  
Related: metamorphic checks, no-advantage posture, and reviewer
commands (compose).

## Rules

| Kind | Meaning |
|---|---|
| `core_local` | Offline/local-first; digests derived from declared fixture payloads |
| `optional_extra` | Optional framework extras; not required for core kit |
| `hardware_gated` | QPU/live paths; owner-ticket gated |
| `refuse_invent_green` | Invented green QPU/cloud digests forbidden |

The schema identifier is `hermetic_reproduction_kit.v1`. Catalogue order is
stable, entry identifiers are unique, and every `core_local` row carries at
least one digest while setting `requires_qpu=False`. Non-core rows carry an
explicit reason. The module validates these constraints when records are
created and again when a serialised registry is audited.

Claim boundary:

> hermetic reproduction kit contract only; core_local entries are offline/
> local-first command metadata with digests derived from declared fixture
> payloads; hardware_gated and refuse_invent_green rows never invent green
> QPU or cloud-GPU reproduction claims

## Catalogue and value objects

`list_hermetic_kit_entry_ids()` returns all canonical identifiers in catalogue
order. `get_hermetic_kit_entry(entry_id)` trims surrounding whitespace and
raises `ValueError` for a blank or unknown identifier.
`iter_hermetic_kit_entries(kind=None, core_only=False)` returns immutable rows;
the two filters compose, and a filter with no matches returns an empty tuple.

The exported immutable value objects are:

| Type | Contract |
|---|---|
| `KitDigestSpec` | Non-empty label, lowercase 64-character SHA-256 digest, and optional exact fixture bytes whose digest must match at construction |
| `HermeticKitEntry` | Identifier, kind, summary, argv, repository-relative working directory, QPU posture, digest tuple, reason, and claim boundary |
| `DigestCheckResult` | Label, match/refusal state, expected and actual digests, message, and claim boundary |

Each object exposes `to_dict()` for JSON-ready data. Fixture bytes are never
embedded in the serialised digest specification; `has_fixture_payload` records
only their availability.

## Registry construction and integrity

`build_hermetic_reproduction_kit()` returns the schema, claim boundary,
aggregate counts, and serialised entries. It does not execute any stored argv.
`assert_hermetic_kit_integrity(payload=None)` verifies a caller-supplied mapping
or a freshly built registry. It rejects:

- an absent, empty, or non-list entry collection;
- non-mapping rows, blank identifiers, or unknown kinds;
- `core_local` rows that require a QPU;
- non-core rows without a reason;
- non-zero blank/core-QPU counters; and
- declared entry counts that differ from the actual row count.

The validator returns a shallow dictionary copy. It is a schema and policy
check, not proof that a command was executed or that hardware reproduced a
result.

## Digest verification

`sha256_hex_of(content)` accepts bytes or bytearray values and returns a
lowercase SHA-256 digest. Other input types raise `TypeError`.

`verify_digest(...)` returns a `DigestCheckResult` rather than raising for
ordinary evidence failures. Blank or malformed expected digests, absent
content, and mismatches all set `matched=False` and `refused=True`. Only an
exact digest match sets `matched=True` and `refused=False`; a blank label raises
`ValueError`.

`verify_kit_entry_digests(entry_id, content_by_label=None,
use_fixture_payloads=True)` checks every digest on one catalogue row. Caller
content takes precedence over built-in fixture bytes. Disabling fixture
fallback makes missing caller content fail closed. Rows without digests return
an empty tuple, while `refuse_invent_green` rows raise `ValueError` instead of
manufacturing successful evidence.

`fixture_payload(label)` exposes only the exact declared fixture bytes for a
known label. Unknown, blank, and payload-free labels raise `ValueError`.

## Offline probes

`probe_hermetic_kit_entry(entry_id, unknown_policy="raise")` reports catalogue
metadata without invoking the command. Known rows include eligibility, refusal,
kind, QPU posture, argv, working directory, and digest count. Under
`unknown_policy="refuse"`, an unknown identifier returns a deterministic
`found=False`, `refused=True` mapping. The default `"raise"` policy rejects it;
any other policy value is invalid.

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
assert kit["blank_entry_count"] == 0

# Known core entry + digest match from fixture payload
results = verify_kit_entry_digests("kit:core.transform_algebra_smoke")
assert all(r.matched for r in results)

# Blank / mismatch digests fail closed
blank = verify_digest(label="x", expected_sha256_hex="", content=b"data")
assert blank.refused is True

# Caller content overrides fixture fallback and fails closed on mismatch
changed = verify_kit_entry_digests(
    "kit:core.transform_algebra_smoke",
    content_by_label={"fixture:transform_algebra_smoke": b"changed"},
    use_fixture_payloads=False,
)
assert changed[0].matched is False
assert changed[0].refused is True

# Invent-green QPU digests refused
probe = probe_hermetic_kit_entry("kit:refuse.invent_green_qpu_digest")
assert probe["refused"] is True

# Unknown identifiers can be represented as an explicit refusal
unknown = probe_hermetic_kit_entry("kit:not-present", unknown_policy="refuse")
assert unknown["found"] is False
assert unknown["refused"] is True
```

Importing the module, listing or filtering the catalogue, serialising records,
checking digests, probing metadata, or validating the registry does not launch
a subprocess, access a network, submit a provider job, consume credentials,
mutate fixtures, or promote a hardware claim. Stored argv values are data only.

## Bounded product status

Shipped: versioned kit contract · command and digest schema · pure digest-check
mode · core local-only catalogue · public docs.

Open: generator from the scorecard evidence catalogue · one-command runner ·
CI job · attested-result verification steps. API and external-reviewer contract
documentation is covered here; independent external reproduction evidence is
still separate from documentation completeness.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
