# Advantage / no-advantage language protocol

This page is the operator-facing guide for **when “advantage” language is
allowed**. Default posture is **no-advantage / research observation**. Decisive
protocol modules remain claim-gated evidence paths — never invent-green marketing.

Related surfaces:

- Existing runners (compose, do not fork):  
  `benchmarks.advantage_protocol`, `benchmarks.decisive_advantage_protocol`,  
  `forecasting.neural_operator_advantage`
- Route and refusal governance: [Governed route matrix](governed_route_matrix.md),
  [Unsuitable scenario registry](unsuitable_scenario_registry.md)
- Module: `scpn_quantum_control.advantage_language_protocol`

## Rules

| Language status | Meaning |
|---|---|
| `no_advantage_default` | Default public posture; no marketing advantage triggers |
| `research_observation` | Claim-bounded research wording only (no marketing triggers) |
| `decisive_gated` | May enter decisive evidence path; **not** a proven advantage claim |
| `refuse_advantage_language` | Explicit refuse path for ungoverned advantage wording |

**Default:** any free-text claim containing marketing advantage triggers without a
bound protocol id is **refused**.

Claim boundary:

> advantage-language governance only; default is no-advantage / research
> observation; decisive or advantage wording requires an explicit protocol
> identity and never invents green quantum-advantage marketing claims

## Public API

```python
from scpn_quantum_control.advantage_language_protocol import (
    issue_no_advantage_certificate,
    probe_advantage_language,
    get_advantage_protocol,
    build_advantage_language_registry,
    assert_advantage_language_registry_integrity,
)

# Default certificate (always non-promotional)
cert = issue_no_advantage_certificate(context="release_notes")
assert cert.language_status == "no_advantage_default"

# Ungoverned advantage language fails closed
blocked = probe_advantage_language("We claim quantum advantage.")
assert blocked.allowed is False

# Neutral wording allowed under default
ok = probe_advantage_language("Local residual audit under claim_boundary.")
assert ok.allowed is True

# Decisive protocol may enter evidence path (not invent-green)
decisive = probe_advantage_language(
    "Entering decisive Kuramoto-XY quantum advantage question.",
    protocol_id="protocol:decisive.kuramoto_xy",
)
assert decisive.language_status == "decisive_gated"
assert decisive.allowed is True  # enter path only — not a proven claim

registry = assert_advantage_language_registry_integrity(
    build_advantage_language_registry()
)
```

## API reference

All public objects are exported by
`scpn_quantum_control.advantage_language_protocol`. Registry and `to_dict()`
payloads contain JSON-ready primitives.

### Types and constants

| API | Contract |
|---|---|
| `ProtocolLanguageStatus` | Literal status: `no_advantage_default`, `research_observation`, `decisive_gated`, or `refuse_advantage_language`. |
| `ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA` | Stable registry schema identifier, currently `advantage_language_protocol.v1`. |
| `ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY` | Shared non-promotional boundary copied into records, certificates, probes, and registries. |

### Data models

#### `AdvantageProtocolRecord`

Frozen, slotted catalogue row containing the protocol id, language status,
summary, evidence-module pointers, reason, and claim boundary. The default row
must have an empty reason; every non-default row requires one. Evidence-module
entries must be non-blank. `to_dict()` serialises the tuple as a list.

#### `NoAdvantageCertificate`

Frozen, slotted non-promotional certificate with a deterministic id, statement,
bound protocol, and claim boundary. Its status is always
`no_advantage_default`. Construction rejects blank fields or any other status;
`to_dict()` returns the complete certificate.

#### `AdvantageLanguageProbeResult`

Frozen, slotted language-gate decision containing the original claim, allowed
flag, effective status, protocol id, matched trigger phrases, reason, and claim
boundary. A refuse-status result cannot be allowed, and every trigger must be
non-blank. `to_dict()` converts triggers to a list.

### Catalogue access

| Function | Parameters | Returns | Failure behavior |
|---|---|---|---|
| `list_advantage_protocol_ids()` | None | Protocol ids in canonical order. | No failure for the built-in catalogue. |
| `get_advantage_protocol(protocol_id)` | Non-blank protocol id. | Matching `AdvantageProtocolRecord`. | Raises `ValueError` for blank or unknown ids. |
| `iter_advantage_protocols(*, language_status=None)` | Optional status filter. | Stable tuple of matching records. | Returns an empty tuple when no row matches. |

The catalogue also includes
`protocol:entanglement.initial_state_observation`, a research-observation row for
population-matched initial-state comparisons. Catalogue evidence modules are
pointers; their presence is not proof that their protocol gates passed.

### Certificates and trigger detection

`issue_no_advantage_certificate(*, context="public_claim",
protocol_id="protocol:default.no_advantage")` normalises spaces in `context`
to underscores and returns a deterministic certificate. The protocol must be
known and cannot be the refuse row. Blank contexts, unknown ids, and refuse-row
bindings raise `ValueError`.

`find_advantage_language_triggers(claim_text)` performs a case-insensitive
ordered scan for the module's governed marketing phrases. It returns a tuple
of matched phrases and raises `TypeError` for `None`.

```python
from scpn_quantum_control.advantage_language_protocol import (
    find_advantage_language_triggers,
    issue_no_advantage_certificate,
)

certificate = issue_no_advantage_certificate(context="release notes")
assert certificate.certificate_id.startswith("no_advantage:release_notes:")
assert find_advantage_language_triggers("Proven quantum advantage")
```

### Language probe

`probe_advantage_language(claim_text, protocol_id=None, *,
unknown_policy="raise")` applies these rules:

| Binding | Neutral/research wording | Trigger wording |
|---|---|---|
| No protocol or blank id | Allowed under default posture | Refused through the ungoverned row |
| `no_advantage_default` | Allowed | Refused |
| `research_observation` | Allowed | Refused |
| `decisive_gated` | Allowed into the evidence path | Allowed into the evidence path, not declared proven |
| `refuse_advantage_language` | Refused | Refused |

Unknown protocol ids raise `ValueError` by default. With
`unknown_policy="refuse"`, they produce a structured refusal instead. Any
other policy value raises `ValueError`; a `None` claim raises `TypeError`.

### Registry and integrity

`build_advantage_language_registry()` assembles the schema, claim boundary,
protocol count, per-status counts, blank count, and canonical records.
`assert_advantage_language_registry_integrity(payload=None)` validates a
supplied registry or builds the canonical one. It raises `ValueError` for an
empty or malformed protocol list, blank/unknown statuses, missing reasons on
non-default rows, or inconsistent blank/protocol counts.

## Safety and side effects

- All APIs are pure and deterministic for their supplied inputs.
- The module does not run benchmarks or QPU/provider jobs, access credentials,
  mutate evidence, publish wording, or promote a claim.
- `decisive_gated` means only that wording may enter an explicit evidence path;
  it never means quantum advantage has been demonstrated.
- A certificate governs wording only. It is not scientific, performance,
  release, legal, or marketing approval.

## Catalogue seeds

- `protocol:default.no_advantage` — default posture
- `protocol:s2.scaling_matrix` — research observation (S2)
- `protocol:decisive.kuramoto_xy` — decisive_gated (decisive module)
- `protocol:neural_operator.structural_surrogate` — classical surrogate research
- `protocol:ungoverned.advantage_language` — refuse catch-all

## Bounded product status

Shipped: protocol catalogue · facade composition · default no-advantage
certificate · partially integrated language-gate probe.

Open: full decisive-evidence schema wiring · scorecard-acceptance scoring hook ·
hermetic reproduction fixtures · fuller promotion-tool hooks.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
