# Advantage / no-advantage language protocol (BL-65)

This page is the operator-facing guide for **when “advantage” language is
allowed**. Default posture is **no-advantage / research observation**. Decisive
protocol modules remain claim-gated evidence paths — never invent-green marketing.

Related surfaces:

- Existing runners (compose, do not fork):  
  `benchmarks.advantage_protocol`, `benchmarks.decisive_advantage_protocol`,  
  `forecasting.neural_operator_advantage`
- BL-52 / BL-53 governance: [Governed route matrix](governed_route_matrix.md),  
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

## Catalogue seeds

- `protocol:default.no_advantage` — default posture
- `protocol:s2.scaling_matrix` — research observation (S2)
- `protocol:decisive.kuramoto_xy` — decisive_gated (decisive module)
- `protocol:neural_operator.structural_surrogate` — classical surrogate research
- `protocol:ungoverned.advantage_language` — refuse catch-all

## Bounded product status (BL-65)

Shipped: S65.0 catalogue · S65.1 facade compose · S65.2 no-advantage certificate ·
language-gate probe (partial S65.4).

Open: S65.3 full decisive evidence schema wiring · S65.5 BL-32 scoring hook ·
S65.6 BL-55 reproduction fixtures · fuller promotion-tool hooks.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
