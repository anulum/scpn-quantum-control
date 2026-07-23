# Metamorphic AD verification catalogue (BL-46)

This page is the operator-facing guide for **gradient correctness beyond
one-off examples**: a versioned catalogue of metamorphic laws, fail-closed
refuse paths (including invent-green hardware formal proofs), and pure residual
band checks.

Related surfaces:

- Transform algebra suite: `scpn_quantum_control.differentiable_transform_algebra`
- BL-53 unsuitable / anti-silent fixtures: [Unsuitable scenario registry](unsuitable_scenario_registry.md)
- Module: `scpn_quantum_control.metamorphic_ad_verification`

## Outcomes

| Outcome | Meaning |
|---|---|
| `executable_local` | Pure residual APIs can evaluate pass/fail locally |
| `evidence_gated` | Claim maps to an in-tree audit suite; not invent-green without running it |
| `permanent_boundary` | Explicit refuse (anti-silent-wrong class) |
| `refuse_invent_green` | Marketing formal-proof / hardware ITP claims refused |

Claim boundary:

> metamorphic AD verification catalogue only; executable_local laws are pure
> or local residual checks, permanent_boundary and refuse_invent_green rows
> never promote hardware formal-proof or silent-wrong recovery claims

## Public API

```python
from scpn_quantum_control.metamorphic_ad_verification import (
    evaluate_linearity_residual,
    probe_metamorphic_law,
    build_metamorphic_ad_registry,
    assert_metamorphic_registry_integrity,
)

# Pure metamorphic residual (real shipped check)
lin = evaluate_linearity_residual(1.0, 2.0, 3.0)
assert lin.passed is True

# Anti-silent / invent-green refuse
boundary = probe_metamorphic_law("law:anti_silent.di_jl_compiled_tape")
assert boundary.refused is True
formal = probe_metamorphic_law("law:formal.hardware_interactive_proof")
assert formal.refused is True

# Unknown IDs fail closed
try:
    probe_metamorphic_law("law:missing")
except ValueError:
    pass

registry = assert_metamorphic_registry_integrity(build_metamorphic_ad_registry())
```

## Bounded product status

Shipped: S46.0 catalogue · pure residual checks · fail-closed probe · public docs.

Open: S46.1 claim-ledger map generator · S46.2–S46.5 full suites wiring · S46.6 CI drift job.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
