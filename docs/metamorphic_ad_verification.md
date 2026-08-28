# Metamorphic AD verification catalogue

This page is the operator-facing guide for **gradient correctness beyond
one-off examples**: a versioned catalogue of metamorphic laws, fail-closed
refuse paths (including invent-green hardware formal proofs), and pure residual
band checks.

Related surfaces:

- Transform algebra suite: `scpn_quantum_control.differentiable_transform_algebra`
- Unsuitable / anti-silent fixtures: [Unsuitable scenario registry](unsuitable_scenario_registry.md)
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

## API reference

All public objects are exported by
`scpn_quantum_control.metamorphic_ad_verification`; registry and `to_dict()`
payloads contain JSON-ready primitives.

### Types and constants

| API | Contract |
|---|---|
| `LawKind` | Literal family: `metamorphic_identity`, `fd_agreement_band`, `anti_silent_wrong`, or `formal_boundary`. |
| `LawOutcome` | Literal outcome: `executable_local`, `evidence_gated`, `permanent_boundary`, or `refuse_invent_green`. |
| `METAMORPHIC_AD_VERIFICATION_SCHEMA` | Stable registry schema, currently `metamorphic_ad_verification.v1`. |
| `METAMORPHIC_AD_CLAIM_BOUNDARY` | Shared non-promotional boundary copied into rows, results, and registries. |

### Data models

`MetamorphicLawRecord` is a frozen, slotted catalogue row containing the law
id, kind, expected outcome, relation, evidence-module pointers, reason,
positive default tolerance, and claim boundary. Executable rows must have no
reason; every non-executable row requires one. `to_dict()` serialises evidence
modules as a list.

`MetamorphicCheckResult` is a frozen, slotted probe/residual result containing
the law id, pass flag, optional residual and tolerance, operator message,
refusal flag, and claim boundary. Refused results cannot pass; residuals must
be non-negative and tolerances positive. `to_dict()` returns the complete
JSON-ready result.

### Catalogue and probes

| Function | Behavior |
|---|---|
| `list_metamorphic_law_ids()` | Returns canonical ids in stable order. |
| `get_metamorphic_law(law_id)` | Returns one row; raises `ValueError` for blank or unknown ids. |
| `iter_metamorphic_laws(*, kind=None, expected_outcome=None)` | Returns rows satisfying both optional filters. |
| `probe_metamorphic_law(law_id, *, unknown_policy="raise")` | Returns deterministic catalogue metadata without running external evidence suites. |

Known executable laws probe as ready for local residual evaluation.
Evidence-gated laws return `passed=False`, `refused=False` until their named
suite is run. Permanent and invent-green boundaries return refused results.
Unknown ids raise by default or return a structured refusal with
`unknown_policy="refuse"`; other policy values raise `ValueError`.

### Residual evaluators

`evaluate_linearity_residual(f_a, f_b, f_ab, *, law_id=..., tolerance=None)`
evaluates `abs((f_a + f_b) - f_ab)`. Its default band is `1e-12`.

`evaluate_chain_rule_residual(outer_at_inner, inner_derivative,
composite_derivative, *, law_id=..., tolerance=None)` evaluates
`abs(outer_at_inner * inner_derivative - composite_derivative)`. Its default
band is `1e-10`.

Both functions require their canonical executable law, finite numeric inputs,
and a positive override tolerance. They return pass/fail results and perform no
automatic differentiation, framework execution, or hardware work.

```python
from scpn_quantum_control.metamorphic_ad_verification import (
    evaluate_chain_rule_residual,
    evaluate_linearity_residual,
)

assert evaluate_linearity_residual(1.0, 2.0, 3.0).passed
assert evaluate_chain_rule_residual(2.0, 3.0, 6.0).passed
assert not evaluate_linearity_residual(1.0, 2.0, 3.1).passed
```

### Registry and integrity

`build_metamorphic_ad_registry()` assembles the schema, claim boundary, law
count, per-outcome counts, blank count, and canonical rows.
`assert_metamorphic_registry_integrity(payload=None)` validates a supplied
registry or builds the canonical one. It raises `ValueError` for empty or
malformed rows, invalid outcomes, missing non-executable reasons, blanks, or
inconsistent counts.

## Safety and side effects

- All catalogue/probe/residual APIs are pure and deterministic.
- Evidence-module entries are pointers, not evidence that a suite passed.
- No API runs AD frameworks, benchmarks, theorem provers, QPU/provider jobs,
  network calls, credential access, evidence mutation, or publication.
- Passing a local residual is not a hardware formal proof, universal gradient
  guarantee, scientific promotion, or release approval.

## Bounded product status

Shipped: S46.0 catalogue · pure residual checks · fail-closed probe · public docs.

Open: S46.1 claim-ledger map generator · S46.2–S46.5 full suites wiring · S46.6 CI drift job.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
