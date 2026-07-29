<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Theory-Hook Promotion API

Import the public surface from `scpn_quantum_control.analysis`:

```python
from scpn_quantum_control.analysis import (
    TheoryHookStatus,
    build_theory_hook_promotion_report,
    get_theory_hook_promotion,
    list_theory_hook_promotions,
    run_theory_hook_evidence,
)
```

The API is read-only and local. It does not access credentials, provider APIs,
QPU services, lab instruments, or control outputs.

## Constants

### `THEORY_HOOK_PROMOTION_SCHEMA`

Serialization identifier: `scpn.theory-hook-promotion.v1`.

### `THEORY_HOOK_PROMOTION_BOUNDARY`

Global non-claim applied to every policy and evidence row. It explicitly
excludes hardware validation, differentiability, criticality certification,
quantum advantage, consciousness evidence, clinical interpretation, and
actuation authority.

## Enums

### `TheoryHookTier`

| Member | Value | Meaning |
|---|---:|---|
| `BOUNDED` | `B` | Small evidence-gated research diagnostic |
| `RESEARCH_ONLY` | `D` | Semantics are insufficient for promotion |

### `TheoryHookRole`

Permitted roles are `optional_control_constraint`,
`synthetic_inverse_problem`, `classical_local_baseline`,
`mutual_information_diagnostic`, `resource_theory_diagnostic`, and
`spectral_diagnostic`.

The role is a use boundary, not an admission flag. In particular, the optional
control-constraint role remains `admitted_for_control = false` until its future
promotion requirements are independently satisfied.

### `TheoryHookStatus`

| Member | Value |
|---|---|
| `BOUNDED_CANDIDATE` | `bounded_candidate` |
| `DIAGNOSTIC_ONLY` | `diagnostic_only` |
| `RESEARCH_ONLY` | `research_only` |

## Data classes

### `TheoryHookPromotionRecord`

Immutable policy record with these fields:

| Field | Type | Contract |
|---|---|---|
| `hook_id` | `str` | Stable unique identifier |
| `title` | `str` | Human-readable label |
| `module` | `str` | Owning import path |
| `tier` | `TheoryHookTier` | BL-98 evidence tier |
| `role` | `TheoryHookRole` | Only permitted role |
| `status` | `TheoryHookStatus` | Current promotion state |
| `differentiable` | `bool` | Always false in schema v1 |
| `evidence_fixture` | `str` | Exact local fixture |
| `allowed_claims` | `tuple[str, ...]` | Narrow supported statements |
| `forbidden_claims` | `tuple[str, ...]` | Claims never granted by local evidence |
| `promotion_requirements` | `tuple[str, ...]` | Evidence needed for reconsideration |
| `references` | `tuple[str, ...]` | Primary-literature identifiers |

Properties `admitted_for_control` and `admitted_for_publication_claim` always
return false. `as_dict()` produces JSON-ready values and includes those negative
capabilities explicitly.

Construction rejects blank identifiers, empty policy lists, duplicate entries,
unsupported differentiability, and any tier-D record not marked
`research_only`.

### `TheoryHookEvidenceRecord`

Immutable result for one fixture:

- `hook_id`: corresponding policy identifier;
- `passed`: conjunction of all named checks;
- `fixture`: exact fixture description;
- `checks`: unique `(name, bool)` pairs;
- `metrics`: unique JSON-ready `(name, value)` pairs.

Construction rejects mismatched aggregate status, blank names, and duplicate
keys. `as_dict()` renders checks and metrics as mappings.

### `TheoryHookPromotionReport`

Contains the schema, global boundary, canonical policy records, one evidence
row per record, and a SHA-256 content digest. `passed` is true only when every
fixture passes. `as_dict()` is deterministic and JSON-ready.

The digest covers schema, claim boundary, policies, and evidence, excluding the
digest field itself.

## Registry functions

### `list_theory_hook_promotions()`

Returns the immutable six-record registry in canonical evidence order. It does
not execute numerical fixtures.

### `get_theory_hook_promotion(hook_id)`

Returns one policy record. Unknown identifiers raise `KeyError`; the function
does not create a permissive default.

```python
record = get_theory_hook_promotion("bipartite_mutual_information")
assert record.status is TheoryHookStatus.RESEARCH_ONLY
assert record.admitted_for_publication_claim is False
```

## Evidence functions

### `run_theory_hook_evidence()`

Executes six tiny deterministic local fixtures and returns a tuple of
`TheoryHookEvidenceRecord` objects. Evidence order must exactly match registry
order or the function raises `RuntimeError`.

The routine uses dense exact local calculations and can take several seconds.
It never contacts a provider.

### `build_theory_hook_promotion_report()`

Runs the fixtures, joins them to the policy registry, and computes the content
digest.

### `render_theory_hook_promotion_markdown(report)`

Returns a concise deterministic Markdown custody record ending in a newline.
Use the CLI for committed JSON and Markdown bytes:

```bash
python scripts/run_theory_hook_promotion_evidence.py
python scripts/run_theory_hook_promotion_evidence.py --check
```

`--check` exits non-zero when either file is missing or differs byte-for-byte
from fresh output.

## Legacy theory APIs

The promotion registry does not change import compatibility for
`compute_qsl`, `learn_hamiltonian`, `build_koopman_generator`,
`compute_quantum_phi`, `magic_at_coupling`, or `compute_sff`. It does constrain
their documented interpretation.

Most importantly, `IntegratedInformationPhi` has no IIT implementation. Exact
Hamiltonian inputs require `allow_mutual_information_proxy=True`; results use
`minimum_bipartite_mutual_information`, set `phi_available = 0.0`, and never
return a `phi` key.

For rationale, mathematical boundaries, primary references, and future gates,
see [Theory-Hook Promotion Matrix](../theory_hook_promotion.md).
