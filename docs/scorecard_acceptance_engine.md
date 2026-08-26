# Scorecard acceptance engine

Fail-closed promotion surface for the **eleven differentiable baseline-scorecard
categories**. Honest ``behind_baseline`` inventory is the default until claim-ledger
and external-comparison evidence packages exist.

Module: `scpn_quantum_control.scorecard_acceptance_engine`  
Composes category ids from `differentiable_baseline_scorecard` (does not rewrite
the committed scorecard artefact).

The engine is a pure decision-support surface. Listing, validating, or asking it
for a promotion decision does not mutate the canonical inventory, regenerate a
scorecard, execute a benchmark, contact a provider, or certify performance.

## Data layers

`ScorecardCategoryRecord` is the immutable inventory layer. It binds one
required category to its honest status, summary, evidence labels, blockers,
required evidence kinds, inventory date, and claim boundary. Construction
rejects unknown categories or statuses, blank text and labels, promoted rows
with open blockers, promoted rows without evidence, and behind-baseline rows
without blockers.

`PromoteDecision` is the immutable decision layer. It records the source and
requested statuses, whether the request is allowed, a reason, and any missing
evidence kinds. An allowed decision cannot carry missing evidence. A decision
describes the request only; it does not update a `ScorecardCategoryRecord`.

The registry payload is the aggregate layer. It carries the schema identifier,
claim boundary, category counts, zero-blank count, and JSON-ready category
rows. `to_dict()` converts immutable tuple fields into lists at the
serialisation boundary.

## Rules

| Status | Meaning |
|---|---|
| `behind_baseline` | Honest inventory; requires blockers |
| `at_baseline` / `exceeds_baseline` | Only via promote with required evidence |
| `not_comparable` | Requires impossibility evidence labels |

Promotion to ready statuses requires evidence identifiers covering:

- `claim_ledger_promoted_row`
- `external_baseline_comparison`

Each required kind must appear in at least one supplied identifier. The engine
does not parse or authenticate the referenced package; callers remain
responsible for digest, provenance, and custody validation.

Claim boundary:

> scorecard acceptance engine only; behind_baseline is an honest inventory
> state; promote refuses invent-green at_baseline/exceeds_baseline without
> required evidence digests and language-safe claims

## Lookup and filtering

`list_scorecard_category_ids()` returns all required identifiers in canonical
order. `get_scorecard_category()` returns one immutable row and raises
`ValueError` for blank or unknown input. `iter_scorecard_categories()` returns
the complete immutable tuple or filters it by a supported status while
preserving order.

All eleven canonical rows currently use `behind_baseline` and carry explicit
blockers. That state is evidence-bounded inventory, not a forecast about future
parity.

## Promotion decisions

`promote_scorecard_category()` applies a fixed fail-closed sequence:

1. Validate the category and requested status.
2. Always permit keeping or returning a row to `behind_baseline`.
3. Require at least one evidence label before declaring `not_comparable`.
4. For `at_baseline` or `exceeds_baseline`, require both evidence-kind labels.
5. Reject the bounded list of unqualified promotional phrases.
6. Return an allowed engine decision while leaving artefact regeneration to a
   separate controlled operation.

Missing evidence produces a refused decision with `missing_evidence`. Banned
language produces a refused decision with a bounded reason and no false claim
that evidence itself was missing.

## Integrity validation

`assert_scorecard_acceptance_integrity()` builds the canonical registry when no
payload is supplied, or validates a caller-provided mapping. It requires a
non-empty list of mapping-shaped rows, every required category exactly through
set coverage, supported statuses, blockers for behind-baseline rows, evidence
for ready rows, zero blank entries, and an exact category count.

Validation rejects malformed or incomplete payloads instead of treating them
as partial success. It does not authenticate evidence identifiers or compare
benchmark results; those remain separate evidence-system responsibilities.

## Public API

```python
from scpn_quantum_control.scorecard_acceptance_engine import (
    assert_scorecard_acceptance_integrity,
    build_scorecard_acceptance_registry,
    list_scorecard_category_ids,
    promote_scorecard_category,
)

reg = assert_scorecard_acceptance_integrity(build_scorecard_acceptance_registry())
assert reg["behind_baseline_count"] == 11
assert len(list_scorecard_category_ids()) == 11

# Invent-green without evidence → refuse
d = promote_scorecard_category(
    "benchmark_promotion",
    target_status="exceeds_baseline",
)
assert d.allowed is False
assert d.missing_evidence
```

## Bounded product status

Shipped:

- Complete category catalogue and honest baseline inventory
- Fail-closed promotion decisions and integrity validation
- JSON-ready record and decision mappings
- Operator guide and real-surface tests

Outside this surface:

- Scorecard artefact regeneration or dashboard presentation
- Hermetic execution of referenced comparison packages
- Evidence authentication, benchmark execution, or provider access

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
