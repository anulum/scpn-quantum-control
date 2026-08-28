# Public API stability programme

Fail-closed **public-vs-internal** stability catalogue for adopters. Declares a
**narrow durable SemVer-intent surface** (`stable_core` + curated CLI entry
points) and classifies workbench / private paths so bulk package exports are
**not** silently reported as guaranteed-stable.

Module: `scpn_quantum_control.public_api_stability`  
Policy sources: `DEPRECATIONS.md`, internal v1 stability gate (CEO scope still
required before a hard v1.0 freeze).

## Rules

| Class | Meaning |
|---|---|
| `semver_stable` | Declared durable contract; guaranteed-stable when active |
| `experimental_workbench` | Documented / importable, **not** SemVer-guaranteed |
| `deprecated` | Public shim staged for removal (replacement + horizon required) |
| `internal` | Outside the public contract |

Unknown or blank symbol ids **fail closed** (never invent-stable).  
Removal / rename / signature break of active public symbols is **refused**
without a prior deprecation record.

Claim boundary:

> public API stability programme only; semver_stable covers the narrow
> declared durable contract (stable_core + curated entry points);
> experimental_workbench and bulk package exports are not guaranteed-stable;
> unknown/blank symbol ids fail closed; removal of public symbols without a
> deprecation record is refused

## Public API

```python
from scpn_quantum_control.public_api_stability import (
    assert_public_api_stability_integrity,
    build_public_api_stability_registry,
    classify_api_path,
    deprecated_public,
    list_public_api_symbol_ids,
    probe_deprecation,
    validate_breaking_change,
    version_compatibility_note,
)

reg = assert_public_api_stability_integrity(build_public_api_stability_registry())
assert reg["blank_entry_count"] == 0

public = classify_api_path("scpn_quantum_control.stable_core.Problem")
assert public.guaranteed_stable is True

internal = classify_api_path("scpn_quantum_control._private_helpers")
assert internal.visibility == "internal"
assert internal.guaranteed_stable is False

probe = probe_deprecation("scpn_quantum_control.kuramoto")
assert probe.is_deprecated is True
assert probe.replacement_target == "oscillatools"

refuse = validate_breaking_change(
    "scpn_quantum_control.stable_core.Problem",
    change_kind="remove",
)
assert refuse.allowed is False
```

### Catalogue discovery

`list_public_api_symbol_ids()` returns the declared identifiers in canonical,
deterministic order. `iter_public_api_symbols()` returns the corresponding
immutable `PublicApiSymbolRecord` values and accepts independent
`stability_class`, `deprecation_state`, and `visibility` filters. A filter with
no matches returns an empty tuple; it does not widen the query.

`get_public_api_symbol(symbol_id)` performs exact lookup after trimming outer
whitespace. Blank and unknown identifiers raise `ValueError`; the function
never synthesises a stable record for an unregistered name.

Each `PublicApiSymbolRecord` exposes the symbol identifier, stability class,
owner surface, deprecation state, visibility, summary, optional replacement
and removal horizon, inventory date, and the non-promotional claim boundary.
`to_dict()` returns the same fields in a JSON-ready mapping.

### Path classification

`classify_api_path(path_id)` applies the following order:

1. Exact catalogue entries retain their declared class and visibility.
2. private modules, tests, internal docs, coordination paths, and fixtures are
   classified `internal` and never guaranteed stable.
3. every other undeclared path is treated as an experimental workbench path,
   not as `semver_stable`.

The returned `PathClassification` records the stripped input, visibility,
best-fit class, `guaranteed_stable` decision, explanation, and whether the
decision came from the catalogue. Blank input raises `ValueError`.

### Deprecation and breaking-change policy

`probe_deprecation(symbol_id)` returns a `DeprecationProbe` for a declared
symbol. Active entries carry empty replacement, horizon, and warning fields.
Deprecated entries carry all three fields from the canonical record. Blank or
unknown identifiers fail through the exact catalogue lookup.

`validate_breaking_change(symbol_id, change_kind=...)` accepts `remove`,
`rename`, or `signature_break` and returns a `BreakingChangeDecision`:

| Surface state | Decision |
|---|---|
| active public | refused; a prior deprecation record is required |
| deprecated public | allowed at the recorded removal horizon |
| internal | allowed because it is outside the SemVer contract |

Unknown symbols and change kinds raise `ValueError`. An `allowed` result is a
policy decision only; it does not edit code, rewrite imports, publish a
release, or prove downstream compatibility.

`deprecated_public(...)` builds a decorator that emits `DeprecationWarning`
with `stacklevel=2` on every wrapped call and then returns the wrapped result.
The symbol identifier, replacement target, and removal horizon must all be
nonblank. The decorator does not mutate the static catalogue, so a live
decorated callable still needs a separately governed catalogue record.

### Registry and integrity

`build_public_api_stability_registry()` emits the schema, claim boundary,
class/visibility counts, zero-blank marker, canonical rows, and policy note.
`assert_public_api_stability_integrity(payload=None)` validates either that
payload or a freshly built registry. It rejects:

- absent, empty, or non-list symbol collections;
- non-mapping rows, blank identifiers, invalid classes or visibility;
- duplicate identifiers or drift from the canonical identifier set;
- deprecated rows without both replacement and removal horizon;
- internal `semver_stable` rows; and
- inconsistent `symbol_count` or `blank_entry_count` values.

The integrity check returns a shallow registry mapping after validation. It
does not dynamically crawl package exports; the full live-inventory generator
and drift job remain open work.

### Constants and record types

`PUBLIC_API_STABILITY_SCHEMA` identifies serialised registry payloads.
`PUBLIC_API_STABILITY_CLAIM_BOUNDARY` is copied into rows, probes, decisions,
and compatibility notes so downstream displays retain the same scope warning.
The exported literal vocabularies are `StabilityClass`, `DeprecationState`,
`Visibility`, and `BreakingChangeKind`; the immutable record classes are
`PublicApiSymbolRecord`, `PathClassification`, `DeprecationProbe`, and
`BreakingChangeDecision`.

## Version compatibility / migration note

See `version_compatibility_note()` and `DEPRECATIONS.md`:

* Pre-v1.0: SemVer pre-1.0 clause still applies until CEO-scoped freeze.
* Kuramoto/accel shims: migrate imports to `oscillatools` /
  `oscillatools.accel` / `oscillatools.neural_operator` before the next major.

`version_compatibility_note()` returns these statements as structured data,
including its schema, pre-v1 caveat, v1 intent, policy-document reference,
migration note, claim boundary, and inventory date. The date is provenance,
not a runtime freshness probe.

## Operational boundaries

This module is a deterministic policy catalogue. Importing or querying it does
not contact a provider, inspect installed entry points, modify
`DEPRECATIONS.md`, emit a warning, execute a migration, create a tag, or freeze
the package. Warnings occur only when a callable wrapped by
`deprecated_public()` is invoked. Any v1 freeze, catalogue expansion, removal,
release, or compatibility claim remains subject to its separate governance and
evidence gates.

## Bounded product status

Shipped: public-vs-internal catalogue · inventory query · deprecation decorator
+ probe · fail-closed unknown/blank · breaking-change refusal without
deprecation · version-compatibility note + docs.

Open: full inventory generator + CI drift job against live package
`__all__` · CEO v1.0 scope decision / tag · rewrite of DEPRECATIONS.md public
surface section to match the narrow freeze.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
