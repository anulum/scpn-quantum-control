# Public API stability programme (BL-97 / W5)

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

## Version compatibility / migration note

See `version_compatibility_note()` and `DEPRECATIONS.md`:

* Pre-v1.0: SemVer pre-1.0 clause still applies until CEO-scoped freeze.
* Kuramoto/accel shims: migrate imports to `oscillatools` /
  `oscillatools.accel` / `oscillatools.neural_operator` before the next major.

## Bounded product status

Shipped: S97.0 public-vs-internal catalogue · inventory query · S97.2
deprecation decorator + probe · fail-closed unknown/blank · breaking-change
refuse without deprecation · S97.3/S97.4 version-compatibility note + docs.

Open: S97.1 full inventory generator + CI drift job against live package
`__all__` · CEO v1.0 scope decision / tag · rewrite of DEPRECATIONS.md public
surface section to match the narrow freeze.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
