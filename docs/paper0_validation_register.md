# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 validation register

# Paper 0 Validation Register

The Paper 0 register is the generated source-accounting layer for the canonical
Paper 0 ledger. It records source spans, generated validation specs, fixture
summaries, and regression tests for Paper 0 ingestion.

## Current State

| Item | Status |
| --- | --- |
| Ledger span | `P0R00001` through `P0R06211` |
| Remaining promotion work orders | `0` |
| Remaining source records | `0` |
| Remaining three-slice batches | `0` |
| Generated validation modules | `466` modules under `scpn_quantum_control.paper0` |
| Public claim boundary | Source-accounting and fixture-preservation only |

## What Completion Means

Paper 0 is fully ingested into the repository's source-validation register. The
planner has no remaining work orders, and every ledger span has a promoted
generated surface or a legacy register surface.

This completion means the repository can now:

- Load generated Paper 0 spec bundles through `scpn_quantum_control.paper0.spec_loader`.
- Validate fixture summaries against recorded source spans and component labels.
- Reconcile promoted Paper 0 coverage without remaining ledger gaps.
- Run focused generated tests for the promoted register surfaces.
- Reference the Paper 0 register from public documentation without exposing
  internal extraction logs.

## What Completion Does Not Mean

The register is not a hardware-evidence ledger and not an external scientific
validation result.

It does not claim that Paper 0 propositions are experimentally confirmed. It
only states that the repository has source-bounded ingestion, fixture
preservation, generated tests, and explicit claim-boundary metadata for the
canonical ledger.

Hardware, simulator, measured-system, and external-validation claims remain
governed by the [Hardware Status Ledger](hardware_status_ledger.md), the
benchmark dashboards, and the specific artefact manifests named by those pages.

## Code Surfaces

| Surface | Purpose |
| --- | --- |
| `scpn_quantum_control.paper0` | Public package namespace for generated Paper 0 validation helpers. |
| `scpn_quantum_control.paper0.spec_loader` | Loads JSON spec bundles from repository artefacts using repository-relative paths. |
| `scpn_quantum_control.paper0.*_validation` | Generated validation modules for source spans and component groups. |
| `scripts/build_paper0_*_specs.py` | Rebuilds validation specs for individual promoted spans. |
| `scripts/run_paper0_*_fixture.py` | Emits fixture reports for individual promoted spans. |
| `tests/test_build_paper0_*_specs.py` | Guards generated spec shape and source-span accounting. |
| `tests/test_paper0_*_validation.py` | Guards validation helper behaviour for promoted spans. |
| `tests/test_run_paper0_*_fixture.py` | Guards fixture runner output and metadata preservation. |

## API Pattern

```python
from scpn_quantum_control.paper0 import validate_upde_fixture
from scpn_quantum_control.paper0.spec_loader import load_upde_validation_spec

spec = load_upde_validation_spec()
result = validate_upde_fixture()

assert spec["claim_boundary"].endswith("not validation evidence")
assert result.coverage_match is True
```

The exact helper names follow the source-span slug used by the generated module.
Prefer importing only the specific validation helper or spec loader required by
the workflow instead of importing every generated register symbol.

## Documentation Standard for Future Paper Registers

Future paper-ingestion registers should preserve the same contract:

- Every generated file carries the standard repository header.
- Every source span records `source_start`, `source_end`, and source record count.
- Every spec and fixture preserves the claim boundary.
- Public docs describe ingestion as source-accounting unless measured evidence is
  linked through a separate evidence ledger.
- Internal extraction artefacts stay under `docs/internal/`.
- Public API pages route readers to a concise register page instead of exposing
  thousands of generated internals as the first-path documentation surface.
