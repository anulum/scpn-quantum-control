<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Studio federation -->

# Studio Federation

The QUANTUM Studio federation surface has two layers:

- schema-A capability publication through `scpn-emit-studio-manifest`;
- schema-B `EvidenceBundle` emission for committed differentiable claim-ledger rows
  and hardware result-pack manifests.

Both layers are conservative. A bundle can make an artefact replayable and
federation-checkable, but it cannot promote a claim beyond the underlying ledger,
hardware pack, or external validation certificate.

## Schema-A manifest

Generate the committed Studio federation document from a source checkout:

```bash
scpn-emit-studio-manifest
```

Check the committed JSON for drift without writing it:

```bash
scpn-emit-studio-manifest --check
```

The generated file is:

```text
docs/_generated/studio_manifest.json
```

It contains the platform capability manifest plus the additive
`architecture-map.v2` extension used by architecture docs and sibling
integrators. The schema-A digest covers verbs, evidence schemas, and the
substrate-axis declarations below.

## Schema-B bundle API

Use the library API when the Hub or a local audit needs concrete evidence
objects rather than a capability declaration:

```python
from pathlib import Path

from scpn_quantum_control.studio import (
    build_claim_ledger_bundles,
    build_hardware_result_pack_bundles,
    validate_bundles,
)

claim_bundles = build_claim_ledger_bundles()
hardware_bundles = build_hardware_result_pack_bundles(
    manifest_path=Path("data") / "hardware_result_packs" / "manifest.json"
)

claim_verdicts = validate_bundles(claim_bundles)
hardware_verdicts = validate_bundles(hardware_bundles)
assert all(verdict.verdict.admitted for verdict in claim_verdicts)
assert all(verdict.verdict.admitted for verdict in hardware_verdicts)
```

The emitted dataclasses are `scpn_studio_platform.evidence.EvidenceBundle`
instances. Use `validate_bundle(...)` or `validate_bundles(...)` before sending
wire dictionaries to a Hub.

## Bundle families

| Family | Builder | Schema | Boundary |
|---|---|---|---|
| Differentiable claim ledger | `build_claim_ledger_bundles()` | `studio.evidence-replay.v1` | Current committed rows are curated `bounded-model` numerical-model evidence. `reference-validated` is only emitted when an external certificate marks a promoted row as validated. |
| Hardware result packs | `build_hardware_result_pack_bundles(...)` | `studio.hardware-result-pack.v1` | Packs are measured `bounded-support` hardware-unmitigated evidence with SHA-256 derivation edges for each committed artefact. |

The ledger builder loads:

```text
data/differentiable_phase_qnode/claim_ledger.json
```

The hardware builder loads:

```text
data/hardware_result_packs/manifest.json
```

## Substrate axes

The Studio manifest publishes the schema-B substrate axes for verbs that carry
substrate-bearing evidence:

| Verb | Declared substrates |
|---|---|
| `analyse` | `classical-reference`, `numerical-model`, `simulator` |
| `execute` | `hardware-unmitigated`, `hardware-mitigated` |

The local emitter maps source classes through `evidence_axes(...)`:

| Source class | Evidence kind | Substrate |
|---|---|---|
| `theory` | `curated` | `classical-reference` |
| `simulator` | `measured` | `simulator` |
| `hardware-unmitigated` | `measured` | `hardware-unmitigated` |
| `hardware-mitigated` | `hardware-validated` | `hardware-mitigated` |
| `falsification` | `falsified` | `numerical-model` |
| `noise-floor` | `noise-limited` | `hardware-unmitigated` |

## Claim boundaries

- Current differentiable claim-ledger rows remain bounded model evidence until
  isolated benchmark artefacts and external comparison rows are attached.
- Hardware result packs preserve committed raw-count artefacts and reproduction
  commands; they do not submit QPU jobs and do not create broader hardware claims.
- Blocked or dependency-gated rows retain explicit upstream blockers in the
  bundle boundary. The platform can admit those bundles in boundary mode, but
  the bundle admission remains rejected.
- The Studio UI module remains absent from the manifest until a real UI binding
  exists.
