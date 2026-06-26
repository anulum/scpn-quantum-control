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
    load_reference_validation_registry,
    measure_coverage_frontier_from_certifications,
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

registry = load_reference_validation_registry()
frontier = measure_coverage_frontier_from_certifications(registry=registry)
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

## WS-3 reference-validation feed

WS-6 coverage must advance from attached reference-validation evidence, not from
claim relabelling. QUANTUM therefore keeps a separate WS-3 registry:

```text
data/differentiable_phase_qnode/reference_validation_certifications.json
```

The registry uses schema `studio.reference-validation-certifications.v1` and is
loaded with:

```python
from scpn_quantum_control.studio import (
    load_reference_validation_registry,
    measure_coverage_frontier_from_certifications,
)

registry = load_reference_validation_registry()
report = measure_coverage_frontier_from_certifications(registry=registry)
```

The committed registry is currently empty. That is intentional: no per-claim
WS-3 certifications are committed yet, so the real differentiable claim ledger
still reports `0.0` answer rate and 13 `bounded-model` rows. A future
certification row must name a known ledger claim whose `promotion_status` is
already `promoted`; candidate, unknown, or duplicate certification rows fail
closed before they can reach the WS-6 measurement.

## SPO `knm.scpn-upde` edge

The SPO federation edge is available from the bridge facade:

```python
from scpn_quantum_control.bridge import (
    build_paper27_scpn_upde_edge,
    validate_scpn_upde_edge_payload,
)

payload = build_paper27_scpn_upde_edge(
    time=0.1,
    trotter_steps=1,
    trotter_order=1,
).to_payload()
validate_scpn_upde_edge_payload(payload)
```

The payload schema is `knm.scpn-upde.v1`. It carries the 16-oscillator Paper-27
`K_nm` matrix, `omega` vector, Trotter compile metadata, SHA-256 integrity
digests, and explicit permissions:

```json
{
  "scope_envelope": "computational-agreement",
  "permissions": {
    "qpu_execution_permitted": false,
    "actuation_permitted": false
  }
}
```

That boundary is load-bearing. The Paper-27 matrix is provisional, so this edge
is only for QUANTUM/SPO computational agreement and reviewable compiler handoff.
It is not physical validation, not a canonical `K_nm` claim, and not live
actuation authority.

## WS-1 recompute kernel

Compile-path claims can now be emitted as recompute-verifiable units:

```python
from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.studio import (
    build_xy_compile_recompute_unit,
    verify_xy_compile_recompute_unit,
)

unit = build_xy_compile_recompute_unit(
    build_knm_paper27(L=16),
    OMEGA_N_16,
    time=0.1,
    trotter_steps=1,
    trotter_order=1,
)
assert verify_xy_compile_recompute_unit(unit).value == "match"
```

The unit schema is `studio.xy-compile-recompute.v1`. It declares
`verifiability_mode = recompute`, `exactness_class = bit-exact`, and the WASM
kernel source:

```text
scpn_quantum_engine/studio_wasm_kernel
```

That Rust crate is dependency-light and builds to `wasm32-unknown-unknown`. Its
exported `scpn_xy_compile_digest` function recomputes the SHA-256 digest over
the structural XY compile terms from the canonical little-endian byte payload.
The Python wrapper mirrors the same byte encoding for local tests and signed
unit construction; browser verification uses the Rust/WASM kernel. This covers
the bit-exact compile decision path only. Continuous simulator values and QPU
counts remain separate tolerance or attestation evidence.

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
- WS-3 reference-validation certifications are the only committed feed that can
  move a promoted ledger row to `reference-validated` in the WS-6 frontier.
- Hardware result packs preserve committed raw-count artefacts and reproduction
  commands; they do not submit QPU jobs and do not create broader hardware claims.
- Blocked or dependency-gated rows retain explicit upstream blockers in the
  bundle boundary. The platform can admit those bundles in boundary mode, but
  the bundle admission remains rejected.
- The Studio UI module remains absent from the manifest until a real UI binding
  exists.
