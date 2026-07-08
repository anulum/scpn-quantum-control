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

The manifest advertises six core-spine verbs (`compile`, `simulate`,
`analyse`, `validate`, `benchmark`, `replay`) and three domain-distinctive
verbs (`differentiate`, `mitigate`, `execute`). `differentiate` is the
contract-reserved QUANTUM verb: gradient evaluation over compiled phase
programmes, fail-closed outside the declared support matrix, producing
`studio.differentiation-evidence.v1` claims.

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
| Differentiable baseline scorecard | `build_scorecard_bundle()` | `studio.differentiation-evidence.v1` | Eleven external-baseline category rows ride in `cases[]` with verbatim statuses; the bundle is curated `bounded-model` and the emitter can never upgrade a category. A scorecard that fails its own validation is refused. |
| Transform-algebra support matrix | `build_support_matrix_bundle()` | `studio.differentiation-evidence.v1` | Thirteen generated support rows ride in `cases[]` with verbatim statuses and measured residuals as case errors; blocked rows stay explicit fail-closed boundaries. The bundle is measured `bounded-model` numerical-model evidence, and an audit that did not pass is never federated. |
| Effective-coupling invariant | `build_coupling_invariant_bundle()` | `studio.coupling-invariant.v1` | `knm.kuramoto.effective-coupling` source inventory from Hamiltonian learning and differentiable coupling learning with parameter-shift verification. `sync_uncertainty` and `zne_uncertainty` are mandatory UQ sources; the bundle is not a DLA parity claim. |

The ledger builder loads:

```text
data/differentiable_phase_qnode/claim_ledger.json
```

The hardware builder loads:

```text
data/hardware_result_packs/manifest.json
```

The support-matrix builder reruns the transform-algebra audit in-process and
content-addresses the committed artefact for its derivation edge:

```text
data/differentiable_phase_qnode/differentiable_transform_support_matrix_20260708.json
```

## Studio web remote (Phase 0)

The `studio-web/` workspace builds the QUANTUM studio's federated UI from one
Vite app: a static portal (`dist/index.html`) and a Module Federation remote
(`dist/remoteEntry.js`, ESM). The federation contract is locked in
`studio-web/module-federation.config.ts` and guarded by tests: federation name
`scpn_quantum_control`, one exposed module `./QuantumStudioPanel`, and
react/react-dom shared as version-pinned singletons.

```bash
cd studio-web
pnpm install
pnpm test:coverage
pnpm build
```

The Phase-0 panel renders the committed evidence surfaces verbatim — the
schema-A capability manifest, the transform-algebra support matrix, and the
baseline scorecard — each at its own claim boundary. Statuses are never
recomputed or upgraded in the UI; a surface that fails its fail-closed guard
renders as a loud `unverifiable` block. The manifest `ui_module` field stays
`null` until the remote is deployed and its URL is real.

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
still reports `0.0` answer rate across its 16 candidate rows (emitted as
curated `bounded-model` bundles). A future
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
the structural XY compile terms from the canonical little-endian byte payload,
and the `scpn_alloc`/`scpn_free` exports let a host hand the payload into guest
memory. The Python wrapper mirrors the same byte encoding for local tests and
signed unit construction; browser verification uses the Rust/WASM kernel. This
covers the bit-exact compile decision path only. Continuous simulator values
and QPU counts remain separate tolerance or attestation evidence.

### Committed browser-verifiable unit (WS-1)

One committed unit lets the studio panel replay the compile digest in the
visitor's browser:

```text
data/studio/xy_compile_recompute_unit_20260708.json
```

It is emitted from the provisional Paper-27 coupling matrix by
`scpn_quantum_control.studio.xy_compile_recompute_artifact` (writer/check CLI,
fail-closed: a unit that fails its own reference verification is never
serialised). The studio-web recompute card loads the CI-built WASM kernel,
recomputes the digest, and compares it to the signed claim. Every tamper is
loud and fail-closed: a forged digest renders `mismatch`, and a stripped
grade, wrong schema, malformed input, or a kernel-level rejection render
`unverifiable`. A tampered unit can never render `match`. The unit's claim
boundary is the bit-exact compile decision path only — never a physical
`K_nm` claim, QPU execution, or actuation authority.

## WS-1 attestation-verifiable QPU results

WS-1 grants two verification modes, and every evidence unit declares which one
it stands on. Compile-path claims are **recompute**-verifiable (above). A QPU
result cannot be replayed — the shot statistics are irreproducible — so it is
**attestation**-verifiable: the trust rests on a hardware provider's own signed
record. The `execute` verb therefore produces a second claim family,
`studio.qpu-result-pack.v1`, alongside `studio.hardware-result-pack.v1`:

```python
from scpn_quantum_control.studio import (
    build_qpu_result_pack_unit,
    present_qpu_result_pack,
)

unit = build_qpu_result_pack_unit(
    pack,
    raw_results_digest="sha256:...",
    circuit_digest="sha256:...",        # bit-exact link to a recompute-verifiable compile
    calibration_ref="calibration/...",  # device calibration snapshot
    attestation=None,                    # a live provider attestation, when one exists
)
present_qpu_result_pack(unit).status     # "unverifiable" without a provider attestation
```

Every unit carries `verifiability_mode = attestation`. The absent-signal is
loud: with no provider attestation `present_qpu_result_pack` renders
`unverifiable` and `seal_qpu_result_pack` refuses to seal — a QPU unit is never
emitted as `verified` on the studio signature alone. The committed hardware
packs carry no live provider attestation yet (BL-29 territory), so their units
are honestly `unverifiable` today; the shape and the fail-closed boundary are
what ships. A supplied attestation must sign the exact `raw_results_digest`, or
it is rejected.

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
