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
| Readout mitigation | `build_readout_mitigation_bundle()` | `studio.mitigation.v1` | Measured raw-versus-corrected parity-asymmetry pairs from real `ibm_kingston` runs ride in `cases[]` verbatim (corrected relative asymmetry as case error, `hardware-mitigated` substrate). The claim boundary carries the artefact's confusion-matrix caveat verbatim: state-specific parity inversion over selected calibration states, not a full `2^n x 2^n` confusion-matrix inversion. `bounded-support`. |
| QEC offline readiness | `build_qec_readiness_bundle()` | `studio.qec-readiness.v1` | Offline distance-3 surface-code decoder logical-failure aggregates ride in `cases[]` verbatim with explicit `simulated` statuses (Monte-Carlo decoder runs under modelled noise, never hardware). The claim boundary carries the artefact's own supported/blocked lists verbatim — fault tolerance, scalable QEC, and hardware logical-error reduction stay blocked. `bounded-model`. |

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

The readout-mitigation and QEC-readiness builders load their committed
artefacts and fail closed on any missing honesty field (method, pairs,
confusion-matrix caveat; decoder aggregates, code distance, readiness
decision, supported/blocked lists):

```text
data/phase2_readout_mitigation/phase2_readout_mitigation_summary_2026-05-05.json
data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json
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
schema-A capability manifest, the transform-algebra support matrix and explorer,
the gradient-plan explanation artefact, and the baseline scorecard — each at its
own claim boundary. Statuses are never recomputed or upgraded in the UI; a
surface that fails its fail-closed guard renders as a loud `unverifiable` block.
The gradient-plan view is sourced from
`data/differentiable_phase_qnode/gradient_plan_explanations_20260709.json`; it
explains the planner-selected method and fail-closed boundaries, but it does not
execute a browser differentiate run. The manifest `ui_module` is LIVE: the
remote is pull-deployed under the Hub origin and the manifest points at
`https://www.anulum.org/studios/scpn-quantum-control/remoteEntry.js`, exposing
`./QuantumStudioPanel` (probe-verified by the platform keeper; a regression
test pins the values against `module-federation.config.ts`).

### 3D Lab

The 3D Lab section renders two orbitable scenes from the live WASM Kuramoto
kernel — the same Rust kernel the Play panel drives. The kernel's one-shot
entry point returns only `[R(t) ; θ_final]`, so the Lab captures the full
phase trajectory `θ_i(t)` by chaining single RK4 steps (the kernel accumulates
raw phases and its canonical input round-trips exactly, so the chain is
bit-identical to the one-shot run) and then **proves** that at runtime: the
captured order-parameter series and final phases are compared bit-exactly
against a one-shot replay, and any divergence renders as a loud
`unverifiable` block. The TypeScript centroid recomputation is additionally
held within `1e-12` of the kernel's own R(t).

* **Phase cylinder** — each oscillator traces `(cos θ_i(t), sin θ_i(t), t)`
  on the unit cylinder with time running up the axis; the heavy strand is the
  order-parameter centroid, whose radius is R(t).
* **Bloch equator** — the final snapshot under the documented classical-limit
  correspondence (a phase is a qubit on the Bloch sphere's equatorial plane):
  spin-coherent points on the equator plus the order parameter `R e^{iψ}` as
  an interior equatorial point. The scene claims no z-axis dynamics, no
  entanglement, and no hardware state.

The Lab enforces its own fail-closed boundary (N ≤ 32, steps ≤ 360 — stricter
than the kernel's limits) and projects with exact, unit-tested orthographic
mathematics into SVG rather than a WebGL engine, keeping the remote inside the
portal's first-paint budget and every rendered element accessible.

### Pull-deploy release contract

Studio repositories are public, so this repo holds **zero deploy
credentials** — no SSH key, no box secret. Deployment is a credential-free
PULL, owner-decided on 2026-07-10: pushing a `studio-remote-v<version>` tag
rebuilds the remote from source and publishes three public GitHub Release
assets (the tag's semver IS the studio version — the remote iterates
independently of the Python package version):

* `studio-deploy.json` — the fleet-normative `studio.deploy-bundle.v1`
  discovery descriptor (hosting contract §3; identical schema across every
  federated studio): studio id, `studio_version`, `bundle_asset`, and the
  tarball's bare lowercase-hex `bundle_sha256`. The reflector scans recent
  non-draft releases for exactly this asset name.
* `scpn-quantum-control-studio-remote.tar.gz` — the deployable bundle: the
  full `vite build` tree plus the shipped WASM kernels AND the committed
  schema-A `manifest.json` staged at the archive root (the platform's stage
  gate requires it to name this studio; the box aggregation composes
  `federation.json` from it). Packed deterministically (sorted members,
  zeroed owners and timestamps) by `tools/package_studio_release.py`.
* `deploy-manifest.json` — this repo's richer release manifest
  (`scpn_qc_studio_release_manifest_v1`): the `sha256:`-prefixed bundle
  digest and byte size, one digest row per bundled file, and the kernel
  toolchain provenance carried verbatim from the in-bundle deploy manifest.

The SCPN-STUDIO reflector reads the descriptor first, compares the digest
against the deployed state, then pulls the tarball and re-verifies it
fail-closed (sha-256 match, `data`-filtered extraction, root `manifest.json`
naming this studio) before deploying into `/studios/scpn-quantum-control/`.
The packer itself fails closed on a missing or stale bundle: every artefact
row of the in-bundle deploy manifest is re-hashed before anything is
packaged.

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
