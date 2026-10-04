<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Studio workbench -->

# Quantum Studio workbench

The existing `QuantumStudioPanel` displays the capability manifest and committed
evidence, with bounded local instruments for XY recomputation, Kuramoto playback,
3D trajectory inspection and program-AD replay. Unsupported or unverifiable
evidence remains visible as such. The panel source is in `studio-web/`; the
federation manifest exposes it as `./QuantumStudioPanel`.

The manifest lists nine verbs: `compile`, `simulate`, `analyse`, `validate`,
`benchmark`, `replay`, `differentiate`, `mitigate` and `execute`. The current panel
shows these in a searchable catalogue and as verbatim capability declarations. It does not provide a general browser
workflow to dispatch them. For local use, the public `scpn-studio-run` CLI
dispatches each verb through its owning handler. Run `scpn-studio-run --help`
for parameters and `docs/studio_federation.md` for evidence and federation
contracts. Optional backend availability and scientific claim boundaries remain
specific to each handler and evidence artefact.

The `execute` handler creates a no-submit dossier and an operator script. It
does not contact a QPU or return hardware counts. Running an operator script is
a separate approved action. A recorded July 2026 keeper check verified a live
Hub panel render; this document makes no assertion about current hosted
availability without a fresh deployment probe.

## Workbench navigation

The Workspace view retains the original capability catalogue, committed cards
and local archive editor. Build opens the existing compile recomputation,
Kuramoto Play and 3D Lab instruments, together with the program editor and compiler
trace inspector. Results opens program-AD replay, evidence
inspection, the support explorer, gradient explanations and scorecard.
Experiments opens the source-bound local experiment workflow described below.
Atlas shows its unavailable workflow and a link back to Workspace. The mode
banner distinguishes standalone and embedded layouts;
federation hosts can pass `mode="embedded"` to the same exposed panel.
Devices & Operations is a separate context destination for dated provider and
device profiles. It retains the requested project, revision and snapshot context.

Hash addresses work on static hosting without server rewrite rules:

| View | Address |
| --- | --- |
| Workspace | `#/workspace` (also the empty address) |
| Build | `#/build` or `#/build/compile-recompute` |
| Experiments | `#/experiments` |
| Results | `#/results` or `#/results/program-ad-replay` |
| Atlas | `#/atlas` |
| Devices & Operations | `#/operations` |

Optional `project`, `revision` and `snapshot` query fields retain exact opaque
identity through links, reload and browser history. For example,
`#/build/compile-recompute?project=my-project&revision=revision-1` opens the
committed instrument while displaying that requested context. Opening an
address does not load or validate those identifiers, change the selected saved
archive, or attach a committed instrument result to that revision. The common
inspector displays requested identity separately from the original storage
owner's admitted saved project, workspace, archive and document digests.

Addresses are bounded to 4096 UTF-16 code units in both input and stable encoded
form; each supplied identifier has a 256-code-unit bound. A character outside
the Basic Multilingual Plane occupies two code units. Empty identifiers, repeated
or unknown fields, malformed UTF-8/percent encoding, control characters and
unsupported paths render an explicit route refusal. The current draft and
saved archive remain unchanged. Return to Workspace to continue editing.

The original editor remains mounted while navigating. A feature import or
render error has its own boundary and offers a Workspace recovery link without
clearing the editor. Unsaved edits survive navigation within the mounted panel;
reload restores the last saved archive. Export remains the independent backup.
View links and breadcrumbs are keyboard accessible, and navigating focuses the
view or its instrument target. Navigation wraps to fit narrow host layouts.

Build, Experiments, Results and unavailable-view modules load on navigation. The compatibility
Workspace overview still includes its original instrument dependencies at
startup; lazy route modules do not imply that all numerical code has been
removed from the initial bundle. Existing federation name, expose key and
version-pinned shared React contracts are preserved.

### Keyboard, tables and display preferences

Use **Skip to current view** to focus the active view directly. **Keyboard help**
opens a modal with the native keyboard actions. Tab and Shift+Tab stay on its
Close control; Escape or Close returns focus to the button that opened it.
Navigation closes the helper and focuses the selected view.

The layout supports light and dark system preferences, 200% zoom and reduced
motion. Actions wrap within the view. Wide data regions have a keyboard focus
target so their contents can be scrolled horizontally. Claim and verification
statuses remain written as text in either theme.

Kuramoto Play and 3D Lab provide **Order parameter data** and **Phase trajectory
data** tables. They show the original sample index, dimensionless order parameter
and each oscillator's phase in radians. Every sample remains available through
Previous samples and Next samples; at most 20 rows are mounted at once. Values
come from the same captured run used by the plots, with their original numeric
precision and oscillator order.

The linked coupling editor provides a **K_nm coupling edges** table alongside
its diagram. Its rows retain directed `j → i` identity, signed coefficient and
original unit. Choosing a row selects the same coefficient in the matrix and
value form. Zero coefficients remain selectable in the matrix.

The accessibility journey audits every delivered view in both themes, including
loading, empty filters, compiler refusals, malformed routes/evidence, offline
kernel transport, altered source digests and delayed old verification. It uses
the locked axe-core source with its default rules, retains complete reports,
and refuses serious or critical findings. A recorded walkthrough with an actual
screen reader supplies the separate assistive-technology evidence.

Run it against the actual built bundle and a separately owned source host:

```bash
PYTHONPATH=src:oscillatools/src:. python tools/studio_browser_journey.py \
  --scenario workbench_accessibility --base-url http://127.0.0.1:4173/ \
  --workspace-source-url http://127.0.0.1:4174/ \
  --output studio-accessibility-journey.json
```

Both addresses must be literal HTTP loopback origins with distinct ports; the
source host must serve the repository's Studio source at its root. The normal
locked Studio dependencies provide the auditor. `--axe-source` can identify an
already installed copy with the exact same content digest. The command performs
no installation, provider submission, workspace export or screenshot capture.

### Verify local navigation

The `workbench_navigation` scenario exercises the original standalone panel,
an independently built federation consumer, a refused route chunk and the
original source panel. It checks saved archive and draft retention, deep links,
reload and browser history, malformed addresses, narrow-layout keyboard focus
and actual compile/gradient WASM recomputation. The federation consumer uses
the real remote `get`/`init` exports and the locked shared React providers.

Prepare the production bundle with `pnpm build` in `studio-web/` and
`python tools/build_studio_wasm_bundle.py` from the repository root. Build the
independent consumer with `pnpm exec vite build --config
browser-tests/federation.vite.ts --outDir <owned-host-directory>` in
`studio-web/`. Serve a separate preview containing the original built bundle
at its root and that consumer under `acceptance-host/`; keep the shipped
production bundle separate from these acceptance fixtures.

The second server is a root Vite server over an exact copy of the original
source, browser fixtures and configuration, with the original data and installed
dependencies available. Copy the two actual built files from `dist/wasm/`
inside that copy's `src/wasm/` so the source-relative imports resolve within
Vite's serving boundary. Studio CI prepares and verifies these copies in its
temporary directory, preserving the canonical source and kernel hashes.

With both owned loopback servers running, use a new output path:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py \
  --scenario workbench_navigation --base-url http://127.0.0.1:4173/ \
  --workspace-source-url http://127.0.0.1:4174/ \
  --output /tmp/workbench-journey.json
```

The runner closes its browser contexts; the server owner stops both servers.
The result includes native V8 records for all fourteen workbench, catalogue,
facade and original storage/controller owners. Set
`STUDIO_WORKBENCH_COVERAGE` to this successful JSON alongside the required
`STUDIO_WORKSPACE_COVERAGE` and original panel-refusal evidence when running
the affected owner coverage cohort. The source qualifier verifies actual code,
source maps and current owner hashes before merging counters through the
existing Vitest provider. Stale, failed or incomplete evidence refuses.

## Local experiment and portable replay

Open **Experiments**, then **Open Kuramoto sample**. The committed classical
mean-field sample becomes an unsaved workspace draft; opening it starts no
worker. Select **Validate experiment archive**, then **Edit source parameters
in Workspace**. The existing linked parameter editor validates typed values
against their immutable specifications. **Save parameter revision** appends a
child revision and retains the original input bytes and parent identity.

Return to Experiments and choose **Prepare numerical plan**. Inspect its exact
revision, plan fingerprint, shipped kernel digest, method, float64 precision,
source shape, actual native limits, declared byte/work estimates and policy
origin. The original source records matching requested/effective settings and
the browser environment observed when it was created. These imported
observations remain historical metadata. The deterministic source has no
numerical seed. Phases use radians; `dt` uses unscaled model-time, and frequency
and coupling use radians per model-time. No conversion to physical seconds is
implied.

**Run numeric byte budget** optionally supplies a canonical nonnegative decimal
ceiling below the recorded source policy. Empty retains that policy; zero
refuses before worker allocation. Changing the field disables execution until
another plan is explicitly prepared. The declared plan includes the original
numeric buffers, source validation, transfer copies and both retained and
transferred kernel bytes. Browser capacity and computation duration are not
measured by this admission. The visible operational timeout controls disposal,
without guaranteeing that the browser scheduler will meet it.

**Run experiment** explicitly starts the original disposable Rust/WASM worker.
The worker validates source, revision, plan and build identity before admission.
Success appears only after its genuine result and observed disposal. The table
contains the original order-parameter samples, and the final-phase list retains
oscillator order and radians. This ABI returns final phases and an order-
parameter trajectory; it does not supply a complete phase history. The result
is a classical Kuramoto calculation, with no quantum-spin, physical-device or
provider execution claim.

**Cancel experiment**, navigation away from Experiments, a selected source edit
or unmount disposes the owned worker. Its original diagnostic events remain
bound to the captured revision and plan. A failed, cancelled or stale attempt
cannot acquire a success badge. A changed immutable revision cannot inherit an
old result or save it as its own. Unconfirmed disposal blocks further runs and
attempt saves in that mounted workbench.

**Save experiment attempt** uses the existing browser archive transaction and
requires an unchanged selected source plus disposed diagnostics. It appends a
`local_run_record.v1`, the exact effective plan/input/policy, and original output
bits for a successful run. The five original workspace document formats remain
unchanged. Imported data cannot install verifiers or execute a recorded kernel;
replay additionally requires its digest and native bounds to match the actually
shipped module. Unsupported or altered producer content refuses admission.
Unexpected operation faults show a fixed message while retaining the draft;
deliberately authored refusals identify the rejected local operation.

Use **Export saved experiment** for the committed browser archive, or **Export
experiment attempt** for a disposed current-source attempt without changing
browser storage. Persistence-unavailable browsers retain portable export and
explicitly disable saved-state claims. Keep an independent exported backup.

In a clean browser context, open Workspace, select the exported local JSON file,
preview the complete archive and explicitly save it. In Experiments, **Prepare
saved replay** re-admits the original completed record and recomputes its exact
numerical plan. It starts no worker and makes no fresh success claim. **Run
experiment** creates a new run and attempt identity, then compares every actual
output binary64 bit with the saved original. An exact match is labelled
**Original float64 replay verified**; a difference remains a failed attempt with
the original numerical diagnostics retained.

The original shared browser dispatcher exercises sample/edit/plan/run/save,
zero-budget refusal, cancellation, a genuine new source revision, worker entry
failure, altered archive bytes, missing WASM and fresh-context import/replay:

```bash
PYTHONPATH=src:oscillatools/src:. python tools/studio_browser_journey.py \
  --scenario local_experiment_journey --base-url http://127.0.0.1:4173/ \
  --workspace-source-url http://127.0.0.1:4174/ \
  --output studio-local-experiment-journey.json
```

The built preview and optional source host follow the preparation described
above. The source host must be a distinct root HTTP loopback origin. Its actual
V8 record requires all ten original experiment, Workbench and workspace owners.
Set `STUDIO_EXPERIMENT_COVERAGE` to successful current-source evidence alongside
the required `STUDIO_WORKSPACE_COVERAGE`. The existing converter/provider
preserves native zero counters and checks exact source maps and hashes before
merging. Raw observations are not a measured coverage percentage.

The separate Python `simulate` handler evolves the original undecomposed Qiskit
operator through the supported local Python/Qiskit solver. Trotter parameters
do not turn that execution into a decomposed circuit calculation. A declared
Rust plan remains inspectable, but full quantum-trajectory execution on that
backend is unavailable and refuses explicitly; it cannot silently execute
through Python while claiming Rust. This handler and the browser's classical
phase worker have distinct model and output contracts.

## Provider and device profiles

Open Devices & Operations, then Open declared profiles to inspect the committed
offline catalogue. Select an exact route explicitly. The view shows provider,
broker, physical device, HAL backend, modality, region, SDK and IR declarations,
capture date and age, calibration timestamp/reference and opaque credential
configuration references. Pulse and analog options retain the native HAL reason
when disabled; an enabled option displays its declaration without executing it.

HAL declarations and route-operation declarations are displayed separately from
supplied observations. Unknown availability, limits and support remain unknown;
an observed zero queue depth and an explicit unsupported value retain their
meaning. Dates and supplied observations do not certify current availability,
calibration freshness or authentication. Open declared profiles performs no
provider refresh and reads no credential values.

The original dated profile format cannot represent a configured native HAL
`max_shots` declaration. The producer refuses such a declaration rather than
dropping its limit; ordinary built-in profiles with no declared shot ceiling
retain their exact existing export. Operator admission uses the native HAL
capacity checks independently of the dated profile export.

Inspect profiles admits `studio.backend-profiles.v1` JSON with exact row and
envelope SHA-256 identities. Counts retain unsigned 64-bit precision. The input
ceiling is 1 MiB of UTF-8 and 256 distinct routes; these are contract limits,
not measured latency guarantees. Extra fields, malformed references, invalid
dates or altered digests refuse the whole import and preserve the prior admitted
selection. Export admitted profiles downloads the exact admitted input text.

Imported plan, calibration and review digests are metadata references. They grant
no run approval. Selecting another route or importing changed dated metadata
clears all dependent references, including when two brokers name the same physical
device. References survive only for the exact same row digest. Selection belongs
to the mounted profile view; export before leaving it or reloading. Workspace
archives and unsaved workspace drafts remain independent.

The native adapter reuses the existing route catalogue and HAL profiles. Its
typed canonical bytes and schema-prefixed SHA-256 digests come from the shared
`scpn_quantum_control.canonical_encoding` contracts owner. Workspace readers use
the same implementation through their existing `studio_workspace.canonical`
imports; HAL export loads no Studio module. Profile wire fields and digests are
unchanged by this shared ownership.

```python
from scpn_quantum_control.hardware.provider_capability_discovery import build_backend_profiles

profiles = build_backend_profiles(observed_at="2026-10-03")
assert profiles["body"]["no_submit"] is True
```

Supply an explicit capture date when producing a new offline catalogue:

```bash
PYTHONPATH=src:oscillatools/src python tools/export_backend_profiles.py \
  --observed-at 2026-10-03 --output /tmp/backend-profiles.json
```

Against an owned built preview with its actual WASM, `operator_backend_profiles`
checks exact integer transport, offline/unknown state, malformed recovery,
route-bound reference invalidation, exact downloads, original workspace draft
custody and genuine compile recomputation. It refuses actual page errors,
requests outside that preview and submission attempts:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py \
  --scenario operator_backend_profiles --base-url http://127.0.0.1:4173/ \
  --output /tmp/backend-profiles-journey.json
```

The runner closes its browser context; the preview owner stops its server.

## Supported program authoring

Open Build to edit a program source or append an operation through the form.
Compile source invokes the shipped Rust/WASM compiler. The result displays
ordered IR, parameter bits, source positions, classical conditions and readout
pairs. **Emitted — not executed** means the source was admitted and recorded.
This action constructs no numerical result and submits no provider job.

Changing source immediately clears the current compiled plan and disables its
export. A late result for an earlier draft is discarded. The last ten compilation
attempts retain their original revision and digest or refusal category. This
history belongs to the mounted editor; export the source before leaving Build
or reloading. Workspace archives and earlier saved results remain independent.

The supported source is a bounded OpenQASM 2.0 subset:

| Construct | Supported form |
| --- | --- |
| Preamble | `OPENQASM 2.0; include "qelib1.inc";` |
| Registers | One `qreg q[n];`, 1–8 qubits; optional `creg c[n];`, 1–64 bits |
| Single-qubit gates | `h x y z s sdg t tdg id sx sxdg` |
| Parameter gates | `rx ry rz p u1` with one parameter; `u2` with two; `u u3` with three |
| Two-qubit gates | `cx cz swap`; `rxx ryy rzz` with one parameter |
| Effects | `measure q[i] -> c[j];`, `reset q[i];`, `barrier q;` or indexed barrier operands |
| Classical control | `if(c==unsigned_integer)` followed by one supported gate |

Parameters are finite decimal binary64 values in radians. Expressions such as
`pi/4`, custom gate definitions, other register names, arbitrary includes and
Python are refused. Indices and conditions use canonical unsigned decimal text.
Conditions retain all 64 bits without conversion to JavaScript numbers. Source
is limited to 1 MiB of UTF-8, 65,536 tokens and 4,096 operations. Comments remain
inert and are retained in the exact original source. An unsupported token or
missing token has an original half-open Unicode scalar span and one-based
line/column. Select offending source transfers that location to the text field.

Export exact source downloads `program.qasm` with the compiled source unchanged,
including its comments and formatting. The `studio.program-source.v1` record
stores SHA-256 of those UTF-8 bytes and ordered IEEE float64 parameter hex values,
including signed zero. Its `execution_status` is `emitted_not_executed`.

Native Python imports use the actual Qiskit compiler through the same admitted
subset. The `ryy` extension binds explicitly to Qiskit's `RYYGate`. Exact native
export uses roundtrippable decimal parameters; it does not approximate them with
pi aliases. An independent nonzero global phase is refused because this source
format cannot encode it. Conditional blocks must contain a single supported gate
with no else branch or independent block phase.
Native operations must use the supported gate's standard native classes;
custom or modified gates cannot export under a built-in name.

```python
from scpn_quantum_control.studio.program_authoring import (
    compile_program_source, export_program_source, import_program_source,
)

source = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(-0.7853981633974492) q[0];
measure q[0] -> c[1];
if(c==2) x q[1];
'''
record = compile_program_source(source)
circuit = import_program_source(record.source)
restored = compile_program_source(export_program_source(circuit))
assert restored.measurements == record.measurements
```

The existing `compile` executive handler accepts `{"program_source": source}`
as its source mode, using the declared Python backend. Network parameters cannot
be mixed into this request. Its generated reproduction script emits the original
source record and verifies its digest; running gates remains a separate action.
The original XY network mode and static-unitary qualifier retain their own
contracts for backends and effectful circuits.

Against an owned built preview with the actual WASM, the shared acceptance runner
checks source cases, exact downloads, structured controls, draft invalidation and
delayed real compiler responses:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py --scenario program_authoring \
  --base-url http://127.0.0.1:4173/ --output /tmp/program-authoring-journey.json
```

Use a new output path. The runner blocks requests outside the preview and closes
its contexts; the process owner stops the preview server.

## Compiler trace inspection

Open Build to import a native compiler trace or open the explicit native example.
The inspector shows each pass's input and output source, IR digests, parameters,
logical-to-physical qubit layouts, classical output layout, readout correspondence,
ordered measurement effects and declared gate/depth/resource changes. The example
contains an actual native qualified physical swap followed by an identity pass.
It is separate from the current editor draft and saved workspace.

Select source operation pins a span in the original source. Navigating subsequent
passes leaves that selection unchanged. Each representation retains its own
source spans; cross-pass operation correspondence is unavailable. The text field
normalises line endings for display and converts caret positions accordingly;
source identities and exported bytes retain the original CRLF/LF. Missing pass
artifacts remain explicit and make the trace incomplete. Unsupported versions,
duplicate fields, altered source/IR digests and inconsistent metadata refuse
without replacing the prior admitted trace or writing the saved archive. A late
import result for an edited draft or unmounted inspector is discarded.

The browser validates metadata identity and structure. It does not rerun the
native operator qualifier. The retained native record digest uses the original
native codec; the new envelope and complete IR snapshots use the typed workspace
codec. These digests bind content and do not attest that an untrusted producer
performed its declared numerical qualification.

Create a trace through the original native compiler:

```python
from pathlib import Path
from scpn_quantum_control.studio.compiler_trace import build_compiler_trace

source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
trace = build_compiler_trace(source, optimisation_level=2)
Path("compiler-trace.json").write_text(trace.to_json(), encoding="utf-8")
```

The existing executive `compile` source request also accepts
`{"program_source": source, "compiler_trace": true, "optimisation_level": 2}`.
The level is an exact integer from zero to three and requires trace mode. The
trace is at `result.outputs.compiler_trace`; import that envelope, rather than
the surrounding CLI execution plan. The generated reproduction script exports
the bare trace and checks both original source and complete trace identities.
Changing the compiler version or settings may correctly fail that identity check.

Trace mode uses the original bounded static-unitary lowering contract. Reset,
conditional gates and nonterminal measurement are explicitly refused; their
effects are never discarded to produce a successful trace. The original source
editor continues to support its own broader effectful source subset. The actual
aggregate basis-lowering pass is recorded; unobserved internal SDK pass stages
are not invented.

The exact backend snapshot retains Qiskit version, settings, reference backend
and little-endian basis convention. Its target is no physical device. Export
admitted trace downloads the original admitted UTF-8 text unchanged, including
the full backend snapshot. **Emitted — not executed** applies to the entire
trace; an included MLIR artifact remains textual interchange output. Import,
inspection and export execute no emitted gates and submit no provider job.

Imports are bounded to 16 MiB of UTF-8, one through 32 pass slots, a source-bearing
first pass, eight qubits, 64 classical bits and 4,096 operations per representation.
Each source has a 1 MiB UTF-8 ceiling. Declared complex128 payloads are
`16 * 2**n` statevector bytes and `16 * 4**n` dense operator bytes, without
allocation; compiler scratch, allocator overhead and available host memory are
outside these declarations.

The genuine built UI/WASM acceptance journey exercises source compilation,
mapped native metadata, Unicode selection, exact trace download, missing
artifacts, refusal and recovery, while checking saved-state and network custody:

```bash
PYTHONPATH=src:oscillatools/src:. python tools/studio_browser_journey.py \
  --scenario compiler_trace_inspector --base-url http://127.0.0.1:4173/ \
  --output /tmp/compiler-trace-journey.json
```

Use a fresh output path and an owned preview with the actual shipped WASM. The
runner closes its browser; the preview server remains its owner's responsibility.

## Capability catalogue

Search by task and intersect the runtime and declared-backend filters. Each row
shows its local CLI entry, settings, evidence schemas and scope. The catalogue
identity binds the ordered projection to the manifest surface and source version.
An installed-source mismatch displays stale/unknown and disables its routes.
A backend in a manifest is a declaration, not proof that it is installed.

Two entries link to existing bounded browser instruments: **XY compile
recomputation** and **program-AD gradient replay**. A link is enabled only after
the corresponding WASM kernel loads and verifies its bounded committed input
through the existing replay owner. Each link opens a hash route and transfers keyboard focus to the actual
instrument. Kernel loading errors remain visible; reload after restoring the
bundle to retry. The instrument rechecks the result when run. Other entries
remain library-only with an explicit reason; no catalogue interaction submits
a provider job or executes arbitrary source.

The committed `docs/_generated/studio_manifest.json` remains the declaration
source. Regenerate with `python -m scpn_quantum_control.studio.federation` after
changing the projection; `--check` detects drift. The catalogue is an additive
architecture-map extension and does not replace the platform schema-A contract.

The Studio CI category runs the catalogue browser journey on the built bundle
and actual WASM, including keyboard opening, intersecting filters, a missing
kernel and recovery. Its browser-only dependency is hash-locked separately in
`requirements-ci-studio-browser.txt`. Against an owned loopback preview, use:

```bash
python tools/studio_browser_journey.py --scenario capability_catalogue \
  --base-url http://127.0.0.1:4173/ --output /tmp/catalogue-journey.json
```

The output path must be new. The runner records browser versions and observed
results, blocks requests outside the preview, and closes its browser contexts.
It connects to an existing preview; the process owner remains responsible for
stopping that preview. This verifies local instruments, not deployed availability
or scientific qualification.


## Operator policy decisions

Open **Devices & Operations**, then **Open policy example**. The inspector
displays the native core's dated decision, all requested and effective values,
winning origins, original policy/environment identities, rejected substitutions
and refusal reasons. It reads a `studio.operator-policy-decision.v1` source
export. A passing historical decision grants no execution authority: HAL checks
the current policy and estimate again at submission time.

The inspector exposes shot, declared concurrency, declared time-limit and cost
ceilings, exact backend/target/region bindings and policy validity dates. Price
amounts use decimal strings in the policy currency. An unknown amount stays
unknown and refuses a configured cost ceiling. The estimate retains its source,
observation time, expiry and complete-request SHA-256; it does not claim actual
charges or the account balance. Concurrency and time limits bound the declared
plan; they do not measure elapsed execution or enforce an account-wide quota.

Import preserves the previously admitted record when validation fails. Export
downloads the exact admitted source text, including integers above JavaScript's
safe-number range. Inspection and export make no provider calls or workspace
storage writes. The committed example uses synthetic conformance inputs.

Generate or check that offline example without submitting anything:

```bash
python tools/export_operator_policy_decisions.py --case unknown_price --output /path/to/new-decision.json
python tools/export_operator_policy_decisions.py --check --output data/studio/operator_policy_decisions.json
```

The output directory must already exist; generation refuses to overwrite a
file. For real source inputs, use `operator_request_from_settings` and
`assess_workspace_operator_policy` from `scpn_quantum_control.studio.workspace`.
The bridge preserves the original `ResolvedSettings` record and requires an
explicit backend, device, region, shots, concurrency, time limit and unattended
flag. The settings policy reference must bind the exact configured operator
policy using schema `operator_policy.v1` and its typed canonical body digest.
Imported settings cannot install that policy or substitute a source workload.

The shared browser runner's `operator_policy_decisions` scenario exercises
eight native verdicts through the actual inspector, exact export, malformed
import retention, network/storage observations and the original WASM recompute:

```bash
python tools/studio_browser_journey.py --scenario operator_policy_decisions \
  --base-url http://127.0.0.1:4173/ --output /path/to/policy-journey.json
```

## Operator review dossiers

Devices & Operations includes an **Operator review dossier** inspector. Import
a native `studio.operator-review-export.v1` export to inspect the original
execute plan, backend profile, exact compiled payload digest and byte count,
target, shots, requested/effective settings, policy verdict, dated estimate,
calibration reference and exclusive expiry. Integer settings retain their exact
precision. Unknown calibration or price stays explicitly unknown.

The native `ExecuteActionHandler.prepare_review` method takes the original
`ExecutiveRequest`, `BackendProfile`, `QuantumWorkload`, compiled bytes,
`ResolvedSettings`, `OperatorPolicyDecision`, calibration metadata or `None`,
and explicit UTC-second creation and expiry. It reuses the original execute
plan and no-submit projection, validates their source couplings and seals an
immutable `OperatorReviewDossier`. Payload bytes must match the original plan
digest; provider, target and shots must match the original policy request and
workload. Expiry cannot extend supplied policy, price or calibration validity.
When the native profile declares `max_shots`, the dossier retains that exact
positive integer and binds it into the profile and execution identities.
Hashes bind supplied metadata; they do not authenticate a provider, certify a
compilation or establish physical calibration accuracy.

`dossier.export_bundle()` retains the original inner dossier text and its
native generated Python verifier. The browser checks exact versions, source
identities and references and downloads those original bytes. It neither
generates executable code nor reassesses provider policy. A refused import
preserves the last admitted source and its exports. Editing a draft immediately
disables human review until that draft is admitted.

**Approve human review** and **Deny human review** create a separate in-memory
human record referencing the original dossier and execution identity. The
native equivalent is `dossier.record_review(choice, reviewer_ref=...,
recorded_at=...)`. Review grants no provider submission authority. Changing
payload, provider, target, shots, source plan/workload/profile, semantic settings
or their provenance, dated policy/price, calibration or expiry invalidates the
prior record. A display-only theme, notation, rounding or layout change
preserves execution identity and the original human source reference.

For a matching execution identity, pending, approved, denied, expired,
source-refused and future states remain distinct. Approval is unavailable
before source creation, after exclusive expiry or for a refused source verdict.
The current clock is checked again at the click and when the review hash
finishes; the view also updates expiry while it stays open. Human review export
contains the original reviewed export and a separate sealed review record.
These controls make no provider calls or browser storage writes and expose no
credential input.

The synthetic exporter provides reproducible offline examples:

```bash
python tools/export_operator_review_dossiers.py --output /path/to/new-review.json
python tools/export_operator_review_dossiers.py --check --output data/studio/operator_review_dossier.json
```

The destination directory must exist; generation refuses an overwrite. The
default example is frozen at `2026-10-04T00:00:00Z`; its source dates stay visible
after it expires. `--as-of` supplies exact UTC seconds for a fresh synthetic
conformance input. It does not query live pricing, accounts or calibration.
The inner dossier and compiled payload are each bounded to one MiB; the browser
accepts at most eight MiB of expanded UTF-8 export text.

**Export native verifier** downloads `verify_operator_review.py`. Running it
prints the original dossier. Supplying `--payload /path/to/compiled.bin` also
checks the exact original byte count and SHA-256, refusing missing, changed,
empty or oversized files. Both modes remain offline and never submit a job.
This verifier is distinct from the legacy provider submission script scaffold.

The shared real-browser scenario exercises native input changes, immutable
downloads, refused drafts, actual wall-clock expiry, storage/network custody and
the existing built WASM recompute:

```bash
python tools/studio_browser_journey.py --scenario operator_review_dossiers \
  --base-url http://127.0.0.1:4173/ --output /path/to/review-journey.json
```

## Settings and policy provenance

The workspace inspector shows the original requested and effective values,
their winning source layer, and the policy and environment references. It
reads the admitted archive without changing its settings or granting permission
to execute them. Exact integers remain exact in the table and in portable JSON.
Embedded `WorkspacePanel` consumers supply the original trusted raw-codec
registry for their policy and environment sources. Unknown producers refuse
archive admission; imported settings cannot install a verifier or policy.

Use `scpn_quantum_control.studio.workspace.resolve_settings` to resolve trusted
defaults, project, experiment and run layers, in that order. Supply an immutable
`SettingsPolicy`, the current environment reference and an existing HAL profile.
Every layer is validated, including a forbidden value later overwritten by a
valid run value. Shot, memory, qubit, concurrency and time-limit requests must have an explicit policy
ceiling; a request above the ceiling refuses and identifies its policy. Device
choices and the declared HAL route cannot be replaced by an imported request.
Resolution reads declarations and never submits a provider job.

`unattended` requires a literal boolean. `region` must match the governing HAL
profile; explicit `None` represents a non-geographic local route or an unknown
cloud region. A cloud operator decision refuses an unknown region. These
operational fields also contribute to `settings_plan_digest`.

`settings_plan_digest` identifies the settings contribution to a numerical
plan. Changes to precision, seed, shots, numeric parameters or units change this
identity. Theme, notation, plot rounding and layout do not; they still change the
full `ResolvedSettings.digest`, which also preserves provenance. Units are
labels of the admitted source values; no conversion is inferred.

```python
from scpn_quantum_control.studio.workspace import (
    confirm_settings_reset, export_settings, import_settings,
    preview_settings_reset, resolve_settings, settings_plan_digest,
)

# policy, environment_ref and profile come from the trusted application.
current = resolve_settings(
    defaults, project_values, experiment_values, run_values,
    policy=policy, environment_ref=environment_ref, profile=profile,
)
numerical_settings_hash = settings_plan_digest(current)
portable_text = export_settings(current)
imported_values = import_settings(portable_text)
candidate = resolve_settings(
    defaults, imported_values, {}, {},
    policy=policy, environment_ref=environment_ref, profile=profile,
)
reset = preview_settings_reset(
    current, defaults,
    policy=policy, environment_ref=environment_ref, profile=profile,
)
# Inspect reset.candidate first; confirmation checks the current full identity.
replacement = confirm_settings_reset(current, reset)
```

Portable `quantum_workspace_settings.v1` JSON contains supported requested
values only, with a 65,536-byte UTF-8 limit. Credentials, policy and environment
authority cannot be imported. Malformed Unicode, duplicate keys and future
versions refuse before returning values; unsupported versions require an
explicit supported migration. Import and reset preview do not alter saved state.
Imported values require a fresh resolution against the current policy. Confirming
a reset after the current record changed refuses; inspect a new preview instead.
The caller applies an accepted replacement explicitly through its normal saved
state transaction.

## Immutable workspace contracts

### Shared contract conformance

The shared fixture registry binds workspace documents, typed canonical bytes
and lossless transport to their real Python and TypeScript consumers. The
supported program-source corpus also runs through native Python, native Rust
and the TypeScript binding to the compiled Rust WASM module. These checks retain
the existing independent byte, digest, measurement-order, parameter and refusal
oracles; a source plan still means emitted source rather than executed hardware.

From the repository root, use the existing locked environment:

```bash
PYTHONPATH=src:oscillatools/src:. python -m tools.studio_contract_quality_gates
PYTHONPATH=src:oscillatools/src:. python -m tools.studio_contract_quality_gates --run
```

The first command validates the declared fixture, source, native-documentation
and dedicated-test owners plus their required CI aggregation. The second runs
all five actual consumers. It requires the already installed browser dependency
lock and compiled Studio WASM kernel; an unavailable runtime fails visibly.
Each consumer has a 120-second limit and failures stop the command immediately.
No retry or optional substitute can produce a passing conformance result.

`tools.ci_workflow_inventory.load_contract_cohorts` exposes the checked local
registry; `validate_contract_workflow` verifies the executable Studio owner,
mandatory conformance step and existing aggregate failure predicate. The Studio
workflow owns runtime parity. Native Python jobs run portable ownership and
refusal tests; the Studio category supplies Node, native Rust and WASM for the
runtime integration tests. Removing a required language, test argument, source
owner or aggregate dependency is a failure, as are conditional or error-tolerant
qualification steps. Original saved documents and corpus files remain unchanged
when a malformed input or deliberately damaged oracle is refused.

The Python `scpn_quantum_control.studio.workspace` API and the browser's
`src/shared/contracts/index.ts` export the same five metadata contracts:

| Schema | Record |
| --- | --- |
| `quantum_workspace.v1` | Project UUID, revision and artefact references, draft and UTC timestamps |
| `experiment_revision.v1` | Parent identities, problem/program inputs, typed parameters and resolved settings |
| `parameter_spec.v1` | Parameter dtype, shape, unit, domain, provenance and dependencies |
| `resolved_settings.v1` | Requested/effective values, origins and original policy/environment references |
| `local_run_record.v1` | Local run/attempt UUIDs, revision/plan identities, ordered events and outputs |

Parsers reject unknown schemas and fields, malformed references and unsupported
numeric values. The `extensions` object preserves optional metadata without
normalisation. Python records take recursively immutable snapshots; browser
records are deeply frozen. Exports return fresh containers. Parsing proves a
single document's structure. Call `admit_workspace` / `admitWorkspace` separately
with complete document, source, parameter-specification and unit indexes before
persisting an imported project.

```python
from scpn_quantum_control.studio.workspace import (
    admit_workspace, parse_workspace_manifest, read_json, write_json,
)

workspace = parse_workspace_manifest({
    "schema": "quantum_workspace.v1",
    "body": {
        "project_id": "10000000-0000-4000-8000-000000000001",
        "revision_refs": [], "draft_ref": None, "artefact_refs": [],
        "created_at": "2026-09-29T00:00:00Z",
        "updated_at": "2026-09-29T00:00:00Z",
    },
    "extensions": {"title": "Local experiment"},
})
wire_text = write_json(workspace.to_dict())
restored = parse_workspace_manifest(read_json(wire_text))
receipt = admit_workspace(restored, {}, {}, {}, {}, {})
```

Within the browser source, import through the shared public entry:

```typescript
import {
  admitWorkspace, documentToWire, parseWorkspaceManifest, readJson, writeJson,
} from "./shared/contracts";

const parsed = parseWorkspaceManifest(readJson(wireText));
if (!parsed.ok) throw new Error(`${parsed.path}: ${parsed.message}`);
const receipt = await admitWorkspace(
  parsed.value, new Map(), new Map(), new Map(), new Map(), new Map(),
);
if (!receipt.ok) throw new Error(receipt.message);
const exportedText = writeJson(documentToWire(parsed.value));
```

The empty indexes above apply to an empty workspace only. Imported nonempty
projects require their actual referenced documents and original producer bytes.
A revision must reference each parameter specification's digest in `input_refs`;
the external key index alone cannot change the revision's unit, domain or dtype.
Admission checks indexed identities, project ownership, parent and parameter
dependency graphs, typed values and exact unit labels. It does not convert units.
Policy, environment, problem, program and plan references retain their roles.

Workspace references always resolve through the validated document index; a raw
verifier cannot substitute for workspace document validation. Raw evidence uses
an explicit schema-to-verifier registry supplied by trusted
application code. Each offline verifier checks its original bytes and returns
its original schema, kind and digest. Unknown producers have no fallback. Imported
data cannot install a verifier, fetch a URL, execute source or launch a worker.
The receipt binds the exact root digest, including its extensions and reference
lists, as well as the verified document/source identities. It does not certify
scientific validation or execution success.

### Portable numeric identity

Use `read_json` / `readJson` and `write_json` / `writeJson` for workspace text.
Plain browser `JSON.parse` loses large integer precision and the distinction
between integer and floating-point tokens. Python integers correspond to browser
`bigint`; Python binary64 floats correspond to browser `number`. Lexical `-0`
is a negative-zero float in both readers. Writers retain float markers and
negative zero. Schema-defined safe integer fields such as shapes and event
sequences normalise to integers; opaque extension values retain their types.

The canonical digest is SHA-256 over the schema, LF and compact UTF-8 tagged
JSON of the **whole document envelope**, including extensions. Integers use
`["integer", "decimal"]`, floats use `["float64", "big-endian IEEE hex"]`, arrays
use `["array", [...]]`, and objects use `["object", [[key, value], ...]]` with
UTF-8 byte-sorted keys. Null, booleans and scalar strings retain their JSON form.
User arrays resembling a tag cannot impersonate a scalar. Strings preserve their
original Unicode spelling; invalid surrogate sequences, nonfinite floats,
ancestor cycles and unsupported objects refuse. Browser object accessors and
custom array prototypes refuse before their methods can influence serialization.

Both implementations allow at most 64 nested containers, 4,096 decimal digits
per integer scalar and 128 MiB of expanded JSON text. Archive admission remains
a separate boundary. Required text fields use the explicit Unicode White_Space
set, so language-specific trimming does not change acceptance. Timestamps require
ASCII UTC components and at most nine fractional second digits. Shape products
must fit a nonnegative safe integer; a zero dimension represents an empty tensor.
Typed elements support `float64`, `int64` and `uint64`, with exact finite or
bounded/enumerated domain checks.

The Studio CI category owns strict Python/TypeScript checks, native API docs,
the shared byte/transport/structural corpus and exact new-owner coverage gates.
These metadata APIs add no numerical kernel, backend or hardware support claim.
Existing Rust/PyO3, Julia and WASM numerical owners keep their original evidence
and codecs. Workspace persistence UI and worker lifecycle integration are
separate from these library contracts.


## Evidence inspector

The program-AD replay card uses the shared evidence inspector. It displays the
source schema, identifier and digest, claim scope, scientific kind/status,
admission, freshness, substrate, numeric parity declarations and provenance.
Absent fields stay visible as **Not supplied**. A seal can be missing, malformed,
or present but unverified; its presence does not validate a scientific claim.
Producer declarations and the result of local recomputation remain separate.

Use **Inspect evidence JSON** to paste an original
`studio.evidence-replay.v1` or `studio.hardware-result-pack.v1` bundle. Click
**Inspect snapshot** to load that text. The viewer projects the native schema-B
fields emitted by `scpn_quantum_control.studio.evidence_bundle`; it does not
implement the platform's admission or grading rules. Unsupported schemas and
partial metadata remain visible with limitations. References and provenance
commands are text only: the viewer does not fetch or execute them. A falsified
or refuted claim remains falsified or refuted, including when it has an
attestation or a separate bounded replay matches.

The same viewer accepts the original bounded program-AD replay artefact. It
passes that producer's binary64 JSON fields to the existing replay parser and
uses the shipped WASM verifier. A changed expected gradient yields a numerical
mismatch; altered input bytes fail their original SHA-256 binding. Neither
outcome is relabelled as proof of intentional forgery. Original source digests
are retained; workspace canonical encoding only binds the UI snapshot.

Verification state includes the complete snapshot, displayed source fields
and current input revision. Replacing any of them clears the displayed result
before the new input is painted. A result arriving for an earlier snapshot
cannot replace the current one. The inspector reuses the original
`useUnitBoundRun` lifecycle guard.

Browser applications can import `EvidenceInspector`, `EvidenceViewer`,
`projectEvidenceBundle` and `projectProgramAdEvidence` from
`src/shared/evidence/index.ts`. A verifier is an optional trusted application
callback, never a function selected by imported JSON. Formats without a
registered verifier explicitly report that verification is unavailable.

The hosted `evidence_inspector` browser journey uses the real bundle and WASM,
checks a delayed A result after B has finished, exercises altered source bytes,
and checks missing schema/source/seal and attested negative metadata. The timing
probe delegates to the browser's real SHA-256 implementation and counts its
completion; it supplies no substitute digest or replay result. Component and
projection tests have a separate exact-coverage gate in the Studio CI category.


## Declared resource admission

Kuramoto Play shows the float64 payload declared by the original RK4 path:
caller numeric values, encoded host and guest input, parsed native input,
six state/stage arrays, and native/guest/retained output. The memory ceiling is
a browser product setting, initially 4096 KiB, and may be reduced explicitly.
It does not report free workstation RAM. Unknown limits and unsupported metadata
refuse before input-vector construction, encoding or native allocation.

When a request is refused, the previous trajectory is cleared. The smaller
configuration button appears only after the complete plan for that proposed
oscillator/step pair passes the same policy. Applying it keeps the chosen
topology and float64 precision. No smaller configuration is chosen silently.

The shared `studio-web/src/shared/resources/index.ts` API also projects declared
statevector, density, adjoint, transfer and graph buffers with checked integer
arithmetic. Hilbert shapes are checked before shifting. Changing precision,
count or concurrency produces a new byte estimate; it does not grant numerical
backend support. An unknown backend's overhead must remain unknown and refuses
admission. A policy verdict supplements the original runtime checks.

The browser plan covers declared numeric payload only. JavaScript objects,
allocator bookkeeping, module baseline storage and stack usage are outside
that byte declaration. A zero extra-byte declaration means no extra payload
was declared; it is not measured zero overhead. Declared derivative workload
units are bounded by the module's oscillator and step limits. They are not
elapsed seconds, measured throughput or a wall-clock deadline. Host/container
execution continues to use the existing `check_execution_memory` observation
and admission owner; browser product ceilings cannot replace those observations.

The hosted `resource_plan_projection` journey uses the built WASM and the real
public controls. Counters forward native allocations and worker construction
unchanged, observe no new worker after a zero-byte refusal, and observe owned
computation resume after an explicitly applied smaller same-method plan.
The admitted worker plan includes both retained and transferred binary bytes.
Invocation from the repository:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py --scenario resource_plan_projection \
  --base-url http://127.0.0.1:4173/ --output /tmp/studio-resource-journey.json
```

Use an owned preview of the built bundle; the command refuses external navigation.
The suite is wired to hosted CI. Wiring is not a statement that it has passed.

The 3D Lab also declares retained phase history, order/phase series, the current
phase copy and scene coordinates before capture and geometry construction.
Scene-object storage, SVG strings and DOM overhead remain explicitly outside
the numeric payload estimate. Lab limits and numerical capture/parity checks
remain the original owners; this declaration does not replace them.

A requested wall-clock ceiling currently refuses before allocation with
`wall_clock_admission_unavailable`. The kernel has no qualified
elapsed-time predictor, so Studio cannot
admit a guaranteed deadline by converting workload units into invented seconds.
Leaving this optional field unset requests the bounded workload without a
wall-clock guarantee; it does not infer a deadline. No smaller-shape action
claims to repair an unsupported time guarantee.

## Owned browser simulation

Kuramoto Play runs the shipped Rust/WASM kernel in a worker. Each mounted panel
owns one active worker at a time. Playback and the original committed reference
run sequentially. Float64 inputs, RK4 equations, source time units and the native
binary codec retain their original owners. Display sampling does not supply a
numerical seed; the existing controls produce deterministic input values.

The resource plan admits the complete declared numeric buffer chain, transport
copies, source-codec validation and retained/transferred binary before a worker
or transferable copy is created. Reference admission also reserves the retained
playback arrays. Saved input arrays, source bytes and previous results are never
transferred. A shared backing buffer, a malformed source request, an unavailable
binary or an exceeded source limit gives a visible refusal. JavaScript objects,
canonical metadata strings, engine baseline storage and allocator bookkeeping
remain outside this payload declaration; this is not a measured heap bound.

Use **Cancel simulation** to stop the current operation. Changing its input,
leaving the route or project, or unmounting the panel disposes its worker and
clears its current plot. A late or duplicate event cannot restore a result for
an earlier input. **Run simulation** starts an explicit new operation using the
current settings after cancellation or recovery.
The cancellation control stays available after completion so keyboard focus
keeps its target; cancelling a completed run clears its displayed plot.

**Simulation timeout (ms)** accepts 1 through 60000 and defaults to 5000. It is
an operational termination timer for each worker. Browser scheduling can delay
delivery; it does not guarantee that computation will finish within that time.
The separate optional wall-clock guarantee retains its explicit refusal.
Completed results and cancellation are published only after the owner observes
disposal. Failed disposal stays visible as an unconfirmed disposal and blocks
further runs in that mounted panel. The host must dispose the outstanding worker
before reopening the panel.

The public `createOwnedKuramotoRun` facade captures a v1 envelope with run,
input-revision, numerical-plan and original WASM SHA-256 identities. Events carry
monotonic sequences. The worker verifies the transferred binary digest and its
native bounds before acceptance, then uses the original simulation binding.
Existing synchronous `simulate` callers remain supported. A custom Play loader
must provide the original `sourceBytes`; a closure without its binary cannot
qualify owned execution and visibly refuses. `instantiateKuramoto` and
`fetchKuramoto` supply those captured bytes.

The shared real-browser runner observes the deployed worker, the independent
two-oscillator reference, native transfer ownership, refusal, cancellation,
timeout and route/project disposal:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py --scenario owned_kernel_worker \
  --base-url http://127.0.0.1:4173/ --output /tmp/studio-owned-worker-journey.json
```

The build manifest requires exactly one emitted worker asset and records its
digest and size alongside the original WASM kernels. An unused worker module
omitted from the built bundle cannot qualify that manifest.


## Linked parameter editing

Preview or restore a supported complete experiment archive in Workspace to
open its parameter editor. The host must provide the archive's original source
verifiers. An empty project has no parameters to edit. Kuramoto Play links back
to Workspace; its bounded playback controls remain a separate instrument.

Parameter specifications supply each label, dtype, shape, unit, domain and
trainable eligibility. Scalar and vector forms, matrix cells and sparse edge
buttons share one selection and draft. For a square matrix, coefficient `[i,j]`
appears as edge `j → i`; negative coefficients are dashed in the diagram. Zero
coefficients have no sparse edge and can be selected through the matrix.

Directed mode edits only the selected coefficient. Choose Symmetric explicitly
to update both coefficients on a subsequent edit; changing the policy alone
does not repair an asymmetric matrix. Trainable checkboxes select a subset of
the original specification's eligibility and follow the same pair policy.

Apply value validates decimal text in the original unit. A separate conversion
button supports float64 SI prefixes: `rad`/`mrad`/`urad`, `s`/`ms`/`us`, and
`rad/s`/`mrad/s`/`rad/ms`. Conversion uses binary64 multiplication and retains the
original stored unit. Other units and integer conversions refuse. Integer
values remain exact decimal strings; float64 negative zero retains its sign.
Nonfinite values, invalid domains, incompatible shapes or units leave the
current semantic draft and saved archive unchanged.

Saving waits for the draft identity and requires all form changes to be applied.
If identity calculation fails, Retry draft identity repeats it without changing
the draft or saved archive. Restore browser cryptography support before retrying.

Undo and redo restore exact draft identity, with up to 64 retained edits.
The editor supports scalar, vector and matrix shapes with at most 4096 total
elements. Save parameter revision appends a new immutable child through the
original workspace transaction. Earlier revision documents, raw source bytes
and result references remain intact. Reload restores the saved values and
trainable mask. Saving grants no execution authority and does not attach old
results to the edited revision. Export a portable backup after saving.

## Local workspace and portable archives

The Local workspace editor keeps the exact archive text, immutable revision
references and original source bytes in native IndexedDB. Create an empty
project, read a local JSON archive, or edit the archive text. An empty project
contains no experiment or result. Editing invalidates the preview while the
saved workspace identity remains visible and unchanged.

Preview checks the entire document/reference graph, typed parameters, source
identities, versions and paths before enabling save. Apply records the complete
archive, project head and selected project in one native transaction. Reload
revalidates the saved archive and restores its exact text; conflicts from
another tab require an explicit reload/reconciliation. Interrupted writes,
quota failures and corrupt saved data surface errors. Missing/evicted cache
requires an independent exported copy rather than a manufactured success.

Export preview archive downloads the current admitted preview. Export saved
archive downloads the last committed archive. These are separate actions:
unpreviewed editor changes do not silently replace the saved backup. Keep an
independent copy; browser cache is not server durability or an independent
backup. Exported files are not encrypted and retain the supplied source data.

The portable container is `quantum_workspace_archive.v1`. Its manifest is the
original `quantum_workspace.v1` document; members carry immutable document JSON
or exact original binary bytes as lowercase hex, with safe relative names,
source schema and SHA-256 identities. Parameter units remain explicit. The
encoded JSON limit is 64 MiB including hex/JSON overhead, decoded member data is
bounded at 128 MiB, at most 1000 members include the root, and nesting is bounded
at 64 levels. ZIP archives, links, executable members, duplicate names/identities,
unsafe paths, corrupt references and unsupported major versions are refused.
The supported version preview reports that no migration is required; no
unknown schema is automatically migrated or allowed to change saved data.

Original source verification belongs to the hosting application. It can supply
already-qualified offline producer codecs through `QuantumStudioPanel`'s
`rawCodecs` property. Imported files cannot install verifiers. Source formats
without an available verifier refuse; the standalone panel does not declare
all scientific producers supported. Workspace canonical hashing never replaces
an original producer's raw digest. Saving or editing grants no numerical,
provider or hardware execution authority and does not rebind the existing
committed result panels to a changed draft.

The hosted browser acceptance command uses two explicitly owned loopback servers:

```bash
PYTHONPATH=. python tools/studio_browser_journey.py \
  --scenario workspace_recovery --base-url http://127.0.0.1:4173/ \
  --workspace-source-url http://127.0.0.1:4174/ --output workspace-journey.json
```

The first serves the built Studio/WASM bundle. The second serves the original
source API for native IndexedDB cases in isolated test contexts. UI export/import
is exercised separately from full-reference synthetic metadata conformance.
Test-only producer registries never enter the production panel. Recorded V8
counters are raw observations; they require original-source mapping and exact
coverage evaluation before any percentage is claimed.

The native recovery journey also attempts path-escaping, oversized, corrupt-identity and future-major archives against a saved project, and checks that the exact prior archive and selection survive each refusal. Separate native cache fault cases remove an archive, corrupt its head and evict the cache; explicit restore uses the exported copy. Failed journeys retain completed observations, original failure diagnostics and already captured native script counters. These are acceptance assertions; authored checks alone do not establish runtime success.

Browser coverage qualification retains the executed script and original owner SHA-256, verifies the inline source map against exact checkout text, and converts actual V8 ranges with the locked converter. The same Vitest V8 provider merges native counters before its unchanged global coverage gate. Set `STUDIO_WORKSPACE_COVERAGE` to the successful `workspace_recovery` JSON when running the combined coverage cohort. Failed, stale, incomplete or unowned evidence refuses; raw counters alone are not a coverage percentage. The owned source acceptance page mounts the real workspace component with an explicitly synthetic test-only producer registry, separately from the built Studio/WASM journey.

The Python browser runner and workspace orchestration, native counter capture and UI helper have a separate branch-coverage gate. Studio CI collects their actual execution across the dedicated URL tests and existing browser journeys, then requires 100% for these complete owners. Its raw coverage data and measured JSON report are retained with the browser evidence. The gate declaration does not establish a passing result; acceptance requires the actual report from the source being reviewed.

Recovery acceptance includes two actual tabs preparing the same native head before one commits and the stale tab attempts its save. The test host also imports the full original synthetic revision/evidence graph through the real File input, checks preview hashes, downloads preview and saved archives, reloads exact saved text, and refuses a future major without changing the saved graph. A fresh isolated context repeats the graph UI import/export. This qualifies storage and host-codec mechanics only; the synthetic producer is never installed in the built product registry.

Archive files must contain valid UTF-8. Malformed byte sequences refuse before replacing the current editor. A leading UTF-8 BOM is preserved in the editor and refused by the original strict JSON reader during preview; it is never silently removed to make an otherwise unsupported archive pass. Saved revisions and original evidence remain unchanged.

Each saved archive snapshot binds its exact JSON source text, including formatting, under the `quantum_workspace_archive_source.v1` canonical string domain. Formatting-only edits create a new snapshot while preserving the original workspace, revision and evidence hashes. Saving over an existing head validates the prior saved archive before the native write transaction, then checks the same head and exact prior source again inside it. Missing, inconsistent or concurrently altered prior cache refuses; it is not silently overwritten by the new draft.


The `workspace_panel_refusal` scenario renders the original Studio panel against
isolated copies of seven damaged JSON inputs. It requires the original facade
and every workspace storage/controller owner, observes nine source refusals,
and checks that the local editor stays mounted without numerical views. The CI
fixture verifies that canonical JSON and copied production source bytes remain
unchanged. Its successful evidence is supplied as `STUDIO_PANEL_REFUSAL_COVERAGE`
alongside workspace recovery evidence; both pass the same source/hash/mapping
qualification before the original coverage provider merges actual counters.
Python runner coverage uses greenlet-aware collection for the synchronous
Playwright lifecycle. Plain thread collection misses code executed after its
context switch and is insufficient evidence for the complete browser owners.
The dedicated runtime refusal cohort serves the original built UI and WASM
through owned HTTP hosts. It checks uncaught page errors, unowned requests,
truncated kernel responses and HTTP 503 during native imports. Failed native
journeys retain their captured counters and partial diagnostics. Closing the
profiled page after delivery of its counters preserves those records and the
actual profiler cleanup error.


`previewWorkspaceArchive(text, codecs, expandedByteLimit)` accepts an optional
smaller positive integer budget for the root and decoded members. It refuses
budgets above the 128 MiB product ceiling. The root counts even when there are
no members; the preview accepts an exact byte bound and refuses an excess
before producer admission or storage mutation. This parameter declares an
application budget and does not represent observed host memory.


Native cache opening has a 10-second product deadline. A queued request behind
another tab's pending database operation refuses explicitly instead of keeping
the editor busy indefinitely. If that native request later succeeds, the
refused connection is closed. `openWorkspaceStore` accepts a shorter positive
integer timeout as its third argument and refuses a value above the product
bound. Saved cache contents and exported archives are separate from this
opening deadline.

The native quantum reference APIs are documented in the
[Lindblad guide](lindblad.md#analytic-single-qubit-reference). They admit a
shared physical initial density matrix, preserve explicit Hamiltonian basis
ordering and distinguish output sampling from integration accuracy. Their
local analytic and Python/Rust kernel conformance supplies a numerical
foundation; it does not add a quantum solver panel or execution authority to
the Studio workspace.
