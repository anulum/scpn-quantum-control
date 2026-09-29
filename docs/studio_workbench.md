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


## Immutable workspace contracts

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
