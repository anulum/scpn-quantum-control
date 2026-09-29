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
