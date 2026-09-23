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
shows these as capability declarations. It does not provide a general browser
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
