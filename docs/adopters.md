# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Adopters

# Adopters and Case Studies

## Self-adoption (GOTM ecosystem)

`scpn-quantum-control` is consumed inside the same ecosystem that
builds it. These repos depend on its public API and ship changes
alongside it.

| Consumer | How it uses scpn-quantum-control | Evidence |
| --- | --- | --- |
| [sc-neurocore](https://github.com/anulum/sc-neurocore) | Classical spiking-neural-network engine. Feeds spike trains into `scpn_quantum_control.bridge.snn_adapter.ArcaneNeuronBridge`, which maps them onto a quantum XY Hamiltonian and returns current-feedback to close the loop. | `src/scpn_quantum_control/bridge/snn_adapter.py` + sc-neurocore's `cross_repo_wiring.py` + CI job `integration-optional` in `.github/workflows/ci.yml`. |
| [scpn-phase-orchestrator](https://github.com/anulum/scpn-phase-orchestrator) | SCPN phase-dynamics orchestrator. Invokes `scpn_quantum_control.bridge.ssgf_adapter.SSGFQuantumLoop` for the quantum-in-the-loop phase-update step. | `src/scpn_quantum_control/bridge/ssgf_adapter.py`. |
| [scpn-control](https://github.com/anulum/scpn-control) | Plasma / disruption control. Uses `scpn_quantum_control.control.q_disruption_iter` (ITER 11-feature disruption classifier) and the `hardware/` runner to benchmark against classical baselines. | `src/scpn_quantum_control/control/q_disruption_iter.py`. |
| [scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) | Tokamak digital twin. Cross-references the DLA parity asymmetry result (Phase 1 campaign on `ibm_kingston`) for quantum-informed stability analysis. | `docs/results.md` §Phase 1; `data/phase1_dla_parity/`. |
| remanentia | Persistent-memory MCP. Indexes `docs/`, `paper/`, and `CHANGELOG.md` for cross-repo recall via `remanentia_recall`. | `.mcp.json` at the GOTM root. |

## External adopters

None currently documented. If you are using `scpn-quantum-control`
in research, teaching, or a commercial product and are willing to
be listed, open a pull request that adds a row to the table below,
or email `protoscience@anulum.li`. Listing is opt-in, one line per
adopter, and takes the form:

```markdown
| [Your project / group name](link) | 1-sentence description of the use case | Link to the code or paper that calls scpn-quantum-control |
```

| Adopter | Use case | Evidence |
| --- | --- | --- |
| _(empty — add your row here)_ | | |

## Self-attribution in your own project

If you want to signal that your project uses
`scpn-quantum-control` but do not need a row in the table above, a
small `Powered by` badge in your README is welcome:

```markdown
[![Powered by scpn-quantum-control](https://img.shields.io/badge/powered%20by-scpn--quantum--control-6929C4)](https://github.com/anulum/scpn-quantum-control)
```

The badge is under the same AGPL / commercial licence as the rest
of the repository; see `LICENSE` and `LICENSES/`.

## Citing this work

Regardless of whether you appear in the adopters table, please
cite via `CITATION.cff` (rendered at
[citeas.org](https://api.citeas.org/product/https://github.com/anulum/scpn-quantum-control))
or directly:

```bibtex
@software{sotek2026scpnqc,
  author  = {Šotek, Miroslav},
  title   = {scpn-quantum-control: Quantum-Native SCPN Phase Dynamics
             and Control},
  year    = {2026},
  version = {0.9.6},
  doi     = {10.5281/zenodo.18821929},
  url     = {https://github.com/anulum/scpn-quantum-control}
}
```

## Review cadence

Annual. If the adopters table hits more than 25 rows, split it
into academic / industrial / educational subsections.

Audit item **C15** in
the internal gap audit stays
partly open until the first external-adopter row lands.
