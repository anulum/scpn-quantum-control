<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- entanglement/tomography paper plan -->

# Phase 3 Entanglement/Tomography Paper Plan

Date: 2026-05-20

## Working Title

`Reduced-Pauli entanglement checks for DLA-sector leakage mechanisms on IBM Heron hardware`

Alternative:

`Mechanism-boundary tomography for parity-sector leakage in Kuramoto-XY quantum circuits`

## Thesis

The paper should not try to claim a new advantage result. Its strongest clean
angle is mechanism separation:

- the earlier DLA-parity programme showed a real, backend-sensitive leakage
  asymmetry;
- the Phase 3 state/layout controls showed that layout and prepared state can
  dominate the original contrast;
- this campaign asks whether the promoted DLA and FIM circuits also show a
  measurable reduced-Pauli entanglement-structure companion.

The paper remains publishable if the result is null: a bounded negative result
would show that reduced-Pauli tomography at this cost does not explain the
observed leakage mechanisms.

## Evidence Status

Completed:

- reduced-Pauli readiness package;
- exact reference rows for 54 observables;
- approval-gated IBM runner;
- no-submit live preflight on `ibm_marrakesh`;
- approved QPU submission on `ibm_marrakesh`;
- raw-count artefact with job IDs;
- generated observable-summary CSV and manifest;
- post-run raw-count reducer;
- paper asset generator with label summary, basis summary, top-deviation
  table, and heatmap figure;
- second-backend `ibm_fez` replication;
- pinned-layout `ibm_fez` repeat with full 16-state correlated readout
  calibration;
- preregistered five-channel ZNE stress test on `ibm_fez` physical qubits
  `[21,22,23,24]`;
- ZNE reducer and manifest:
  `docs/campaigns/phase3_entanglement_zne_manifest_2026-05-20_ibm_fez_zne.md`.

Current first-pass result:

- completed artefact:
  `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T004334Z.json`;
- jobs: `d86g7h1789is738vkreg`, `d86ggpis46sc73f6v170`;
- 54 observable rows;
- mean absolute deviation from exact reference: `0.12989296537986128`;
- maximum absolute deviation from exact reference: `0.5560906424788263`;
- observable rows SHA256:
  `3d18308d60fe32827bae7517f18fd71690240b105779287408c4749cb0e7dc72`.
- paper asset manifest:
  `data/phase3_entanglement_tomography/entanglement_tomography_paper_assets_2026-05-20.md`.

Pending:

- final venue formatting and bibliography pass.

Completed ZNE stress-test result:

- completed artefact:
  `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T023600Z.json`;
- jobs: `d86hs6qs46sc73f70h90`, `d86hsltg7okc73el4lg0`;
- 15 scale rows over 5 preregistered channels;
- scale-1 mean absolute deviation: `0.4196274906488838`;
- linear-ZNE mean absolute deviation: `0.44124749932943935`;
- full-readout-mitigated linear-ZNE mean absolute deviation:
  `0.4468173734963394`;
- channel-summary SHA256:
  `775594dd3efc8f519111365179659d41903cf774bcde1d20a378d53b6ee8e7aa`.

## Minimum Run

| Field | Value |
|---|---:|
| Backend | `ibm_marrakesh` |
| Physical qubits | `[1, 2, 3, 4]` |
| Main circuits | 162 |
| Readout circuits | 4 |
| Total circuits | 166 |
| Main shots | 2048 |
| Readout shots | 8192 |
| Estimated QPU minutes | 1.5217 |
| Budget ceiling minutes | 25.0 |
| Max transpiled depth | 388 |
| Max basis-expansion ratio | 1.0718232044198894 |

Executed submission command:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh --submit --confirm-budget
```

Executed analysis command:

```bash
python scripts/analyse_phase3_entanglement_tomography.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_<backend>_<timestamp>.json
```

## Claim Boundary

Safe after successful raw-count analysis:

- reduced-Pauli correlators were measured for the preregistered DLA and FIM
  circuit families;
- measured correlators can be compared with exact references under a fixed
  layout and shot budget;
- the result either supports or rejects a small-system entanglement-structure
  companion to the leakage/retention mechanism.

Blocked even after a positive result:

- quantum advantage;
- scalable tomography;
- backend-general entanglement dynamics;
- full-state tomography;
- claims about unmeasured qubits, depths, layouts, or backends.

## Result Decision Tree

| Raw-count outcome | Paper framing |
|---|---|
| Correlators track exact references within uncertainty | Hardware supports the reduced-Pauli reference structure for the promoted circuits; leakage mechanism is not explained by gross entanglement-observable drift. |
| Correlators deviate coherently by family or depth | Entanglement-structure companion is visible and becomes a candidate mechanism for follow-up controls. |
| Uncertainty dominates deviations | Measurement-cost boundary; the promoted tomography block is too small to support mechanism claims. |
| Readout correction flips or erases the comparison | Readout-calibration boundary; no entanglement interpretation should be promoted. |

## Figures and Tables

Minimum paper assets:

1. Table of preflight gates: circuits, shots, depth, basis-expansion ratio,
   QPU-minute estimate.
2. Heatmap of measured minus exact Pauli expectations by circuit label and
   observable.
3. Family-level aggregate table for DLA shallow/signal and FIM lambda pair.
4. Largest-deviation table for the dominant measured correlator channels.
5. Boundary table: supported claims, blocked claims, falsifiers.
6. ZNE stress-test table for the four DLA transverse channels and one FIM
   control channel.

Generated assets:

- `figures/phase3/phase3_entanglement_deviation_heatmap_2026-05-20.png`
- `figures/phase3/phase3_entanglement_deviation_heatmap_2026-05-20.pdf`
- `data/phase3_entanglement_tomography/entanglement_tomography_label_summary_2026-05-20.csv`
- `data/phase3_entanglement_tomography/entanglement_tomography_basis_summary_2026-05-20.csv`
- `data/phase3_entanglement_tomography/entanglement_tomography_top_deviations_2026-05-20.csv`

## Submission Target

Best fit after raw counts:

- workshop or short communication first, then expand if the hardware result is
  mechanistically sharp;
- categories: `quant-ph`, with possible `cs.ET` cross-list if positioned as a
  reproducibility/control package;
- do not submit as a quantum-advantage paper.

## Current Manuscript Source

Manuscript source:

- `paper/submissions/submission_005_phase3_reduced_pauli_entanglement/phase3_entanglement_tomography.tex`

Built PDF:

- `paper/submissions/submission_005_phase3_reduced_pauli_entanglement/phase3_entanglement_tomography.pdf`

Historical scaffold:

- `paper/submissions/submission_005_phase3_reduced_pauli_entanglement/phase3_entanglement_tomography_short_paper.md`

The canonical LaTeX manuscript now records the completed raw-count execution,
readout-sensitivity extensions, second-backend replication, full correlated
readout calibration, and the five-channel ZNE stress test. Next paper work is
venue-specific formatting and final bibliography polishing, not additional QPU
spend for this paper.
