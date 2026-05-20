<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- entanglement/tomography IBM execution package -->

# Phase 3 Entanglement/Tomography IBM Execution Package

Date: 2026-05-20

## Decision

- Campaign: reduced-Pauli entanglement/tomography follow-up.
- Backend preflight target: `ibm_marrakesh`.
- Live preflight artefact:
  `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T001956Z.json`.
- Completed execution artefact:
  `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T004334Z.json`.
- Status: `completed`.
- Hardware submission performed: `True`.
- Job IDs: `d86g7h1789is738vkreg`, `d86ggpis46sc73f6v170`.

## Live Preflight Summary

| Field | Value |
|---|---:|
| Main circuits | 162 |
| Readout circuits | 4 |
| Total circuits | 166 |
| Main shots | 2048 |
| Readout shots | 8192 |
| Estimated QPU minutes | 1.5217 |
| Budget ceiling minutes | 25.0 |
| Selected physical qubits | `[1, 2, 3, 4]` |
| Mean readout error | 0.005584716796875 |
| Backend pending jobs | 0 |
| Maximum transpiled depth | 388 |
| Maximum basis-expansion ratio | 1.0718232044198894 |

The preflight passed the preregistered guards: the circuit count remains below
the DLA+FIM ceiling, estimated QPU time remains below the budget ceiling, and
measurement-basis expansion remains below the `1.20` depth-ratio guard.

## Execution Commands

Live readiness only:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh
```

Approved QPU submission:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh --submit --confirm-budget
```

Post-run raw-count analysis:

```bash
python scripts/analyse_phase3_entanglement_tomography.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_<backend>_<timestamp>.json
```

## First-Pass Analysis

Post-run analysis produced:

- JSON summary:
  `data/phase3_entanglement_tomography/entanglement_tomography_summary_2026-05-20.json`
- Observable rows:
  `data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20.csv`
- Manifest:
  `docs/phase3_entanglement_tomography_manifest_2026-05-20.md`

Result snapshot:

| Metric | Value |
|---|---:|
| Observable rows | 54 |
| Mean absolute deviation from exact reference | 0.12989296537986128 |
| Maximum absolute deviation from exact reference | 0.5560906424788263 |
| Rows SHA256 | `3d18308d60fe32827bae7517f18fd71690240b105779287408c4749cb0e7dc72` |

## Paper Spin

Working title:

> Reduced-Pauli entanglement checks for DLA-sector leakage mechanisms on IBM
> Heron hardware

Core result class:

- compare measured reduced-Pauli correlators against exact classical
  references for the promoted DLA parity and FIM-pair circuits;
- report whether the leakage/retention mechanism has a measurable
  entanglement-structure companion;
- include a null or negative result as a bounded measurement-cost and
  mechanism-separation result.

Blocked claims:

- quantum advantage;
- scalable tomography;
- backend-general entanglement dynamics;
- full-state reconstruction;
- claims on unmeasured subsystems, unmeasured depths, or unmeasured backends.

Paper-facing artefacts:

- Plan: `docs/phase3_entanglement_tomography_paper_plan_2026-05-20.md`
- Canonical LaTeX source:
  `paper/phase3_entanglement_tomography/phase3_entanglement_tomography.tex`
- Built PDF:
  `paper/phase3_entanglement_tomography/phase3_entanglement_tomography.pdf`
- Historical Markdown scaffold:
  `paper/phase3_entanglement_tomography/phase3_entanglement_tomography_short_paper.md`

## Completed Extensions

The initial execution package was extended on the same date with second-backend
replication, pinned-layout full correlated readout calibration, and a
preregistered five-channel ZNE stress test on `ibm_fez`.

Extension artefacts:

- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T014536Z.json`
- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T020452Z.json`
- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T023204Z.json`
- `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T023600Z.json`
- `data/phase3_entanglement_tomography/entanglement_zne_summary_2026-05-20_ibm_fez_zne.json`
- `docs/phase3_entanglement_zne_manifest_2026-05-20_ibm_fez_zne.md`

Extension jobs:

- backend replication and full-readout calibration:
  `d86hdk8p0eas73dkv9eg`, `d86hedp789is738vm7mg`
- ZNE subset and 16-state readout calibration:
  `d86hs6qs46sc73f70h90`, `d86hsltg7okc73el4lg0`

ZNE snapshot:

| Metric | Value |
|---|---:|
| ZNE scale rows | 15 |
| ZNE channels | 5 |
| Scale-1 mean absolute deviation | 0.4196274906488838 |
| Linear-ZNE mean absolute deviation | 0.44124749932943935 |
| Full-readout-mitigated linear-ZNE mean absolute deviation | 0.4468173734963394 |
| Quadratic-ZNE mean absolute deviation | 0.4477931807530503 |

The ZNE extension is interpreted as a mechanism stress test only. The simple
global-folding lane does not erase the four dominant DLA transverse deviations;
the FIM control channel behaves differently and improves under linear
extrapolation.

## Boundary

This package records a completed, approved QPU submission and first-pass
reduced-Pauli analysis. It supports only the bounded claim that measured
small-system reduced-Pauli correlators deviate from exact references under the
specified backend, layout, circuit family, shots, repetitions, and calibration
window. It does not support scalable tomography, quantum advantage, or
backend-general entanglement dynamics.
