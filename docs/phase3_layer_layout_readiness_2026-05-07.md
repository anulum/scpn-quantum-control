<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Layer-Selective Readiness Manifest -->

# Phase 3 Layer-Selective Readiness Manifest

Date: 2026-05-07

## Decision

- Ready for hardware comparison: `False`
- Readiness decision: `blocked_missing_comparators`
- Backend: `ibm_marrakesh`
- Hardware submission: `False`
- QPU minutes spent: `0.0`

## Artefacts

- JSON summary: `data/phase3_layer_layout/layer_selective_readiness_ibm_marrakesh_2026-05-07.json`
- Resource rows: `data/phase3_layer_layout/layer_selective_transpile_rows_2026-05-07.csv`
- Source artefact: `data/phase3_state_layout_dla/phase3_state_layout_ibm_marrakesh_2026-05-06T224531Z.json`
- Source SHA256: `03068ddaa9794f1ac19614e700887a84dd013cd5af107f49b39c3cff9e5674ac`

## Blocker

Generate default, SABRE, and true layer-selective transpilation rows from a fresh backend snapshot before considering the 152-circuit hardware follow-up.

## Reproduction

```bash
./.venv-linux/bin/python scripts/analyse_layer_selective_readiness.py
```
