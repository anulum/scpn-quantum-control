<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- layer-selective comparator manifest -->

# Phase 3 Layer-Selective Comparator Matrix

Date: 2026-05-07

## Decision

- Backend: `ibm_marrakesh`
- Readiness decision: `blocked_layer_selective_worse_than_default`
- Ready for hardware comparison: `False`
- Hardware submission: `False`
- QPU minutes spent: `0.0`

## Artefacts

- JSON summary: `data/phase3_layer_layout/layer_selective_comparator_matrix_ibm_marrakesh_2026-05-07.json`
- Comparator rows: `data/phase3_layer_layout/layer_selective_comparator_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/generate_layer_selective_comparator_matrix.py
```

## Claim Boundary

This artefact is a no-submit transpilation-resource matrix. It can
promote or block the optional hardware follow-up, but it is not
hardware evidence and does not authorise QPU submission.
