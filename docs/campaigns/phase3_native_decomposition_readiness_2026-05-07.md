<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- native decomposition readiness manifest -->

# Phase 3 Native Decomposition Readiness

Date: 2026-05-07

## Decision

- Readiness decision: `blocked_current_xy_invalid_no_native_gain_vs_generic`
- Ready for optional hardware: `False`
- Hardware submission: `False`
- QPU minutes spent: `0.0`

## Artefacts

- JSON summary: `data/phase3_native_decomposition/native_decomposition_readiness_2026-05-07.json`
- Transpile rows: `data/phase3_native_decomposition/native_decomposition_transpile_rows_2026-05-07.csv`
- Equivalence rows: `data/phase3_native_decomposition/native_decomposition_equivalence_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/generate_native_decomposition_readiness.py
```

## Boundary

This readiness package is an offline compiler/equivalence gate. It is
not hardware evidence and does not authorise QPU submission.
