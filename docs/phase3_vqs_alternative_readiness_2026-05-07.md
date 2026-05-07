<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- VQS alternative readiness manifest -->

# Phase 3 VQS Alternative Readiness

Date: 2026-05-07

## Decision

- Readiness decision: `blocked_no_vqs_candidate_passed_promotion_gate`
- Ready for optional hardware: `False`
- Hardware submission: `False`
- QPU minutes spent: `0.0`

## Artefacts

- JSON summary: `data/phase3_vqs_alternative/vqs_readiness_2026-05-07.json`
- Candidate rows: `data/phase3_vqs_alternative/vqs_candidate_rows_2026-05-07.csv`
- Resource rows: `data/phase3_vqs_alternative/vqs_resource_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/generate_vqs_alternative_readiness.py
```

## Boundary

This readiness package is an offline exact-state and resource gate. It is
not hardware evidence and does not authorise QPU submission.
