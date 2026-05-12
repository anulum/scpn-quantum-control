<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- multi-circuit QEC readiness manifest -->

# Phase 3 Multi-Circuit QEC Readiness

Date: 2026-05-07

## Decision

- Readiness decision: `blocked_physics_aware_decoder_did_not_beat_baselines`
- Ready for optional hardware: `False`
- Hardware submission: `False`
- QPU minutes spent: `0.0`
- Max encoded depth: `38`
- Max encoded qubits: `68`

## Artefacts

- JSON summary: `data/phase3_multicircuit_qec/qec_readiness_2026-05-07.json`
- Decoder rows: `data/phase3_multicircuit_qec/qec_decoder_rows_2026-05-07.csv`
- Resource rows: `data/phase3_multicircuit_qec/qec_resource_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/generate_multicircuit_qec_readiness.py
```

## Boundary

This readiness package is an offline distance-3 surface-code
logical-metric gate. It is not hardware evidence, not a
fault-tolerance claim, and does not authorise QPU submission.
