<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- entanglement/tomography readiness manifest -->

# Phase 3 Entanglement/Tomography Readiness

Date: 2026-05-07

## Decision

- Readiness decision: `ready_for_optional_hardware_preregistration`
- Ready for optional hardware: `True`
- Hardware submission: `False`
- QPU minutes spent: `0.0`
- Mode: `reduced_pauli_tomography`
- Basis settings: `9`
- Total optional hardware circuits: `166`

## Artefacts

- JSON summary: `data/phase3_entanglement_tomography/entanglement_tomography_readiness_2026-05-07.json`
- Observable rows: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/generate_entanglement_tomography_readiness.py
```

## Boundary

This readiness package provides exact classical references and circuit-count
gates only. It is not hardware evidence and does not authorise QPU
submission.
