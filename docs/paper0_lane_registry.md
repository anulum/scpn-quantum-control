<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Paper 0 lane registry -->

# Paper 0 Lane Registry

Status: source-bounded programme registry; no QPU or hardware submission.

This generated artefact turns the processed Paper 0 source register into
downstream experimental lanes. It is an index of candidate work and
blockers, not external validation evidence.

## Registry summary

- Schema: `paper0_lane_registry_v1`
- Lane count: `3`
- QPU submission enabled: `False`
- Hardware submission enabled: `False`

## Lanes

| Lane ID | Evidence class | Source span | Blockers | Related spec |
|---|---|---|---|---|
| `P0L01_front_matter_context` | `source_boundary_context` | `P0R00018 to P0R00104` | no_external_validation_submission, downstream_boundary_projection | docs/internal/paper0_foundational_extraction/paper0_front_matter_context_validation_specs_2026-05-13.json |
| `P0L02_axiom_ii_infoton_geometry` | `source_derivation_slice` | `P0R00770 to P0R00774` | downstream_validation_requires_new_hardware, empirical_evidence_not_claimed | `none` |
| `P0L03_foundational_viability_postulate` | `source_postulate_bundle` | `P0R00464 to P0R00505` | no_hardware_submission_path | `none` |

## Reproducibility gate

Regenerate and compare this registry with:

```bash
scpn-bench paper0-lane-registry-gate
```

## Claim boundary

Paper 0 lane registry is source-bounded; it does not claim external empirical validation and has no QPU submission and no hardware execution path.
