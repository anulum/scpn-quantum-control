<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S2 slice progress report -->

# S2 Slice Progress Report

Date: 2026-05-07

## Decision

- Progress decision: `ready_for_next_bounded_no_qpu_slice`
- Slice count: `3`
- Sizes: `[8, 10, 12]`
- Total rows: `15/15` ok
- Hardware submission: `False`
- Advantage claim: `False`
- Full campaign complete: `False`

## Aggregate artefact

- JSON report: `data/s2_advantage_scaling/s2_slice_progress_report_2026-05-07.json`

## Slice summary

| n | rows ok/executed | total wall ms | max memory bytes | slowest baseline |
|---:|---:|---:|---:|---|
| 8 | 5/5 | 2889.020 | 8935750 | classical_ode |
| 10 | 5/5 | 17356.462 | 142658102 | mps_tensor_network |
| 12 | 5/5 | 490662.466 | 2416024559 | mps_tensor_network |

## Boundary

This report aggregates completed bounded no-QPU S2 slices only. It does
not establish hardware scaling, full campaign completion, or quantum advantage.
The next expansion should remain deliberate because the
`n=12` slice already makes the dense and tensor-network rows expensive.
