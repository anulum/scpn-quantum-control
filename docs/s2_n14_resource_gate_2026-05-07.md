<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S2 n=14 resource gate -->

# S2 n=14 Resource Gate

Date: 2026-05-07

## Decision

- Resource gate decision: `blocked_for_scheduled_or_offloaded_no_qpu_run`
- Hardware submission: `False`
- Advantage claim: `False`
- Full campaign complete: `False`
- Interactive n=14 promotion: `False`

## Resource comparison

- n=14 estimated dense matrix bytes: `4294967296`
- n=14 estimated statevector bytes: `262144`
- Prior n=8..12 max recorded memory bytes: `2416024559`
- Dense/prior-memory ratio: `1.7777`

## Artefact

- JSON report: `data/s2_advantage_scaling/s2_n14_resource_gate_2026-05-07.json`

## Boundary

This is a resource-gate report, not an n=14 execution result. It does
not establish hardware scaling, full S2 completion, or quantum advantage.

## Recommended next step

Run n=14 only as a scheduled/offloaded no-QPU job, or explicitly label
a capped run with skipped dense/TN rows as a scout rather than a completed
full slice.
