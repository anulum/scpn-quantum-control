<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1 IBM live readiness -->

# S1 IBM Live Readiness

Date: 2026-05-20

Preregistration date: 2026-05-06

Backend: `ibm_kingston`

Submission state: `live_metadata_and_transpile_no_submission`

Hardware submission: `false`

Capability status: `ready`

Readiness status: `ready_for_pair_runner`

## Transpiled Dynamic-Circuit Payload

| Field | Value |
|---|---:|
| Qubits | 156 |
| Classical bits | 6 |
| Depth | 717 |
| Shots per circuit | 1024 |
| Repetitions | 12 |
| QPU-second ceiling | 120.0 |
| Transpiler seed | 20260520 |

Operation counts:

```json
{
  "cz": 183,
  "if_else": 3,
  "measure": 6,
  "reset": 3,
  "rz": 380,
  "sx": 363,
  "x": 2
}
```

## Pair-Runner Status

No live-readiness blocker remains for the corrected S1 dynamic-circuit
payload. The paired feedback/open-loop runner has superseded this no-submit
probe and completed IBM execution:

- feedback job: `d86qn3lg7okc73elg2eg`
- matched open-loop control job: `d86qn65g7okc73elg2hg`
- result note: `docs/s1_ibm_feedback_pair_result_2026-05-20.md`
- analysis: `data/s1_feedback_loop/s1_feedback_analysis_summary_ibm_kingston_20260520T123941Z.json`

## Claim Boundary

This artefact captures live backend metadata and transpilation only. It does not submit IBM jobs and cannot support an S1 hardware-control claim.
