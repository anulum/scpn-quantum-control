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

Readiness status: `blocked`

## Transpiled Dynamic-Circuit Payload

| Field | Value |
|---|---:|
| Qubits | 156 |
| Classical bits | 6 |
| Depth | 720 |
| Shots per circuit | 1024 |
| Repetitions | 12 |
| QPU-second ceiling | 120.0 |
| Transpiler seed | 20260520 |

Operation counts:

```json
{
  "cz": 183,
  "if_else": 6,
  "measure": 6,
  "rz": 380,
  "sx": 363,
  "x": 2
}
```

## Remaining Blockers

- provider submitter for paired feedback/control S1 arms is not implemented
- live IBM sampler-result to preregistered r_live raw-count package conversion is not implemented
- explicit hardware approval record for this package hash and QPU-second ceiling is not present

## Claim Boundary

This artefact captures live backend metadata and transpilation only. It does not submit IBM jobs and cannot support an S1 hardware-control claim.
