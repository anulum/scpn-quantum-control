# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- HAL status normalisation freeze policy

## HAL Status Normalisation Policy

This document defines the frozen canonical status contract for hardware HAL
adapters. Any alias change is a contract change and must follow the evidence
gate below.

### Canonical status set

- `submitted`
- `queued`
- `running`
- `completed`
- `cancelled`
- `failed`
- `unknown` (fallback only)

### Frozen alias matrix

Two contract tiers are enforced:

- **Extended tier** (Azure, Braket, qBraid, Qiskit, IonQ, Strangeworks, Pasqal,
  OQC, IQM, Quantinuum): full matrix below.
- **Baseline tier** (currently QuEra Bloqade): only completion/queue/running/
  cancellation/failure aliases.

| Canonical | Accepted aliases |
| --- | --- |
| `completed` | `complete`, `completed`, `success`, `succeeded` |
| `running` | `running`, `in_progress`, `in-progress`, `inprogress` |
| `submitted` | `submitted`, `initializing`, `initialising`, `starting`, `creating`, `created` |
| `queued` | `queued`, `pending` |
| `cancelled` | `cancelled`, `canceled`, `aborting`, `cancelling`, `canceling` |
| `failed` | `failed`, `error` |

### Evidence-gated alias extension rule

New aliases are forbidden unless all conditions hold:

1. A real provider token is captured in artifact evidence (`job metadata`,
   provider payload, or execution logs) and linked from the session report.
2. The alias is added to the frozen matrix tests in
   `tests/test_hardware_hal_status_normalisation_contract.py`.
3. Adapter-level tests are updated for impacted backends.
4. Full HAL regression (`tests/test_hardware_hal_*.py`) remains green.
5. This policy file is updated in the same change.

Without these five conditions, reject alias additions.

### Baseline-tier completion extras

Baseline-tier adapters may additionally accept `done` and `finished` as
completion aliases, but those aliases are not part of the extended-tier common
contract.
