<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S2 full-campaign plan -->

# S2 Full Scaling Campaign Plan

Date: 2026-05-07

## Decision

- Campaign decision: `ready_for_deliberate_no_qpu_full_classical_campaign`
- Hardware submission: `False`
- Advantage claim: `False`
- Ready required rows: `32`

## Artefacts

- JSON summary: `data/s2_advantage_scaling/s2_full_campaign_plan_2026-05-07.json`
- Planning rows: `data/s2_advantage_scaling/s2_full_campaign_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/plan_s2_full_scaling_campaign.py
```

## Boundary

This is a no-QPU execution plan. It does not run the heavy
classical/simulator campaign, does not submit hardware jobs, and does
not support quantum-advantage language.
