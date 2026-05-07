<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S2 full-campaign slice -->

# S2 Full Scaling Campaign Slice

Date: 2026-05-07

## Decision

- Slice decision: `completed_no_qpu_campaign_slice`
- Hardware submission: `False`
- Advantage claim: `False`
- Full campaign complete: `False`
- Executed rows: `5`

## Artefacts

- JSON summary: `data/s2_advantage_scaling/s2_full_campaign_slice_n14_2026-05-07.json`
- Executed rows: `data/s2_advantage_scaling/s2_full_campaign_slice_rows_n14_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/run_s2_full_campaign_slice.py
```

## Boundary

This is a bounded no-QPU execution slice. It is not the full S2
campaign, not hardware evidence, and not a quantum-advantage claim.
