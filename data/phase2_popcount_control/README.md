# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 2 popcount-control dataset

# Phase 2 Popcount-Control Dataset

Preregistered IBM `ibm_kingston` control run executed on 2026-05-05 to test
the excitation-count confound in the DLA parity leakage experiment.

## Files

| File | Purpose |
|---|---|
| `phase2_popcount_control_2026-05-05T135318Z.json` | Promoted raw-count dataset. |
| `phase2_popcount_control_summary_2026-05-05.json` | Reproduced statistical summary. |

## Hardware jobs

| Block | Job ID | Circuits | Shots |
|---|---|---:|---:|
| Main parity leakage | `ibm-run-7d468e2b1e44b406` | 360 | 4096 |
| Readout baseline | `ibm-run-b3424c38cfe03c86` | 5 | 8192 |

Live transpilation:

- Max depth: `414`
- Mean depth: `212.312329`
- Max gate count: `1086`
- Mean gate count: `536.989041`

## Integrity

Raw JSON SHA256:

`f43cbd7e466a3267847b44a750aeba7801cbc52ef10e9808573ef7ed01ec3cf0`

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python \
  scripts/analyse_phase2_popcount_control.py --verify-integrity
```

## Claim boundary

This dataset tests the excitation-count confound. It should not be used as
broad quantum-advantage evidence, monotone scaling validation, GUESS mitigation
validation, or multi-device replication.
