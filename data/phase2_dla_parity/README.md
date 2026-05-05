# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 DLA parity reduced A+G dataset

# Phase 2 DLA Parity Reduced A+G Dataset

This directory contains the raw-count IBM hardware output for the reduced Phase
2 DLA parity replication run on `ibm_kingston`, executed on 2026-05-05.

The run was deliberately narrowed to the QPU-minimised A+G plan:

- Block A: high-statistics `n=4` DLA parity replication.
- Block G: same-run readout baseline for `n=4,6,8` basis states.
- Blocks B-F were not submitted.

## Files

| File | SHA-256 | Contents |
|---|---|---|
| `phase2_reduced_ag_2026-05-05T121357Z.json` | `7c5f2a32d5a113d916d84d26d27a69336846364d5ee23ba4621b059125e0f5d5` | 612 raw-count circuit records with metadata, job identifiers, and per-circuit parity statistics. |
| `phase2_reduced_ag_summary_2026-05-05.json` | `0f5d83805f2df9abfcfb979857b9264298fdb6995b41a227cab21c84c61ce39e` | Reproduced count-to-statistic summary emitted by `scripts/analyse_phase2_dla_parity.py --verify-integrity --json`. |

## IBM jobs

| Block | Job ID | Backend | Circuits | Shots | IBM reported quantum seconds |
|---|---|---|---:|---:|---:|
| A main | `d7stu94t738s73ch5keg` | `ibm_kingston` | 600 | 4096 | 660 |
| G readout | `d7su3tkt738s73ch5ql0` | `ibm_kingston` | 12 | 8192 | 27 |

Total IBM-reported usage: 687 quantum seconds.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/analyse_phase2_dla_parity.py --verify-integrity
```

Expected high-level output:

- Fisher chi2: `140.671952`
- Fisher p: `3.773718e-20`
- Significant depths at `p < 0.05`: `6 / 10`
- Complete main counts: 600 circuits at 4096 shots each
- Complete readout counts: 12 circuits at 8192 shots each

## Promotion scope

This dataset supports only the reduced Phase 2 `n=4` DLA parity replication and
readout-control claim. It does not support broad quantum advantage, `n=6-12`
scaling, GUESS mitigation, frontier, multi-QPU, or live-loop claims.
