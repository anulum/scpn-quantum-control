# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Phase 2 B-C scaling dataset

# Phase 2 B-C Scaling Dataset

This directory contains the raw-count IBM hardware output for the preregistered
Phase 2 B-C scaling run on `ibm_kingston`, executed on 2026-05-05.

## Files

| File | SHA-256 | Contents |
|---|---|---|
| `phase2_scaling_bc_2026-05-05T124722Z.json` | `f9718c3789329dbaa96a1667f8a581e3d1774632b961a1760c044138ccab6550` | 280 raw-count B-C circuit records with metadata, job identifier, live-transpilation summary, and per-circuit parity statistics. |
| `phase2_scaling_bc_summary_2026-05-05.json` | `9aa50a434c81fc1a38f1e2f887425808bae3ed2dd37725f4f381515df580e0d8` | Reproduced count-to-statistic summary emitted by `scripts/analyse_phase2_scaling_bc.py`. |

## IBM job

| Job ID | Backend | Circuits | Shots | IBM reported quantum seconds |
|---|---|---:|---:|---:|
| `d7sudr2udops7397ae30` | `ibm_kingston` | 280 | 4096 | 305 |

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/analyse_phase2_scaling_bc.py data/phase2_scaling_bc/phase2_scaling_bc_2026-05-05T124722Z.json --sha256 f9718c3789329dbaa96a1667f8a581e3d1774632b961a1760c044138ccab6550
```

Expected high-level output:

- `n=6`: Fisher chi2 `46.531552`, p `1.883218e-07`, 2/4 significant depths.
- `n=8`: Fisher chi2 `29.420107`, p `2.675193e-04`, 3/4 significant depths.

## Interpretation scope

This dataset is mixed scaling evidence. It supports a positive `n=8`
middle-depth sign, but `n=6` has negative significant asymmetry at depths 8 and
20. Therefore it falsifies a simple monotone scaling story and should not be
summarised as broad scaling validation.

It does not support D-E larger scaling, GUESS mitigation, broad quantum
advantage, frontier, multi-QPU, or live-loop claims.
