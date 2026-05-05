# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Phase 2 B-C scaling manifest, 2026-05-05

# IBM Phase 2 B-C Scaling Manifest — 2026-05-05

This manifest preregisters the next continuation run after the completed reduced
A+G replication. It is limited to scaling blocks B and C only.

## Status

| Field | Value |
|---|---|
| Status | Preregistered; not submitted at document time |
| Backend | `ibm_kingston` |
| Approved blocks | B (`n=6`) and C (`n=8`) |
| Deferred blocks | A, D, E, F, G |
| New readout baseline | none; reuse completed A+G same-day `n=6,8` readout controls |
| Dry-run command | `PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --dry-run --backend ibm_kingston --skip A D E F G` |
| Live command | `PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --confirm-promo-active --backend ibm_kingston --skip A D E F G --max-live-depth 900` |

## Scientific purpose

This run tests whether the DLA parity asymmetry observed and replicated at
`n=4` survives at `n=6` and `n=8`.

It can validate:

- early system-size scaling of the parity-sector asymmetry;
- whether the middle-depth sign remains positive beyond `n=4`;
- whether the depth window shifts or narrows with system size.

It can falsify:

- a simple scalable-parity interpretation if `n=6` and `n=8` lose the sign;
- a monotone scaling story if the effect is present at one size but absent at
  the other;
- any claim that Phase 2 already established `n=6-12` scaling.

## Circuit inventory

| Block | n | Depths | Sectors | Reps | Shots | Circuits |
|---|---:|---|---|---:|---:|---:|
| B | 6 | `4,8,14,20` | even, odd | 20 | 4096 | 160 |
| C | 8 | `4,8,14,20` | even, odd | 15 | 4096 | 120 |
| Total | 6,8 | preregistered above | even, odd | mixed | 4096 | 280 |

Dry-run result on 2026-05-05:

| Metric | Value |
|---|---:|
| Circuits | `280` |
| Simulator ISA depth min | `65` |
| Simulator ISA depth max | `269` |
| Simulator ISA depth mean | `160.1` |
| Simulator gate-count min | `198` |
| Simulator gate-count max | `1396` |
| Simulator gate-count mean | `670.7` |

The live command must pass the pre-submit backend-transpilation gate before any
IBM job is submitted. The live max-depth guard is `900`; if exceeded, the script
must abort without submission.

## Promotion rules

The run is not promoted until all gates below pass:

| Gate | Requirement |
|---|---|
| Raw counts | 280 B-C circuits with 4096 shots each. |
| Job binding | Every circuit is bound to an IBM job identifier. |
| Reproducer | `scripts/analyse_phase2_scaling_bc.py` recomputes all published values from raw counts. |
| Integrity | Raw JSON and summary JSON have SHA-256 hashes recorded in `data/phase2_scaling_bc/README.md`. |
| Ledger | Hardware ledger names the exact artefacts and scope. |
| Scope | Promote only `n=6,8` scaling evidence; no D-E, GUESS, broad advantage, frontier, or multi-QPU claim. |

## Expected interpretation

Positive result: same-sign asymmetry at `n=6` and/or `n=8`, especially at middle
depths, supports early scaling beyond `n=4`.

Null result: constrains the claim to `n=4` replication and marks larger-size
scaling as unestablished.

Mixed result: identifies a size/depth window and should be reported as such.
