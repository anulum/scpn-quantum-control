# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM Phase 2 preregistered run manifest, 2026-05-05

# IBM Phase 2 Preregistered Run Manifest — 2026-05-05

This manifest is the pre-submission contract for the next IBM hardware run.
It is intentionally narrow: it permits a Phase 2 DLA parity expansion only,
not a broad quantum-advantage claim and not any frontier or live-loop claim.

## Submission status

| Field | Value |
|---|---|
| Status | Preregistered; not submitted |
| Prepared date | 2026-05-05 |
| Target script | `scripts/phase2_full_campaign_ibm.py` |
| Primary backend | `ibm_kingston` |
| Replication backend | `ibm_marrakesh` only if credit window permits |
| Available QPU budget assumption | Approximately 120 minutes remaining |
| Required account state | IBM promotional or credit window active |
| Submission guard | `--confirm-promo-active` plus explicit human approval |
| Live-ready reduced command | `PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --confirm-promo-active --backend ibm_kingston --skip B C D E F --max-live-depth 1100` |
| Dry-run command | `PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --dry-run --backend ibm_kingston --skip B C D E F` |
| Dry-run result | Passed on 2026-05-05 |
| 2026-05-05 live attempt | Aborted and cancelled after hardware transpilation drift |
| Cancelled job | `ibm-run-ca8b9612732b84dc`, IBM status `CANCELLED`, reported usage `0` quantum seconds |

No IBM job may be submitted from this manifest until the promotional or credit
window is live and the exact command is approved immediately before execution.

## Scientific objective

Primary objective: test whether the Phase 1 DLA parity asymmetry observed at
`n=4` on `ibm_kingston` reproduces with higher statistics and extends across
larger system sizes under the same raw-count evidence standard.

The promotable claim, if the run succeeds, is limited to:

- DLA parity leakage asymmetry as a function of Trotter depth and system size.
- Backend-specific Heron r2 hardware evidence, not device-independent proof.
- Raw-count-backed statistics only.

The manifest does not permit claims about broad quantum advantage, frontier
scaling, multi-QPU validation, or live-loop IBM execution.

## Minimum live circuit plan

The first live run should spend only the QPU time needed to validate the
central claim: high-statistics `n=4` replication plus a same-run readout
baseline. Scaling and mitigation sweeps are deferred until the primary
raw-count dataset is retrieved and reviewed.

| Block | Experiment | n | Depths / preps | Sectors | Reps | Shots | Circuits | Live status |
|---|---|---:|---|---|---:|---:|---:|---|
| A | High-statistics DLA parity | 4 | `2,4,6,8,10,14,20,30,40,50` | even, odd | 30 | 4096 | 600 | approved for the first live command |
| G | Readout baseline | 4, 6, 8 | `000..`, `111..`, alternating states | baseline | 1 | 8192 | 12 | approved for the first live command |
| Total | Reduced primary run | 4, 6, 8 | preregistered above | mixed | mixed | mixed | 612 | QPU-minimised validation run |

The reduced live script estimate is approximately `5.6` minutes at `0.55`
seconds per circuit. The estimate is not a guarantee; it is a pre-run budget
guard. Wall-clock, queue, and backend scheduling behaviour are not promoted
evidence.

## Deferred circuit plan

The full campaign inventory was dry-run once and remains documented, but blocks
B-F are not approved for the first live run. They require separate approval
after block A+G counts are retrieved and reviewed.

| Block | Experiment | n | Depths / preps | Sectors | Reps | Shots | Circuits |
|---|---|---:|---|---|---:|---:|---:|
| B | Scaling | 6 | `4,8,14,20` | even, odd | 20 | 4096 | 160 |
| C | Scaling | 8 | `4,8,14,20` | even, odd | 15 | 4096 | 120 |
| D | Scaling | 10 | `4,8,14` | even, odd | 12 | 4096 | 72 |
| E | Scaling | 12 | `4,8,14` | even, odd | 8 | 4096 | 48 |
| F | GUESS calibration | 4 | `4,8,14`; noise scales `1,3,5` | even, odd | 10 | 4096 | 180 |
| Deferred total | Follow-up only | 4–12 | preregistered above | mixed | mixed | mixed | 580 |

The full campaign inventory is `1192` circuits. It must not be submitted as the
first live command under the current 120-minute remaining-budget constraint.

## Dry-run transpilation gate

Reduced dry-run timestamp: `2026-05-05T120231Z`.

| Metric | Value |
|---|---:|
| Circuit count | `612` |
| Estimated QPU cost at 0.55 s/circuit | `5.6` minutes |
| ISA depth minimum | `1` |
| ISA depth maximum | `605` |
| ISA depth mean | `221.4` |
| Gate-count minimum | `4` |
| Gate-count maximum | `1503` |
| Gate-count mean | `544.3` |

Decision: the dry-run transpilation budget gate passed. It authorises
readiness only; it does not authorise QPU submission.

## Aborted live attempt — 2026-05-05

A first live reduced A+G attempt was started on 2026-05-05 after IBM readiness
approval. The live hardware transpilation path produced deeper circuits than
the reduced simulator dry-run manifest:

| Source | Max depth |
|---|---:|
| Reduced dry-run manifest | `605` |
| Live hardware transpilation log | `1014` |

This triggered the manifest drift abort rule. The submitted main-batch job
`ibm-run-ca8b9612732b84dc` was immediately cancelled.

IBM job metadata after cancellation:

| Field | Value |
|---|---:|
| Status | `CANCELLED` |
| Created UTC | `2026-05-05T12:06:37.041668Z` |
| Running UTC | `2026-05-05T12:06:38.800236Z` |
| Finished UTC | `2026-05-05T12:07:19.703948Z` |
| Reported quantum seconds | `0` |
| Reported usage seconds | `0` |

Decision: the cancelled job is quarantined execution evidence only. It is not
raw-count evidence, not validation, and not promotable.

## Abort criteria

Abort before submission if any condition below holds:

| Gate | Abort condition |
|---|---|
| Credit window | Promotional or credit allocation is not live. |
| Backend | Target backend is unavailable, paused, or replaced by an unreviewed backend. |
| Script drift | `scripts/phase2_full_campaign_ibm.py` differs from the committed manifest assumptions. |
| Dry-run | Dry-run fails, or max ISA depth or circuit count changes without manifest update. |
| Evidence path | Raw-count output directory, job log, and post-run review path are not writable. |
| Approval | Human approval is absent in the current turn. |

Abort after submission if any condition below holds:

| Gate | Abort condition |
|---|---|
| Empty counts | Any circuit returns zero total counts. |
| Missing job identifier | Any retrieved count set cannot be bound to an IBM job identifier. |
| Partial run | A batch fails and the partial output cannot be isolated from promoted evidence. |
| Backend mismatch | Result metadata names a backend other than the approved backend. |
| Readout baseline | Readout baseline is missing or cannot be matched to the run. |
| Hardware transpilation drift | Live hardware transpilation exceeds the manifest budget before usable counts are retrieved. |

## Evidence capture path

The submission script writes live results under:

- `<private-local-record>`
- `<private-local-record>`
- `results/ibm_runs/` through `HardwareRunner`

Before any public promotion, the raw counts must be copied into a reviewed,
tracked data directory with a manifest equivalent to the Phase 1 dataset:

- `data/phase2_dla_parity/{run_files}.json`
- `data/phase2_dla_parity/README.md`
- a SHA-256 integrity table
- a reproducer that recomputes every promoted statistic from raw counts
- a hardware-ledger row naming the exact artefacts

`private local workspace/` output is execution evidence for review, not public proof by
itself.

## Statistical tests

The post-run reproducer must compute, from raw counts only:

| Statistic | Rule |
|---|---|
| Parity leakage | Opposite-parity counts divided by total counts per circuit. |
| Per-point mean and SEM | Grouped by `(n, depth, sector, noise_scale)` as applicable. |
| Per-depth asymmetry | `(leak_even - leak_odd) / leak_odd` with uncertainty propagation. |
| Welch tests | Even versus odd leakage for every preregistered depth and system size. |
| Combined evidence | Fisher combined statistic per system size and for the preregistered primary block. |
| Readout baseline | State-retention table for every baseline prep; used as a systematic bound. |
| GUESS calibration | Noise-scale trend for block F only; never mixed silently with unmitigated rows. |

Promotion thresholds:

- Primary `n=4` block must reproduce the Phase 1 sign at the preregistered
  middle-depth region.
- The combined `n=4` Fisher test must be significant at `p < 0.001`.
- Larger-`n` blocks are exploratory unless their raw-count reproducer and
  multiplicity handling are reviewed separately.
- Negative or null results must remain in the ledger.

## Submission commands

Reduced dry-run, already passed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --dry-run --backend ibm_kingston --skip B C D E F
```

Reduced live submission now includes a pre-submit hardware-backend transpilation
check. The accepted live depth budget is `1100`, above the observed `1014`
depth from the cancelled attempt and still below a manifest-controlled guard:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --confirm-promo-active --backend ibm_kingston --skip B C D E F --max-live-depth 1100
```

Deferred scaling/GUESS follow-up, blocked until the reduced primary run is
retrieved and reviewed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/phase2_full_campaign_ibm.py --confirm-promo-active --backend ibm_kingston --skip A G
```

## Current decision

Prepared for IBM run approval gate: yes.

Prepared to submit IBM jobs without a live credit window and explicit current
approval: no.

Prepared to promote new hardware validation immediately after submission: no.
Promotion requires raw-count retrieval, integrity review, a committed
reproducer, and an updated hardware ledger row.
