# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IBM popcount-control manifest

# IBM Popcount-Control Manifest -- 2026-05-05

This manifest preregisters the next QPU run. Its purpose is to test whether the
observed `n=4` DLA parity leakage asymmetry is primarily a parity-sector effect,
primarily an excitation-count / amplitude-damping effect, or a combination of
both.

## Motivation

The promoted Phase 1 and Phase 2 A+G protocols compare:

| Label | State | Parity | Popcount |
|---|---|---:|---:|
| Original even | `0011` | even | 2 |
| Original odd | `0001` | odd | 1 |

This leaves an excitation-count confound. Under a simple `T1` amplitude-damping
model, the higher-excitation state can leak faster independently of the DLA
sector structure.

## Submitted scope

Run only `n=4` controls on the same XY-Trotter circuit family.

Depths:

`d in {4, 6, 8, 10, 14, 20}`

States:

| Arm | State | Parity | Popcount | Purpose |
|---|---|---:|---:|---|
| E0 | `0011` | even | 2 | Original even reference. |
| E1 | `0101` | even | 2 | Within-even state swap at fixed popcount. |
| O0 | `0001` | odd | 1 | Original odd reference. |
| O1 | `0010` | odd | 1 | Within-odd state swap at fixed popcount. |
| O3 | `0111` | odd | 3 | Excitation-inversion arm against E0. |

Repetitions:

`12` reps per `(depth, state)` point.

Shots:

`4096` shots per circuit.

Circuit count:

`6 depths * 5 states * 12 reps = 360` parity circuits.

Readout baseline:

`0011`, `0101`, `0001`, `0010`, `0111` at `8192` shots each.

Total submitted circuits:

`365`

Estimated QPU use:

Using the recent B-C observed rate, `305` quantum seconds for `280` circuits,
this run should cost about `398` quantum seconds, or `6.64` minutes, before
backend variation. Abort if live transpilation substantially exceeds the
precommitted depth budget.

## Abort and budget rule

Before QPU submission, transpile every circuit on the selected live backend.

Abort before submission if:

- live maximum transpiled depth exceeds `700`;
- live maximum gate count exceeds the recent B-C maximum by more than 50%;
- IBM backend selection changes to a non-Heron backend;
- any generated circuit has missing measurement or zero shots;
- the account has less than 20 minutes of remaining QPU time.

## Analysis plan

For each state and depth, compute parity leakage from raw counts:

`L(d, state) = shots with parity != parity(initial) / total shots`

Primary comparisons:

| Comparison | Null / falsification use |
|---|---|
| E0 vs O0 | Reproduce the original parity-sector / popcount-correlated contrast. |
| E0 vs E1 | Estimate within-even same-popcount state spread. |
| O0 vs O1 | Estimate within-odd same-popcount state spread. |
| E0 vs O3 | Excitation-inversion test; pure `T1` expects O3 to leak at least as much as E0. |

Decision criteria:

- If E0 vs O0 remains positive but E0 vs O3 reverses or vanishes, the original
  result is likely dominated by excitation count.
- If E0 vs O0 remains positive and E0 vs O3 also shows lower leakage for the
  odd state despite higher popcount, the DLA-sector interpretation is
  strengthened.
- If E0 vs E1 or O0 vs O1 spreads are comparable to E0 vs O0, the original
  effect is state-preparation/layout sensitive and should be downgraded.
- If all contrasts vanish, treat the prior result as calibration-window
  dependent until multi-day replication is available.

## Promotion rule

Do not promote the run unless all of the following are committed:

- raw count dictionaries;
- IBM job IDs;
- live transpilation depth and gate summaries;
- SHA256 hash of the raw-count JSON;
- reproducer script or existing reproducer update;
- result summary with claim boundary;
- paper update preserving the no-broad-advantage boundary.

## Paper wording before this run is complete

Use conservative wording:

`parity-sector and excitation-number correlated leakage asymmetry`

Do not write:

`DLA parity alone causes the asymmetry`
