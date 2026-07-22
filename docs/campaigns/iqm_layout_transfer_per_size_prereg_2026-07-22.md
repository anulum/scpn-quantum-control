<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM Layout-Transfer Per-Size Preregistration -->

# IQM Garnet Layout-Transfer Per-Size Powered Preregistration (FU-3)

Date: 2026-07-22

This preregistration commits the design, decision rules, and budget for the
per-size powered follow-up to the executed layout-transfer campaign
(`iqm_layout_transfer_square_lattice_prereg_2026-07-21.md`, executed
2026-07-21 with Amendment 1). It is committed and pushed BEFORE any of its
own hardware data exist. Execution requires a separate explicit owner GO.

## Motivating (already-collected) evidence

The 2026-07-21 layout-transfer campaign FAILED its frozen all-three-sizes
primary (`wins_all_sizes_optimised_vs_default = false`): the pooled
default-minus-optimised corrected-error difference was positive (+0.0345,
bootstrap CI90 [+0.026, +0.043] — the optimiser helps on average) but the
per-size profile was non-uniform: n = 8 (0.147 vs 0.253) and n = 16
(0.292 vs 0.314) favoured the optimiser while n = 12 REVERSED (optimised
error 0.209 vs default 0.185) at 2,048 shots per arm. The default arm beat
the naive arm at every size (+0.0989 pooled). Two open questions:

> Q1: Is the n = 12 reversal a real per-size disadvantage of the
> calibration-aware optimiser against the default transpiler placement,
> or shot noise at the executed per-arm budget?
> Q2: Is the optimised-vs-default effect profile heterogeneous across
> sizes beyond shot noise?

The 2026-07-21 data are design input here and are never re-used as
confirmatory evidence for this campaign's endpoints.

## Circuit Matrix

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| Circuit family | committed `iqm_layout_transfer_benchmark` builders (TROTTER_DEPTH 5, transpiler seed 20260721, IQM basis `r`/`cz`) — identical to the executed campaign |
| Sizes | `8, 12, 16` |
| Arms | `optimised` (best_chain_region → optimise_kuramoto_layout), `default` (transpiler placement), `naive` (Amendment 1: lexicographically smallest connected chain) — all three recomputed on the submission day's calibration snapshot and recorded BEFORE submission |
| Repetitions | `4` per arm/size (execution-order replicates) |
| Main shots | `2048` per repetition → `8192` per arm/size (4× the executed campaign) |
| Readout | per size: all-zeros / all-ones over the union of the arms' measured qubits at `1024` shots (exact tensored per-qubit correction, unchanged) |
| Observable | corrected order-parameter error `E = |R̂_corrected − 0.5|` per arm/size, fail-closed denominator — identical to the executed campaign |

Circuit count: `3 × 3 × 4 = 36` main + `3 × 2 = 6` readout = `42` circuits.
Shot count: `73,728` main + `6,144` readout = `79,872` shots.

**Batching disclosure (frozen):** the whole matrix submits in one pass —
mains batched into one job (or the minimum number the batch limit allows)
and readout into one more. All arms of every size ride in the SAME batch,
so window-level drift (the FU-1 finding, quantified by FU-W) cancels in
the per-size ARM DIFFERENCES; the campaign makes no cross-window claim.

## Endpoints and Decision Rules (frozen)

Per size `n`, pool the 4 repetitions per arm into the corrected errors
`E_opt(n)`, `E_def(n)`, `E_naive(n)` and form the primary difference
`D(n) = E_def(n) − E_opt(n)` (positive = optimiser advantage, the executed
campaign's sign convention). All intervals and tests use the committed
bootstrap machinery (10,000 resamples, seed 20260722).

- **Primary (per-size resolution):** two-sided bootstrap test of
  `D(n) ≠ 0` at every size, Holm–Bonferroni adjusted across the three
  sizes, α = 0.05. Power: the executed pooled bootstrap CI90 gives a
  per-size SE ≈ 0.0090 at 2,048 shots per arm, hence ≈ 0.0045 at 8,192;
  the minimum detectable per-size |D| is ≈ 0.017 at 90 % power under the
  worst Holm threshold. The executed n = 12 reversal (−0.024) and the
  executed n = 8 advantage (+0.106) both exceed this.
- **Secondary S1 (n = 12 anomaly):** sign and bootstrap CI95 of `D(12)`,
  reported regardless of the primary outcome — the direct resolution of
  the executed reversal.
- **Secondary S2 (pooled replication):** pooled `D` across sizes with
  bootstrap CI90, compared descriptively against the executed
  +0.0345 [+0.026, +0.043].
- **Secondary S3 (heterogeneity):** inverse-variance Cochran's Q across
  the three per-size `D(n)` (bootstrap variances), α = 0.05 — the formal
  test of Q2.
- **Secondary S4 (naive reference):** per-size `E_def(n) − E_naive(n)`
  with bootstrap CI95, descriptively against the executed uniform
  default-over-naive result.
- **Secondary S5 (correction sensitivity):** raw vs corrected `D(n)` side
  by side (the correction is exact for the observable; divergence flags a
  readout-model violation and is reported, never silently fixed).
- Depth-parity gate (multiplicative, `max ≤ min·(1 + tolerance)` per size
  across arms, the committed `depth_parity_gate`) enforced at dry run AND
  at submit exactly as executed; a violation blocks submission (no
  post-hoc arm handicap).

Every outcome branch is publishable: a Holm-surviving negative `D(12)`
establishes a real per-size optimiser disadvantage; a positive or null
`D(12)` bounds the executed reversal as shot noise; S3 settles whether
layout transfer is uniform or size-modulated.

## Live Readiness Gates (block submission until all pass)

- `IQMFakeGarnet` dry run of all 42 circuits from committed code only,
  with all arms' layouts computed from the day's calibration snapshot and
  the per-size depth-parity gate green;
- `garnet:mock` zero-spend integration submit;
- calibration check on the day's live snapshot for all arms' chains
  (every chain edge calibrated), layouts recorded before submission;
- explicit owner GO immediately before submission.

## Budget and Stop Rules

- Grant context: 500 IQM Resonance credits (expiry 2026-11-22).
- Expected cost: ~2–3 jobs ≈ 2–3 credits (single-pass batching).
- Abort if the submission consumes more than 6 credits on the owner's
  dashboard reading.
- No submissions beyond this 42-circuit matrix without a fresh
  preregistration; a second-window replication (if ever wanted) is a NEW
  preregistration, not an extension.

## Claim Boundary

Blocked regardless of outcome: quantum advantage; coherence protection;
claims about non-sampled sizes, devices, or calibration windows; any
modification of frozen submissions under `paper/submissions/` (results
extend the NEW manuscript `submission_006` or its successor only). IQM
and IQM Resonance are credited in every resulting output.

## Submission Boundary

This preregistration is complete once committed and pushed. QPU execution
stays blocked until the readiness gates pass and the owner grants a
separate explicit GO.
