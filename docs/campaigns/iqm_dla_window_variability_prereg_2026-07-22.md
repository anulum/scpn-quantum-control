<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM DLA Window-Variability Preregistration -->

# IQM Garnet Parity-Asymmetry Window-Variability Preregistration (FU-W)

Date: 2026-07-22

This preregistration commits the design, decision rules, and budget for the
window-variability follow-up to the executed d10 sign-replication campaign
(`iqm_dla_d10_retest_prereg_2026-07-22.md`). It is committed and pushed
BEFORE any of its own hardware data exist (per the standing rule that the
freeze must be git-provable). Execution requires a separate explicit owner
GO per window submission.

## Motivating (already-collected) evidence

The 2026-07-22 d10 sign-replication campaign found that the negative
depth-10 asymmetry observed on 2026-07-21 does NOT replicate in a fresh
execution window: the even-minus-odd leakage difference moved from
−0.0237 (2026-07-21 powered block) to +0.0133 (2026-07-22 retest), a
cross-window shift of 0.0370 with z = 2.74, two-sided p = 0.006, on the
same pinned layout and the same calibration set. The depth-10 anomaly is
therefore attributed to window-to-window drift, not to a stable
depth-dependent sign change.

That finding converts drift from a confound into the object of study:

> Q1: Does the parity-asymmetry difference drift between execution windows
> by more than binomial shot noise, at each depth?
> Q2: How large is the between-window standard deviation τ per depth, and
> is the shallow-depth (d4) strong positive asymmetry stable in sign
> across windows while deeper values fluctuate?

The 2026-07-21/22 data are design input here and are never re-used as
confirmatory evidence for this campaign's endpoints; every window analysed
below is newly collected.

## Circuit Matrix (per window)

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| `n` | `4` |
| Circuit family | committed `iqm_dla_pinned_n4_d{4,8,10,12}_{even,odd}` builders (same code path as all prior campaigns) |
| States | `0011` even, `0001` odd |
| Depths | `4, 8, 10, 12` |
| Repetitions | `4` per state/depth (execution-order replicates inside the batch) |
| Main shots | `1024` per repetition → `4096` per state/depth per window |
| Readout states | `0011`, `0001`, `0000`, `1111` at `2048` shots, run EVERY window (fresh per-window readout snapshot is part of the design) |
| Layout | `[2, 7, 12, 13]` pinned (identical to all 2026-07 campaigns); fallback `[9, 4, 3, 8]` recorded before submission if unavailable |

Per window: `4 × 2 × 4 = 32` main + `4` readout = `36` circuits,
`32,768` main + `8,192` readout = `40,960` shots.

**Batching disclosure (frozen):** each window submits in one pass — all 32
mains batch into one job and the 4 readout states into a second job
(`batch_all`, the d10-retest pattern). Within a window the repetitions are
execution-order replicates, NOT independent windows; the independent unit
of this campaign is the WINDOW. Expected cost ≈ 2 jobs ≈ 2 credits per
window at the observed ~1 credit/job billing.

## Windows (frozen definition)

- A window is one full 36-circuit pass submitted after an explicit owner
  GO for that window.
- Consecutive windows are separated by ≥ 12 h wall clock.
- Target `W = 10` windows; the campaign is analysable and publishable from
  `W = 6` onward (the test degrees of freedom adapt to the achieved `W`;
  fewer than 6 windows → report as incomplete, no confirmatory claim).
- Windows MAY span calibration-set changes; the active calibration set and
  a live calibration snapshot (`dump-calibration`) are recorded per window
  and used as descriptive covariates only.

## Endpoints and Decision Rules (frozen)

Per window `w` and depth `d`, the analysis pools the 4 repetitions into
`leak_even(d, w)` and `leak_odd(d, w)` (4,096 shots per arm) and forms the
difference `Δ(d, w) = leak_even − leak_odd` with binomial variance
`se²(d, w) = p_e(1−p_e)/4096 + p_o(1−p_o)/4096`.

- **Primary (depth-10 heterogeneity):** Cochran's Q homogeneity test of
  `Δ(10, w)` across the executed windows against the shot-noise-only null
  (`Q = Σ_w (Δ_w − Δ̄)² / se²_w` with the inverse-variance weighted mean,
  `Q ~ χ²(W−1)` under H0), α = 0.05. Rejection ⇒ between-window drift at
  depth 10 exceeds binomial shot noise. Failure to reject ⇒ bounded null:
  at `W = 10` and the anticipated `se ≈ 0.011` per window arm pair, the
  design detects a between-window standard deviation `τ ≥ 0.019` with
  ≈ 90 % power (scaled-χ² approximation); the 2026-07-21→22 shift (0.0370
  between two windows, implying `τ ≈ 0.026` under a two-draw model) is
  detected with > 95 % power.
- **Secondary S1 (per-depth heterogeneity):** the same Q test at depths 4,
  8, and 12, Holm–Bonferroni adjusted across the three secondary depths.
- **Secondary S2 (drift magnitude profile):** DerSimonian–Laird
  method-of-moments point estimate `τ̂(d)` per depth, reported
  descriptively (no CI claim) together with the shot-noise `se` so the
  ratio is auditable.
- **Secondary S3 (shallow-anchor sign stability):** fraction of windows
  with `Δ(4, w) > 0`, with an exact (Clopper–Pearson) 95 % interval. The
  2026-07-21 d4 asymmetry (+0.230 relative) predicts near-universal
  positivity; instability here would undercut the pooled-direction claim
  of the powered block and MUST be reported prominently if observed.
- **Secondary S4 (calibration covariates, descriptive only):** per-window
  scatter of `Δ(10, w)` against the recorded per-window CZ fidelity and
  readout error on the pinned layout; no test, no claim beyond
  description.
- Per-window drift table (per-repetition rows inside each window) as in
  all prior campaigns.
- Readout handling: four-state exact-state parity-confusion correction per
  window using THAT window's readout block, identical method to the
  powered block; raw-count endpoints remain primary exactly as before.

Either primary outcome is publishable: heterogeneity confirmed quantifies
drift as a real noise floor for single-window depth-profile claims (and
retroactively explains the d10 anomaly); the bounded null shows
single-window profiles are shot-noise limited at the committed power. No
coherent-dynamics claim is available in any branch (exact statevector
baseline pins noiseless leakage at zero for every depth; committed module
`analysis/dla_parity_exact_baseline.py`).

## Live Readiness Gates (block each submission until all pass)

- `IQMFakeGarnet` dry run of the full 36-circuit window matrix from
  committed code only (once, before window 1);
- transpiled depth envelope: the executed/interpolated ladder d4→69,
  d8→129, d10→159, d12→189 with the standing +25 % rule, i.e. d4 ≤ 86,
  d8 ≤ 161, d10 ≤ 198, d12 ≤ 236, enforced at every submit;
- `garnet:mock` zero-spend integration submit (once, before window 1);
- pinned-layout calibration check on the day's live snapshot before EVERY
  window (all three chain edges calibrated), fallback substitution
  recorded before submission if required;
- explicit owner GO immediately before EVERY window submission.

## Budget and Stop Rules

- Grant context: 500 IQM Resonance credits granted 2026-07-22, expiry
  2026-11-22 (owner-verified from the dashboard; balances are never
  available in the client API).
- Expected total: ~2 credits × 10 windows ≈ 20 credits.
- Submit window 1 alone first; abort the campaign if window 1 consumes
  more than 5 credits on the owner's dashboard reading.
- Hard cap: abort if cumulative campaign spend exceeds 35 credits.
- No submissions beyond the preregistered per-window 36-circuit matrix and
  the `W = 10` target without a fresh preregistration.

## Claim Boundary

Blocked regardless of outcome: quantum advantage; coherence protection;
coherent-dynamics parity claims; any causal attribution of drift to a
specific hardware mechanism (S4 is descriptive); extrapolation beyond the
sampled device, layouts, and depths; any modification of frozen
submissions under `paper/submissions/` (results extend the NEW manuscript
`submission_006` or its successor only). IQM and IQM Resonance are
credited in every resulting output.

## Submission Boundary

This preregistration is complete once committed and pushed. QPU execution
stays blocked until the readiness gates pass and the owner grants a
separate explicit GO for each window.
