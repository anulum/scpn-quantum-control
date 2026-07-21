<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM DLA Depth-Profile Preregistration -->

# IQM Garnet Parity-Asymmetry Depth-Profile Preregistration

Date: 2026-07-22

This preregistration commits the design, decision rule, and budget for the
depth-profile follow-up to the executed powered backend-sensitivity block
(`iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md`, executed
2026-07-21, primary rejected H0 at p = 1.1 × 10⁻⁵). It is committed BEFORE
any of its own hardware data exist. Execution requires a separate explicit
per-submit owner GO.

## Motivating (already-collected) evidence

The 2026-07-21 powered block found the pooled even-over-odd parity-leakage
asymmetry replicates the IBM direction, but with a depth profile unlike
IBM's flat reference: +0.230 at depth 4, +0.083 at depth 6, −0.049 at depth
10 (sign reversed), with both sectors approaching ~0.46–0.48 total leakage at
depth 10 (near sector equilibration). Two open questions follow:

> Q1: Where between depths 6 and 10 does the Garnet asymmetry cross zero?
> Q2: Is the decay consistent with sector equilibration (total leakage
> saturating toward ~0.5 compresses the asymmetry), or does the odd sector
> genuinely overtake the even sector at depth?

Those depths (8, 12) were NOT part of the executed matrix; this campaign
collects them. The 2026-07-21 data are design input here and are never
re-used as confirmatory evidence for this campaign's endpoints.

## Circuit Matrix

| Field | Value |
|-------|-------|
| Device | IQM Garnet (20 qubits, square lattice) via Resonance |
| `n` | `4` |
| Circuit family | committed `iqm_dla_pinned_n4_d{8,12}_{even,odd}` builders (same code path as d4/6/10) |
| States | `0011` even, `0001` odd |
| Depths | `8, 12` |
| Repetitions | `4` per state/depth |
| Main shots | `1024` per repetition → `4096` per state/depth |
| Readout states | `0011`, `0001`, `0000`, `1111` at `2048` shots (fresh block — new calibration window) |
| Layout | `[2, 7, 12, 13]` pinned (same as the powered block); fallback `[9, 4, 3, 8]` recorded before submission if unavailable |

Circuit count: `2 × 2 × 4 = 16` main + `4` readout = `20` circuits.
Shot count: `16,384` main + `8,192` readout = `24,576` shots.

## Endpoints and Decision Rules (frozen)

Metric: relative asymmetry `(leak_even − leak_odd) / leak_odd` per depth,
with per-depth one-sided two-proportion z-tests (4,096 shots per arm per
depth; minimum detectable relative asymmetry ≈ ±0.076 at 90 % power for a
baseline odd-arm leak near 0.42 — the committed Lane 1 secondary power).

- **Primary (decay ordering):** one-sided two-proportion comparison that the
  even-minus-odd leakage difference at depth 8 exceeds that at depth 12
  (`Δ₈ > Δ₁₂`, α = 0.05, z-test on the difference of differences with
  binomial variances). This tests the monotone-decay reading of the executed
  profile on entirely new data.
- **Secondary S1 (zero-crossing localisation):** per-depth signs with Wilson
  95 % intervals; the crossing is reported as the interval between the last
  significantly positive and first significantly non-positive depth across
  the JOINT profile {4, 6, 8, 10, 12}, with the cross-calibration-window
  caveat stated wherever the joint profile is shown (d4/6/10 are from
  2026-07-21; d8/12 from this campaign).
- **Secondary S2 (equilibration check):** total leakage `(leak_even +
  leak_odd)/2` per depth with Wilson intervals — reported descriptively
  against the saturation-toward-0.5 explanation.
- Repetition drift table as in the powered block.
- Readout handling: four-state exact-state parity-confusion correction, no
  full-matrix claim (identical to the powered block).

Either primary outcome is publishable: `Δ₈ > Δ₁₂` supports monotone decay;
failure to reject reports the bounded null at the committed power. No
coherent-dynamics claim is available in any branch (the exact statevector
baseline pins noiseless leakage at zero for every depth; committed module
`analysis/dla_parity_exact_baseline.py`).

## Live Readiness Gates (block submission until all pass)

- `IQMFakeGarnet` dry run of all 20 circuits from committed code only;
- transpiled depth envelope: linear interpolation/extrapolation of the
  executed ladder (d4→69, d6→99, d10→159, i.e. ≈ 15 layers per Trotter step)
  predicts ≈ 129 at d8 and ≈ 189 at d12; the frozen gate is the same
  +25 % rule applied to the prediction, so d8 ≤ 161 and d12 ≤ 236;
- pinned-layout calibration check on the day's live snapshot (all three
  chain edges calibrated), fallback substitution recorded before submission
  if required;
- explicit owner per-submit GO immediately before each block.

## Budget and Stop Rules

- Expected cost by the observed 2026-07-21 batching pattern (~1 credit per
  job; 4 mains batch to one job per repetition, readout to one job): ~5
  jobs ≈ 5 credits. Balance before submission read from the Resonance
  dashboard by the owner.
- Submit repetition 1 (4 mains + 4 readout) alone first; abort the campaign
  if that block consumes more than one quarter of the visible allowance.
- No submissions beyond this 20-circuit matrix without a fresh
  preregistration.

## Claim Boundary

Blocked regardless of outcome: quantum advantage; coherence protection;
coherent-dynamics parity claims; extrapolation beyond the sampled device,
calibration window, layout, and depths; any modification of frozen
submissions under `paper/submissions/` (results extend the NEW manuscript
`submission_006` or its successor only). IQM and IQM Resonance are credited
in every resulting output.

## Submission Boundary

This preregistration is complete once committed. QPU execution stays blocked
until the readiness gates pass and the owner grants a separate explicit
per-submit GO.
