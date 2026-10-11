<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM Garnet preregistered campaigns manuscript -->

# The Calibration Epoch as the Unit of Evidence: Preregistered Parity-Leakage and Layout-Transfer Experiments on IQM Garnet

**Status: DRAFT — not submitted. Publication requires the owner's direct go.**

Miroslav Šotek (ORCID 0009-0009-3560-0851), with Anulum Fortis & Arcane Sapience
(protoscience@anulum.li)

All quantum-hardware results in this work were obtained on IQM Garnet through
**IQM Resonance**; we thank IQM for extended Resonance access. The two core
experiments ran on 20 qubits of the square-lattice device on 2026-07-21
(calibration set `246a4930-54e3-4cd9-a2d1-fcc0919675f5`); explicitly identified
follow-ups used fresh windows and their own recorded calibration sets. The
window-variability campaign (Section 6) ran in ten windows between 2026-07-22
and 2026-09-04, and the depth-ordering confirmation (Section 7) in twelve
calibration epochs between 2026-09-23 and 2026-10-06, each under its own
recorded calibration set.

## Abstract

We report five preregistered experiments on the IQM Garnet 20-qubit
square-lattice processor, each with its decision rule committed before any
hardware data existed and each reported regardless of outcome. First, the
even-over-odd parity-sector leakage asymmetry previously measured on IBM Heron
r2 replicates on Garnet at preregistered power (relative asymmetry +0.068,
one-sided p = 1.1 × 10⁻⁵). Because the noiseless parity leakage of these
number-conserving XY-Trotter circuits is exactly zero, this is a statement
about device noise. Second, the asymmetry is not stable: over ten execution
windows it varies beyond shot noise at fixed depth (Cochran Q = 19.01 on 9
degrees of freedom, p = 0.025), more strongly when windows are pooled by
provider calibration set (p = 0.0083), and an isolated sign reversal at depth
10 does not replicate. Third, a confirmatory campaign that takes the
calibration epoch as its independent unit — twelve epochs, 688,128 shots —
finds the asymmetry larger at depth 8 than at depth 12 (mean contrast +0.0173,
one-sided p = 0.026); the lower bound is +0.0030 and three of twelve epochs
have the opposite sign. Fourth, a calibration-aware layout optimiser developed
on heavy-hex devices fails its frozen transfer rule: it is better than
automatic placement at chain lengths 8 and 16 and worse at 12, and a powered
follow-up confirms the reversal. All preregistrations, calibration snapshots,
job identifiers, raw counts and analysis scripts are public.

## 1. Motivation and prior state

Two threads of our published programme meet on a square lattice.

First, the **layout thread**: our fidelity-aware placement stack — a
community-detection analysis pass, a Kuramoto-XY-aware discrete layout cost
model, and a multi-restart discrete optimiser — was built and benchmarked
entirely on IBM heavy-hex devices. Whether calibration-aware placement gains
survive a change of lattice topology is an open, publishable question in both
directions, and square-lattice Garnet is the natural second data point.

Second, the **parity thread**: on IBM Heron r2 (`ibm_fez`, `ibm_kingston`) we
measured a robust even-over-odd parity-sector leakage asymmetry in
number-conserving Kuramoto-XY Trotter circuits (+10.8 % ± 1.1 pooled, Fisher
p ≪ 10⁻¹⁶; DOI 10.5281/zenodo.18821929). An exact statevector baseline
subsequently proved the noiseless parity leakage of these circuits is exactly
zero at every depth (excitation-number conservation), reframing the result:
the asymmetry is a property of device noise, not of coherent dynamics, and
the live question became whether that noise structure is backend-universal
(DOI 10.5281/zenodo.20382000). A 2026-05-13 Garnet pilot (six layouts,
12,288 shots) was too weak to answer: combined p between 0.05 and 0.57,
sign agreement 1/3 to 3/3 across layouts. Only layout [2, 7, 12, 13] agreed
with IBM in sign at all three depths — which is why the present confirmatory
block pins that layout, discloses the selection, and treats the May data as
design input, never as evidence.

## 2. Methods common to both experiments

**Device and access.** IQM Garnet (20 qubits, square lattice) via IQM
Resonance, 2026-07-21, single calibration window. Live calibration was
snapshotted through the published quality-metrics endpoint before circuit
freezing (per-edge CZ gate fidelity; per-qubit readout error), committed as a
JSON artefact, and used both to drive the optimiser arm of Experiment 1 and to
verify layout availability for Experiment 2.

**Preregistration.** Both campaign documents were committed before execution
(`docs/campaigns/iqm_layout_transfer_square_lattice_prereg_2026-07-21.md`,
`docs/campaigns/iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md`),
freezing hypotheses, endpoints, shot budgets, stop rules, and decision rules.
One pre-submission amendment was made to Experiment 1 (Section 3.1) before
any hardware data existed; the original text and the evidence that forced the
amendment are retained in the document.

**Readiness gates.** Neither experiment was submitted before passing: unit
tests to the repository standard (100 % line and branch coverage on the new
harness modules), a complete `IQMFakeGarnet` dry run of every circuit from
committed code, committed exact classical baselines, and — for Experiment 1 —
a transpiled two-qubit depth-parity gate across arms; for Experiment 2 — a
transpiled depth envelope (May reference depth 159 + 25 %). A zero-spend
server-side `garnet:mock` submission validated the full submit–retrieve loop
before each first real block.

**Spend discipline.** Blocks were submitted incrementally, with dashboard
credit checks between blocks (the first block of each experiment carried a
preregistered abort threshold). The
entire dual campaign consumed roughly 13 of 30 monthly credits.

## 3. Experiment 1 — layout-optimiser transfer to the square lattice

### 3.1 Design

Identical logical circuits per size: two-edge-colour width-2 XY-Trotter
chains (5 Trotter steps = 20 two-qubit layers, constant in n), initial state
at quarter filling (`1000` repeated), which fixes the conserved mean-Z
order-parameter proxy at exactly R = 0.5 for every size and depth. The
endpoint, `err(arm, n) = |R_hw − 0.5|`, uses the absolute mean Z-magnetisation
proxy — a sum of single-qubit marginals, so the two readout-calibration
circuits per size (all-zeros, all-ones over the union of measured qubits)
support an exact tensored per-qubit readout correction for this observable;
no full-matrix mitigation is claimed.

Three arms per size, identical circuits, different placements:

1. **optimised** — highest-fidelity connected chain region from the live
   calibration snapshot, order polished by the discrete Kuramoto layout
   optimiser under a deterministic, transpiler-free chain depth model;
2. **default** — Qiskit automatic placement (optimisation level 1, frozen
   transpiler seed);
3. **naive** — calibration-blind topological baseline (Amendment 1): the
   lexicographically smallest connected chain on the coupling graph.

*Amendment 1 (pre-submission).* The originally preregistered naive arm —
consecutive physical indices (0, …, n−1) — is not a connected path on Garnet;
the fake-backend readiness run showed routing SWAPs inflating its two-qubit
depth to 75/110/83 versus 40 for the other arms, violating the frozen
depth-parity validity gate at every size. The arm was redefined
(calibration-blind, topology-only) before any hardware data existed. After
amendment, all three arms transpile to identical two-qubit depth (40) at
every size, so fidelity differences are attributable to *which* qubits and
edges are used, not to circuit volume.

Matrix: 3 sizes × 3 arms × 2,048 shots + 3 × 2 readout states × 1,024 shots
= 15 circuits, 24,576 + 6,144 shots. On the live calibration the three arms
separated onto genuinely different regions at n = 8: optimised
(16, 11, 10, 15, 14, 13, 17, 18), default (1, 0, 3, 2, 7, 12, 13, 17), naive
(0, 1, 4, 3, 2, 7, 8, 9).

**Frozen primary rule.** `err(optimised) < err(default)` at **all three**
sizes AND the bootstrap 90 % CI of the pooled error difference (10,000
shot-level multinomial resamples of every circuit including the readout
calibrations, seed 20260721) excludes zero. Secondary: `err(default)` versus
`err(naive)`; two-qubit depth parity within 10 % as a validity gate.

### 3.2 Results

Readout-corrected order-parameter error versus the exact reference R = 0.5
(jobs `019f8639-96be…`, `019f8639-980e…`, `019f8642-e162…`,
`019f8642-e326…`, `019f8643-650c…`, `019f8643-6712…`):

| n | optimised | default | naive | optimised wins? |
|---|-----------|---------|-------|-----------------|
| 8 | **0.1475** | 0.2533 | 0.2971 | yes |
| 12 | 0.2088 | **0.1846** | 0.4138 | **no** |
| 16 | **0.2924** | 0.3144 | 0.3378 | yes |

Pooled `err(default) − err(optimised)` = **+0.0345**, bootstrap CI90
[+0.0260, +0.0431]. Pooled `err(naive) − err(default)` = **+0.0989**, CI90
[+0.0898, +0.1079]. Depth parity held exactly (40/40/40 at every size).

### 3.3 Interpretation (bounded, as preregistered)

**The frozen primary rule fails.** The all-sizes condition breaks at n = 12,
so no transfer claim is made — even though the pooled interval excludes zero.
We report exactly the preregistered bounded conclusion:

- calibration-aware placement carries a real pooled advantage on this device
  and calibration window, but the advantage is **not uniform across sizes**;
  the automatic transpiler matched or beat it at one of three sizes;
- **placement itself matters strongly** on the square lattice: the
  calibration-blind topological baseline is worse than automatic placement at
  every size, by three times the optimiser's pooled margin.

A transfer claim therefore required a fresh design powered for per-size
conclusions and uniformity. That preregistered follow-up has now executed.

### 3.4 Powered per-size layout-transfer follow-up

The per-size layout-transfer study was frozen on 2026-07-22 before its
hardware data existed
(`docs/campaigns/iqm_layout_transfer_per_size_prereg_2026-07-22.md`). It
repeated every arm four times at each size (8,192 main shots per arm/size),
added a fresh all-zero/all-one readout pair per size, and submitted the whole
42-circuit, 79,872-shot matrix in one pass on 2026-07-26 under calibration set
`c2097be4-1e23-49bc-adaa-8e8c01df6223`. Provider transpilation preserved an
identical two-qubit depth of 40 for every main circuit.

The frozen corrected `default − optimised` endpoint was significant at all
three sizes after Holm correction, but not in one direction:

| n | default − optimised | bootstrap CI95 | Holm p | direction |
|---|--------------------:|----------------|-------:|-----------|
| 8 | +0.01868 | [+0.00755, +0.02968] | 0.001200 | optimiser advantage |
| 12 | **−0.08728** | [−0.10090, −0.07459] | 0.000600 | **optimiser disadvantage** |
| 16 | +0.04964 | [+0.04109, +0.05823] | 0.000600 | optimiser advantage |

The powered design therefore confirms, rather than resolves away, the n = 12
reversal. Cochran's heterogeneity statistic rejects a common effect across
sizes (Q = 292.43 on 2 df, p = 3.16 × 10⁻⁶⁴). Raw-count sensitivity preserves
all three directions. The pooled difference is slightly negative (−0.00632,
CI90 [−0.01187, −0.00099]) because the n = 12 disadvantage dominates; it must
not be used to erase the per-size result. This is a size-dependent layout
effect in one Garnet calibration window, not a quantum-advantage,
backend-universal, or calibration-aware-superiority claim.

## 4. Experiment 2 — parity-sector noise asymmetry at preregistered power

### 4.1 Design

Committed May-campaign builders, bit-identical circuits: n = 4 XY-Trotter
chains, states `0011` (even sector) and `0001` (odd sector), depths 4, 6, 10,
four repetitions per state/depth at 1,024 shots (calibration-drift
resolution), plus one four-state readout block (`0011`, `0001`, `0000`,
`1111` at 2,048 shots) — 28 circuits, 32,768 shots, all on the pinned layout
[2, 7, 12, 13] (fallback [9, 4, 3, 8] was not needed; all three layout edges
were calibrated, CZ fidelities 0.982–0.990). Transpiled depths at submit
time: 69 / 99 / 159 for depths 4 / 6 / 10 — the deepest exactly reproducing
the May envelope reference of 159, within the frozen +25 % bound of 198.

**Frozen primary.** Pooled across depths and repetitions (12,288 shots per
arm), one-sided two-proportion z-test of `leak_even > leak_odd` at α = 0.05;
committed power analysis: 90 % power at +0.044 relative asymmetry (the
weakest IBM per-depth reference). Secondary: per-depth Wilson intervals and
one-sided tests, per-repetition drift table, sign agreement with the IBM
direction. Claim boundary: with noiseless leakage exactly zero, every
statement is about device noise; no coherent-dynamics claim is available.

### 4.2 Results

Jobs `019f866c-9c99…`, `019f866c-9f20…`, `019f8677-33b7…`, `019f8677-7a0b…`,
`019f8677-ca9a…`:

**Primary: H0 rejected.** Pooled `leak_even` = 0.4194 > `leak_odd` = 0.3928;
relative asymmetry **+0.0677**; z = 4.247; one-sided **p = 1.1 × 10⁻⁵**.
Garnet shows a positive even-over-odd parity-sector leakage asymmetry in the
IBM direction at preregistered power.

**Secondary depth profile (disclosed, not preregistered as an endpoint):**

| depth | leak_even | leak_odd | relative asymmetry | one-sided p | sign = IBM? |
|-------|-----------|----------|--------------------|-------------|-------------|
| 4 | 0.3811 | 0.3098 | **+0.230** | 5.8 × 10⁻¹² | yes |
| 6 | 0.4197 | 0.3875 | **+0.083** | 1.5 × 10⁻³ | yes |
| 10 | 0.4575 | 0.4812 | **−0.049** | 0.98 | **no** |

### 4.3 Interpretation

The pooled IBM-direction asymmetry replicating on a second superconducting
architecture — square-lattice Garnet after heavy-hex Heron r2 — is evidence
against the asymmetry being an artefact of one device family. To be precise
about what the two devices share and what they do not: **shared** is the
pooled even-over-odd sign of the parity-sector leakage asymmetry at
statistical power; **divergent** is the depth dependence (flat on IBM,
decaying and sign-reversing on Garnet), which may well reflect a different
underlying mechanism. "Replication" here therefore binds only to the pooled
directional effect, not to a claim that the two devices exhibit the same
noise structure.

Put side by side, the two devices differ first in scale. On `ibm_kingston`
(the Phase 2 reference: 30 repetitions per sector and depth) the parity
leakage of this circuit family rises from 0.08 at depth 2 to 0.28–0.29 at
depth 20, and the even-minus-odd difference stays between +0.005 and +0.010
from depth 4 to depth 20. On Garnet the leakage is already 0.31–0.38 at depth
4 and reaches 0.48–0.50, the value of a fully mixed parity, at depth 12; the
difference is +0.071 at depth 4 in the July window and shrinks as the leakage
saturates. Garnet thus reaches within twelve Trotter steps a regime that Heron
r2 has not reached at twenty. The devices share the sign of the asymmetry and
differ in its size and in its dependence on depth. The comparison is
descriptive: different days, different numbers of repetitions, and not a
preregistered endpoint.

The initial secondary profile was qualitatively different from IBM's:
IBM's reference asymmetry was approximately flat in depth (≈ +0.044 to
+0.087), while the first Garnet window fell from +0.230 at depth 4 to
−0.049 at depth 10. That isolated negative point initially motivated a
zero-crossing hypothesis, but it was not a preregistered endpoint. The
preregistered follow-ups in Section 5 do not support that hypothesis: the
depth-8 versus depth-12 ordering is a bounded null (p = 0.064), a powered
fresh-window depth-10 retest is positive, and the two depth-10 windows differ
significantly (p ≈ 0.006). The surviving bounded interpretation is therefore
window-to-window variability of device-noise asymmetry, not a depth-localised
zero crossing. No zero-crossing or equilibration-mechanism claim is made.

## 5. Preregistered depth-profile follow-up (executed)

A depth-profile campaign (depths 8 and 12, same layout, states, builders,
and shot discipline; 20 circuits, 24,576 shots) was preregistered separately
(`docs/campaigns/iqm_dla_depth_profile_prereg_2026-07-22.md`) — frozen
before any of its data existed — and executed on 2026-07-22 under the same
calibration set as Experiment 2 (jobs `019f8772-6338…`, `019f8772-665d…`,
`019f87eb-30b1…`, `019f87eb-f2f8…`, `019f87ec-81b0…`).

**Its frozen primary (decay ordering) fails to reject the null:**
Δ₈ = +0.0369 exceeds Δ₁₂ = +0.0132 in the hypothesised direction, but the
one-sided test gives z = 1.52, p = 0.064 > 0.05, so the decay ordering is
not established at the committed power and we report the bounded null with
the same prominence as any positive result.

Secondary, descriptive structure across the joint profile (depths 4/6/10
from Experiment 2's execution window, 8/12 from this one — a cross-window
comparison frozen into the preregistration, mitigated by the shared
calibration set):

| depth | relative asymmetry | mean total leakage | sign |
|-------|--------------------|--------------------|------|
| 4 | +0.230 | 0.345 | + |
| 6 | +0.083 | 0.404 | + |
| 8 | +0.087 | 0.442 | + |
| 10 | −0.049 | 0.469 | − |
| 12 | +0.027 | 0.489 | + |

Two hypothesis-generating observations arose: the mean total leakage rises
monotonically toward 0.5, consistent with sector equilibration compressing
the asymmetry at depth; and the sign pattern (+, +, +, −, +) places the
depth-10 negative between positive neighbours, suggesting a window
fluctuation rather than a systematic zero crossing.

**A third preregistered campaign tested the second reading directly**
(`docs/campaigns/iqm_dla_d10_retest_prereg_2026-07-22.md`, frozen before
its data; 8,192 shots per arm — 90 % power at the observed −0.0237 effect,
which the depth-profile arms could not resolve; jobs `019f8809-42d7…`,
`019f8809-448d…`). **The depth-10 negative sign does not replicate:** in
the fresh window the depth-10 asymmetry is positive (leak_even 0.4611 >
leak_odd 0.4478; +0.030 relative), the one-sided test for the negative
sign gives p = 0.956, and the preregistered cross-window secondary finds
the two windows' depth-10 asymmetries genuinely differ (z = 2.74,
two-sided p ≈ 0.006). Per the frozen wording this strengthens — without
proving — the window-fluctuation reading; with the fresh-window point
positive, every depth in the joint profile carries a positive sign, and
the surviving open question is the size of the window-to-window
variability of the asymmetry at fixed depth, not a zero crossing.

## 6. Window-to-window variability of the asymmetry (preregistered, executed)

Section 5 left one question: how large is the window-to-window variability of
the parity-sector asymmetry at a fixed depth. A fourth campaign was
preregistered to measure it directly
(`docs/campaigns/iqm_dla_window_variability_prereg_2026-07-22.md`). One
nominal window is a complete 36-circuit execution on the pinned layout [2, 7,
12, 13] at depths 4, 8, 10 and 12, submitted as one main job and one readout
job, with 4,096 shots per state and depth; the window is the unit of custody
and of the frozen analysis. Ten windows were executed between 2026-07-22 and
2026-09-04.

**Frozen primary (depth 10, window level).** The even-minus-odd leakage
difference is heterogeneous across the ten windows beyond shot noise: Cochran
Q = 19.01 on 9 degrees of freedom, p = 0.025, with a between-window standard
deviation τ = 0.0115 against a mean shot-noise standard error of 0.0109 and a
weighted mean difference of +0.0179. The preregistered rule "drift exceeds
shot noise" is met.

**Secondary, by depth (Holm-adjusted).** Homogeneity is rejected at depth 8
(adjusted p = 4.4 × 10⁻⁴) and not at depth 4 (0.053) or depth 12 (0.40). The
between-window τ is 0.0107, 0.0176, 0.0115 and 0.0022 at depths 4, 8, 10 and
12. The depth-4 difference is positive in nine of ten windows (Clopper–Pearson
95 % interval 0.55 to 1.00).

**Calibration epochs (amendment, sensitivity analysis).** While the campaign
ran we found that several nominal windows carried the same provider
calibration set: windows 2–3 and windows 7–10. Windows under one calibration
set are technical replicates, not independent observations. An amendment
(`docs/campaigns/iqm_dla_window_variability_epoch_amendment_2026-09-04.md`),
committed before windows 8 to 10 were counted for inference, adds an epoch-pooled analysis and requires both results to be reported together. Pooled to
six calibration epochs, the depth-10 heterogeneity is stronger, not weaker: Q
= 15.55 on 5 degrees of freedom, p = 0.0083, τ = 0.0129 against a shot-noise
standard error of 0.0085. Homogeneity is rejected at depth 4 (adjusted p =
0.012) and depth 8 (5.6 × 10⁻⁵), not at depth 12 (0.64; τ estimated at zero).

**What this establishes.** On this device and layout the parity-sector
asymmetry at depths 4, 8 and 10 varies between calibration epochs by an amount
comparable to or larger than the shot noise of a 4,096-shot arm. Adding shots
inside one calibration window therefore narrows the wrong interval: the
reported uncertainty of a single-window asymmetry understates how far the next
window will be. This is why the depth-ordering question of Section 5 (one-sided p = 0.064 inside one calibration set) could not be settled by more
shots, and it fixes the design of the confirmatory campaign that follows: the
calibration epoch is the independent unit.

**Boundary.** A changed calibration-set identifier marks an operational epoch;
it does not prove physical independence, and slow drift can cross calibration
boundaries. No cause of the variability is claimed; the calibration covariates
were preregistered as descriptive only.

## 7. Depth-ordering confirmation across calibration epochs (preregistered, executed)

**Design.** Frozen before any of its data existed
(`docs/campaigns/iqm_dla_depth_profile_powered_epoch_prereg_2026-09-04.md`).
The six calibration
epochs of Section 6 gave the depth contrast C = (leak_even,8 − leak_odd,8) −
(leak_even,12 − leak_odd,12) a mean of 0.0164 and a between-epoch τ of 0.0150.
Those values were used to choose the sample size and for nothing else: their
counts and their six calibration sets are excluded from the confirmatory
endpoint. The campaign consists of twelve distinct, previously unused
calibration epochs. Each epoch runs the complete matrix in one pass: twelve
repetitions of each of the four state-by-depth cells at 1,024 shots (12,288
shots per cell), plus four readout states at 2,048 shots — 48 main circuits in
one provider job and 4 readout circuits in a second, 57,344 shots per epoch
and 688,128 in total. An epoch is admitted only if its calibration set differs
from all earlier ones and from the six design epochs, every earlier epoch has
a complete paired calibration and count record, and the three edges of the
layout exist in the snapshot. No circuit, layout, depth, repetition or shot
count may change in response to a result; there is no optional stopping and no
replacement of an admitted epoch.

**Analysis (frozen).** A DerSimonian–Laird random-effects model of the twelve
epoch contrasts with binomial within-epoch variances; one-sided test of H0:
mean(C) ≤ 0 with the Hartung–Knapp–Sidik–Jonkman statistic on 11 degrees of
freedom, its scale floored at one so that a small observed residual cannot
narrow the interval. The null is rejected only if the estimated mean is
positive and p < 0.05. Projected power at the design mean is 0.92. Raw parity
leakage is primary; no readout correction is applied to the endpoint.

**Execution.** Twelve epochs between 2026-09-23 and 2026-10-06 on IQM Garnet,
twelve distinct calibration sets, 24 provider jobs, 624 circuits. Every epoch
passed the admission gate on its own day; the provider's payload digests equal
the digests written to the submission journal before each call; the frozen
validator accepted the counts of every epoch. The endpoint analysis was run
once, after the twelfth epoch was validated, with the analysis script
unchanged since its freeze on 2026-09-04 (`data/iqm_paper_replication/iqm_dla_depth_profile_powered_analysis_2026-10-11.json`).

**Result.** The null is rejected. The mean depth contrast is +0.0173 (standard
error 0.0080; t = 2.176 on 11 degrees of freedom, one-sided p = 0.026), with
between-epoch τ = 0.0261 (Cochran Q = 108.0 on 11 degrees of freedom). The
one-sided 95 % lower bound of the mean, from the same standard error and t
distribution, is +0.0030. The contrast is positive in nine of twelve epochs
and negative in three (epochs 1, 4 and 9). On this device and layout the even-odd leakage asymmetry is, on average over calibration epochs, larger at depth
8 than at depth 12: the ordering that Section 5 could not establish inside one
calibration set is established across twelve.

The between-epoch τ is larger than the 0.0150 assumed when the sample size was
chosen, so the campaign had less power than the planned 0.92; the rejection
stands on the preregistered test as run, and the lower bound is close to zero.
The result is a statement about the mean over epochs: single epochs differ by
far more than their shot noise (per-epoch shot-noise standard error of the
contrast about 0.0088), and in three of twelve the sign is reversed.

**Secondary (preregistered).** Random-effects summaries of the asymmetry at
each depth separately: +0.0225 at depth 8 (standard error 0.0046, one-sided p
= 0.00024, τ = 0.0146) and +0.0051 at depth 12 (standard error 0.0056, p =
0.19, τ = 0.0183). The six design epochs, which are not part of the endpoint,
had contrasts −0.0159, +0.0010, +0.0127, +0.0466, +0.0361 and +0.0202. The
four readout states of every epoch are committed with the counts and are not
used in the endpoint.

| epoch | date | calibration set | jobs (main, readout) | Δ₈ | Δ₁₂ | contrast |
|---|---|---|---|---|---|---|
| 1 | 2026-09-23 | `ad7e2bf0-48ac-493d-aa9a-51d24d0345d3` | `01a0cfbc-c583…`, `01a0cfbc-c711…` | +0.0109 | +0.0231 | -0.0122 |
| 2 | 2026-09-24 | `7c39177a-a124-4974-85e6-466d9c1daddf` | `01a0d203-4e04…`, `01a0d203-4fd7…` | +0.0162 | -0.0014 | +0.0176 |
| 3 | 2026-09-25 | `af33e1ea-0ad5-4521-8c27-09343f0bc390` | `01a0d916-2e1a…`, `01a0d916-2f53…` | +0.0094 | +0.0046 | +0.0049 |
| 4 | 2026-09-26 | `d10715c4-4242-4af7-be1e-29f3fabf7e1a` | `01a0dec6-b2b3…`, `01a0dec6-b3f0…` | +0.0008 | +0.0145 | -0.0137 |
| 5 | 2026-09-27 | `e1084b22-ca16-4f83-98af-500a91ef5e5c` | `01a0e410-5e40…`, `01a0e410-5f79…` | +0.0263 | +0.0233 | +0.0030 |
| 6 | 2026-09-29 | `a1a74f13-3d45-4be2-afe8-3aa35e23da14` | `01a0ef34-e362…`, `01a0ef34-e611…` | +0.0382 | -0.0242 | +0.0624 |
| 7 | 2026-10-01 | `36cf327c-ecbe-470b-a719-aa864c078268` | `01a0f933-9f69…`, `01a0f933-a1de…` | -0.0033 | -0.0122 | +0.0090 |
| 8 | 2026-10-02 | `a4b99350-ad83-4838-947d-078b0a2dffe3` | `01a0fbf5-baa8…`, `01a0fbf5-bc08…` | +0.0436 | -0.0214 | +0.0650 |
| 9 | 2026-10-03 | `0660d260-dd41-43c9-950f-b0f9009fd452` | `01a101dc-eada…`, `01a101dc-ec11…` | +0.0273 | +0.0417 | -0.0144 |
| 10 | 2026-10-04 | `e431ada3-f5ef-4a60-95f7-513eeeb15ac7` | `01a10729-ba87…`, `01a10729-bd8d…` | +0.0293 | -0.0017 | +0.0310 |
| 11 | 2026-10-05 | `ff6836ae-8df9-4d0f-92d3-09c1617d9ccc` | `01a10da8-ca46…`, `01a10da8-cbf2…` | +0.0253 | +0.0113 | +0.0140 |
| 12 | 2026-10-06 | `6dd7fae6-4011-43d7-ad45-beed980d3a3f` | `01a11091-86f0…`, `01a11091-890d…` | +0.0452 | +0.0041 | +0.0412 |

Δ is the even-minus-odd leakage difference at the given depth, pooled over the
twelve repetitions of the epoch; the contrast is Δ₈ − Δ₁₂.

**Boundary.** A statement about device noise on IQM Garnet, this layout, these
two depths, this circuit family and this access period. Noiseless parity
leakage of these circuits is exactly zero, so nothing here is a statement
about coherent dynamics. No causal mechanism, no transfer to another backend,
no advantage claim.

## 8. Outlook: other devices and processor classes

Everything reported here and in our earlier work was measured on two families
of superconducting processors: heavy-hex IBM Heron r2 and square-lattice IQM
Garnet. The pooled even-over-odd sign of the parity-sector leakage asymmetry
is common to both; its depth dependence is not, and on Garnet it moves between
calibration epochs by more than the shot noise. Two devices cannot tell
whether these are properties of superconducting hardware in general, of
particular two-qubit-gate implementations, or of these two machines.

The measurement is small enough to be repeated on any current gate-model
processor: four qubits, number-conserving XY-Trotter circuits of depth 4 to
12, two initial states, and a few thousand shots per arm. Because the
noiseless parity leakage of these circuits is exactly zero, the observable is
a noise diagnostic that needs no classical reference beyond the counts
themselves. Processor classes whose dominant error channels, connectivity and
recalibration cadence differ from those of transmon devices — trapped ions,
neutral atoms, spin qubits, and superconducting architectures with other
couplers or lattices — are the informative comparison. We make no prediction
for them: an asymmetry of the same sign, of the opposite sign, or a bounded
null on another class would each constrain the explanation.

The protocol is published so that it can be run without us: the
preregistration texts, circuit builders, admission checks, submission journals
and frozen analysis scripts are in the public repository (Section 9). The one
design lesson we would carry over is the unit of analysis: on Garnet the
calibration epoch, not the shot, is the independent observation, and the
equivalent recalibration event of another platform should be identified before
the sample size is fixed. The confirmatory design used here costs twelve
epochs of 57,344 shots. We welcome replications by device operators and
independent groups, and would run the same preregistered protocol on other
hardware where access is available.

## 9. Reproducibility

Everything needed to reproduce all five campaigns is committed to the public
repository (github.com/anulum/scpn-quantum-control):

- preregistrations with frozen decision rules (and the amendment history);
- live calibration snapshot with calibration-set identifier;
- exact circuit payloads (QPY) and circuit-label manifests;
- submission records with raw IQM job identifiers (publishable; the access
  token is never committed);
- raw measurement counts for every reported hardware circuit;
- for the twelve-epoch campaign, the per-epoch submission journals with the
  digest of every circuit payload written before the provider call, and the
  admission check each epoch passed before it was submitted;
- frozen analysis scripts (fixed bootstrap seeds) whose committed artefacts
  contain every number quoted here;
- the harness modules with 100 % test coverage, hermetic against synthetic
  lattices.

## 10. Claim boundaries

No quantum-advantage claim (all circuits are classically simulable at these
sizes — that is what makes the exact references possible). No
coherence-protection claim. No coherent-dynamics claim for the parity
asymmetry (noiseless leakage is exactly zero). No layout-transfer claim
(the frozen rule failed). No extrapolation beyond the sampled device,
calibration windows, chain lengths, depths, and schedules. The core results
are statements about IQM Garnet on 2026-07-21 under calibration set
`246a4930-54e3-4cd9-a2d1-fcc0919675f5`; the per-size study is separately
bounded to its
2026-07-26 window under calibration set
`c2097be4-1e23-49bc-adaa-8e8c01df6223`. The window-variability and
depth-ordering results are statements about the pinned layout
[2, 7, 12, 13] over the windows and calibration epochs listed in Sections 6
and 7; a distinct calibration set marks an operational epoch and does not
prove physical independence, and the depth ordering is established for the
mean over twelve epochs, not for an individual epoch.

## Acknowledgements

Quantum-hardware access was provided by **IQM** through the **IQM Resonance**
platform under an extended-access research grant. We thank the IQM support
team for the vetting and provisioning of that access.
