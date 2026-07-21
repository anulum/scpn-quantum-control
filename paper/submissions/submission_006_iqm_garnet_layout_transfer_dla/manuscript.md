<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IQM Garnet dual preregistered campaign manuscript -->

# Two Preregistered Experiments on a Square-Lattice Superconducting Processor: Layout-Optimiser Transfer Fails Its Frozen Rule While Parity-Sector Noise Asymmetry Replicates at Power

**Status: DRAFT — not submitted. Publication requires the owner's direct go.**

Miroslav Šotek (ORCID 0009-0009-3560-0851), with Anulum Fortis & Arcane Sapience
(protoscience@anulum.li)

All quantum-hardware results in this work were obtained on IQM Garnet through
**IQM Resonance**; we thank IQM for extended Resonance access. Circuits ran on
20 qubits of the square-lattice Garnet device on 2026-07-21, within a single
calibration window (calibration set `246a4930-54e3-4cd9-a2d1-fcc0919675f5`).

## Abstract

We report two preregistered experiments executed back-to-back on the IQM
Garnet 20-qubit square-lattice processor, both with decision rules frozen and
committed before any hardware data existed, and both published regardless of
outcome direction.

**Experiment 1 (layout transfer).** We asked whether the fidelity gains of a
calibration-aware Kuramoto-XY layout optimiser, developed and benchmarked on
IBM heavy-hex topologies, transfer to a square lattice. Three placements of
identical two-edge-colour XY-Trotter chains (n = 8, 12, 16) were compared: a
calibration-aware optimised chain, the transpiler's automatic placement, and a
calibration-blind topological baseline. The frozen primary rule — the
optimised arm must win at all three sizes and the pooled bootstrap 90 %
confidence interval must exclude zero — **failed**: the optimised arm won at
n = 8 and n = 16 but lost to the automatic transpiler at n = 12, despite a
positive pooled advantage of +0.035 (CI90 [+0.026, +0.043]) in
readout-corrected order-parameter error. We therefore make no transfer claim
and report the preregistered bounded result: the calibration-aware advantage
on this device is real in aggregate but not uniform across sizes, while
placement per se matters unambiguously (the topological baseline is worse than
automatic placement at every size; pooled +0.099, CI90 [+0.090, +0.108]).

**Experiment 2 (parity-sector noise asymmetry).** A published IBM Heron r2
result found that hardware parity-sector leakage in number-conserving
XY-Trotter circuits is asymmetric between even and odd excitation-number
sectors (+10.8 % ± 1.1 relative). An exact statevector baseline pins the
noiseless leakage of these circuits at zero, so any such asymmetry is a
statement about device noise; the open question was whether the asymmetry is
backend-universal or backend-specific. Our earlier 12,288-shot Garnet pilot
was underpowered and mixed-sign. Here, a preregistered 32,768-shot design
(90 % power at +0.044 relative asymmetry, the weakest IBM per-depth reference)
on the pinned layout [2, 7, 12, 13] **rejects the null**: pooled even-sector
leakage 0.4194 exceeds odd-sector leakage 0.3928 (relative asymmetry +0.068,
z = 4.25, one-sided p = 1.1 × 10⁻⁵). The IBM-direction asymmetry replicates on
a second superconducting architecture at preregistered power. The depth
profile, however, differs sharply from IBM's: +0.230 at depth 4, +0.083 at
depth 6, and −0.049 (sign reversed) at depth 10, versus IBM's flat ≈ +0.044
profile — the asymmetry on Garnet decays with circuit depth and crosses zero
between depths 6 and 10. This depth structure was not preregistered as an
endpoint and is reported as secondary, hypothesis-generating evidence.

Both experiments are fully reproducible from the public repository: committed
circuit files, calibration snapshots, raw counts, job identifiers, and frozen
analysis scripts with fixed seeds.

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

**Spend discipline.** Blocks were submitted incrementally with owner
approval per block and dashboard credit checks between blocks (the first
block of each experiment carried a preregistered abort threshold). The
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

A transfer claim would require a design powered for per-size conclusions and
uniformity; that is future work requiring a fresh preregistration.

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

The depth profile, however, is qualitatively different: IBM's reference
asymmetry was approximately flat in depth (≈ +0.044 to +0.087), whereas
Garnet's decays monotonically from +0.230 at depth 4 and crosses zero between
depths 6 and 10. At depth 10 both sectors leak close to half the shots
(0.46–0.48), approaching sector equilibration, which plausibly compresses any
asymmetry; the d4 asymmetry (largest, at the lowest total leakage) is
therefore the cleanest signature. Localising the zero crossing and testing
the equilibration explanation requires depths between 6 and 10 and beyond —
preregistered as a follow-up (Section 5) and **not** claimed here.

## 5. Follow-up preregistration

A depth-profile campaign (depths 8 and 12, same layout, states, builders,
and shot discipline) is preregistered separately
(`docs/campaigns/iqm_dla_depth_profile_prereg_2026-07-22.md`) to localise the
zero crossing observed above. It is committed before execution; its data are
not part of this manuscript's claims.

## 6. Reproducibility

Everything needed to reproduce both experiments is committed to the public
repository (github.com/anulum/scpn-quantum-control):

- preregistrations with frozen decision rules (and the amendment history);
- live calibration snapshot with calibration-set identifier;
- exact circuit payloads (QPY) and circuit-label manifests;
- submission records with raw IQM job identifiers (publishable; the access
  token is never committed);
- raw measurement counts for all 43 hardware circuits;
- frozen analysis scripts (fixed bootstrap seeds) whose committed artefacts
  contain every number quoted here;
- the harness modules with 100 % test coverage, hermetic against synthetic
  lattices.

## 7. Claim boundaries

No quantum-advantage claim (all circuits are classically simulable at these
sizes — that is what makes the exact references possible). No
coherence-protection claim. No coherent-dynamics claim for the parity
asymmetry (noiseless leakage is exactly zero). No layout-transfer claim
(the frozen rule failed). No extrapolation beyond the sampled device,
calibration window, chain lengths, depths, and schedules. Results are
statements about IQM Garnet on 2026-07-21 under calibration set
`246a4930-54e3-4cd9-a2d1-fcc0919675f5`.

## Acknowledgements

Quantum-hardware access was provided by **IQM** through the **IQM Resonance**
platform under an extended-access research grant. We thank the IQM support
team for the vetting and provisioning of that access.
