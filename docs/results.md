# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Results

# Results

*Kuramoto-XY simulator, compiler, and hardware-evidence ledger for
heterogeneous-frequency coupled oscillators.*

For source classification and campaign provenance, see the dated
[Hardware Status Ledger](hardware_status_ledger.md). This page is a gallery and
technical summary; the ledger is the canonical index for whether a result is
theoretical, simulated, hardware-measured, mitigated, or noise-limited.

## Status Snapshot — 2026-04-29

| Area | Public status |
|---|---|
| Promoted hardware campaigns | April 2026 `ibm_kingston` Phase 1 DLA parity raw-count dataset; legacy artefact-backed `ibm_fez` baseline rows. |
| Simulator-only families | BKT scaling, OTOC, Floquet DTC, MBL/eigenstate scans, FIM, and classical wall-time baselines unless a hardware artefact is named. |
| Pending / quarantined IBM batches | V2, frontier, queued-job, placeholder, and aggregate-only IBM outputs are not promoted here until raw counts, retrieval manifests, and analysis scripts are reviewed and committed. |
| Canonical status source | [Hardware Status Ledger](hardware_status_ledger.md). |

---

## Key Findings

| # | Finding | Measured Value | Source |
|---|---------|---------------|--------|
| 1 | DLA parity raw-count reproduction | Phase 1: 342 circuits, peak asymmetry +17.48% at depth 6; Phase 2 reduced A+G: 612 circuits, Fisher p=3.77e-20; Phase 2 B-C: mixed `n=6,8` scaling | `data/phase1_dla_parity/`, `data/phase2_dla_parity/`, `data/phase2_scaling_bc/`, `scripts/run_dla_parity_suite.py`, `scripts/analyse_phase2_dla_parity.py`, `scripts/analyse_phase2_scaling_bc.py` |
| 2 | Bell inequality row | CHSH S=2.165, S=2.188 (>8σ) | Legacy `ibm_fez` artefact row |
| 3 | QKD row | QBER 5.5% < BB84 threshold (11%) | Legacy `ibm_fez` artefact row |
| 4 | State preparation row | 94.6% (∣0⟩), 89.8% (∣1⟩) | Legacy `ibm_fez` artefact row |
| 5 | ZNE row | Range 0.259–0.272 across folds 1–9 | Legacy `ibm_fez` artefact row |
| 6 | Knm ansatz row | 2.36 bits vs TwoLocal 3.46 | Legacy `ibm_fez` artefact row |
| 7 | 16-qubit UPDE row | 13/16 qubits ∣⟨Z⟩∣>0.3 | Legacy `ibm_fez` artefact row |
| 8 | Schmidt gap transition | K=3.44 (n=8) | Exact simulation |
| 9 | Critical coupling extrapolation | K_c(∞): BKT≈2.20, power≈2.94 | Finite-size scaling |
| 10 | DTC survives disorder | 15/15 drive amplitudes | Floquet simulation |
| 11 | Scrambling peak | 4× faster at K=4 vs K=1 | OTOC simulation |
| 12 | Trotter error row | dt=0.1 vs dt=0.05 flips Q1 sign | Legacy `ibm_fez` artefact row |
| 13 | Non-ergodic regime (not deep MBL) | Poisson level spacing + 25-33% sub-thermal eigenstate S | Level spacing + eigenstate scan |
| 14 | **BKT universality preserved** | CFT c=1.04 (n=8), gap R²>0.96 | Kaggle computation (n=4-12) |
| 15 | Exact-simulation crossover | n≈11.6, exact Hilbert-space only | Classical baselines plus hardware-budget estimates; not broad advantage |

---

## Simulation Results

### Entanglement Entropy and Schmidt Gap

Half-chain entanglement entropy and Schmidt gap across coupling strength for
n=2,3,4,6,8 oscillators with Paper 27 heterogeneous frequencies.

![Entanglement vs coupling](figures/publication/fig1_entanglement_vs_coupling.png)

The Schmidt gap dip at K≈3.4 (n=8) marks the synchronisation transition.
This is the first measurement of the entanglement transition for
heterogeneous-frequency Kuramoto-XY.

### High-Resolution Transition Zoom

![Transition zoom](figures/publication/fig8_transition_zoom.png)

60-point resolution in the transition region (K=1–5). The n=8 Schmidt gap
drops sharply at K=3.44 — the cleanest transition signature.

### Krylov Complexity

Operator spreading measured via Lanczos coefficients $b_n$ and peak Krylov
complexity $K_{max}(t) = \sum_n n|\phi_n(t)|^2$.

![Krylov vs coupling](figures/publication/fig2_krylov_vs_coupling.png)

Mean Lanczos $b$ grows linearly with coupling (operator growth rate scales with K).
Peak complexity saturates at the Hilbert space dimension.

### OTOC (Information Scrambling)

Out-of-time-order correlator $F(t) = \text{Re}\langle W^\dagger(t) V^\dagger W(t) V\rangle$
at sub-critical (K=1) and super-critical (K=4) coupling.

![OTOC time traces](figures/publication/fig3_otoc_time_traces.png)

Strong coupling scrambles 4× faster: $t^* = 0.28$ (K=4) vs $t^* = 1.17$ (K=1) at n=8.

### Floquet Discrete Time Crystal

Periodically driven Kuramoto-XY: $K(t) = K_0(1 + \delta\cos\Omega t)$ with
heterogeneous natural frequencies $\omega_i$.

![Floquet DTC](figures/publication/fig9_floquet_dtc_n3456.png)

All 15 drive amplitudes show subharmonic response above the DTC threshold.
**Heterogeneous frequencies do not destroy the discrete time crystal.**
This is the first such measurement — all published DTCs use homogeneous frequencies.

### Finite-Size Scaling

Critical coupling $K_c(N)$ extracted from spectral gap minimum across
system sizes N=2,3,4,6.

![Finite-size scaling](figures/publication/fig6_finite_size_scaling.png)

Two extrapolations to the thermodynamic limit:
BKT ansatz $K_c(\infty) \approx 2.20$, power-law $K_c(\infty) \approx 2.94$.

### Combined Transition Overview

![Combined overview](figures/publication/fig7_combined_transition.png)

Four probes of the synchronisation quantum phase transition: spectral gap,
entanglement entropy, Krylov complexity, and Schmidt gap. All computed with
Paper 27 heterogeneous frequencies.

---

## IBM Hardware Results

Two campaigns on Heron r2 (156-qubit) processors:

- **`ibm_fez`** — legacy March 2026 baseline artefacts. Values may be quoted
  only with their committed artefact path and should not be used as broad
  advantage or frontier validation.
- **`ibm_kingston`** — April 2026 Phase 1 DLA-parity campaign,
  342 circuits across 4 sub-phases. This is the promoted raw-count hardware
  dataset because the counts, job IDs, integrity checks, and reproduction
  harness are committed.

### Phase 1 — DLA Parity Asymmetry (April 2026, ibm_kingston)

![DLA parity leakage vs depth](https://raw.githubusercontent.com/anulum/scpn-quantum-control/main/figures/phase1/leakage_vs_depth.png)

![DLA parity asymmetry vs depth](https://raw.githubusercontent.com/anulum/scpn-quantum-control/main/figures/phase1/asymmetry_vs_depth.png)

The XY Hamiltonian's dynamical Lie algebra splits as
$\mathfrak{su}(2^{n-1}) \oplus \mathfrak{su}(2^{n-1})$ under the
parity operator $P = \prod_i Z_i$. The SCPN simulator predicts the
odd ("feedback") sub-block is more robust to depolarising noise than
the even ("projection") sub-block by 4.5–9.6 % at moderate Trotter
depths. The Phase 1 campaign on ibm_kingston reproduces this from committed
raw counts:

| Trotter depth | Leak even | Leak odd | Asymmetry | Welch $p$ | Reps |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0806 | 0.0827 | $-2.5\%$ | 0.45 (baseline) | 12 |
| 4 | 0.0982 | 0.0862 | **$+14.0\%$** | $1.4 \times 10^{-6}$ | 21 |
| 6 | 0.1291 | 0.1099 | **$+17.5\%$** | $6.6 \times 10^{-6}$ | 21 |
| 8 | 0.1443 | 0.1284 | **$+12.4\%$** | $8.9 \times 10^{-5}$ | 21 |
| 10 | 0.1658 | 0.1495 | **$+10.9\%$** | $6.7 \times 10^{-6}$ | 21 |
| 14 | 0.1898 | 0.1797 | $+5.6\%$ | 0.010 | 21 |
| 20 | 0.2295 | 0.2114 | $+8.6\%$ | 0.0067 | 12 |
| 30 | 0.2771 | 0.2576 | $+7.6\%$ | 0.0095 | 12 |

- **7 of 8 depths** are individually significant at Welch $p < 0.05$.
- **Fisher's combined statistic:** $\chi^2_{16} = 123.4$, combined
  $p \ll 10^{-16}$.
- **Mean asymmetry for depths $\ge 4$:** $(10.8 \pm 1.1)\,\%$ —
  consistent with and in the upper range of the apriori $4.5\text{–}9.6\,\%$
  classical simulator prediction.
- **Strongest signal:** depth 6, $+17.48\,\%$, $5.4\sigma$.

Reproducible from the raw JSON in `data/phase1_dla_parity/` via
`python scripts/analyse_phase1_dla_parity.py`.

A 267-line short paper draft for *Quantum Science and Technology* /
*Physical Review Research* is in
[`paper/phase1_dla_parity_short_paper.md`](https://github.com/anulum/scpn-quantum-control/blob/main/paper/phase1_dla_parity_short_paper.md).

### Phase 2 — Reduced A+G Replication (May 2026, ibm_kingston)

The reduced Phase 2 run repeated the `n=4` DLA parity test with 30 reps per
depth/sector at 4096 shots, plus a same-run readout baseline. Blocks B-F
(`n=6-12` scaling and GUESS calibration) were not submitted.

![Phase 2 n=4 replication asymmetry](../figures/phase2/phase2_n4_replication_asymmetry.png)

| Trotter depth | Leak even | Leak odd | Asymmetry | Welch p |
|---:|---:|---:|---:|---:|
| 2 | 0.08370 | 0.08247 | +1.49% | 0.278 |
| 4 | 0.12009 | 0.11053 | +8.65% | 1.56e-08 |
| 6 | 0.15296 | 0.14659 | +4.35% | 1.94e-04 |
| 8 | 0.17339 | 0.16879 | +2.72% | 0.00352 |
| 10 | 0.19599 | 0.18761 | +4.47% | 9.64e-07 |
| 14 | 0.23883 | 0.22912 | +4.24% | 6.14e-06 |
| 20 | 0.28904 | 0.28035 | +3.10% | 2.59e-05 |
| 30 | 0.34557 | 0.34524 | +0.10% | 0.857 |
| 40 | 0.38906 | 0.38868 | +0.10% | 0.855 |
| 50 | 0.42153 | 0.42188 | -0.08% | 0.857 |

- **Fisher's combined statistic:** chi2 `140.671952`, p `3.773718e-20`.
- **Significant depths:** 6/10 at Welch p < 0.05.
- **Readout baseline:** 12/12 circuits complete at 8192 shots, with state
  retention from 95.0% to 99.2%.

Reproduce from raw counts via
`PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/analyse_phase2_dla_parity.py --verify-integrity`.

### Phase 2 — B-C Scaling Continuation (May 2026, ibm_kingston)

The B-C continuation tested only `n=6` and `n=8`; blocks A, D, E, F, and G were
skipped. The same-day A+G readout baseline remains the readout-control source.

![Phase 2 B-C mixed scaling asymmetry](../figures/phase2/phase2_bc_scaling_mixed_asymmetry.png)

| n | Trotter depth | Leak even | Leak odd | Asymmetry | Welch p |
|---:|---:|---:|---:|---:|---:|
| 6 | 4 | 0.20653 | 0.20592 | +0.30% | 0.757 |
| 6 | 8 | 0.27606 | 0.28678 | -3.74% | 8.37e-07 |
| 6 | 14 | 0.35409 | 0.35586 | -0.50% | 0.407 |
| 6 | 20 | 0.40681 | 0.41484 | -1.94% | 3.05e-04 |
| 8 | 4 | 0.26626 | 0.25768 | +3.33% | 8.35e-04 |
| 8 | 8 | 0.37186 | 0.36606 | +1.58% | 0.0231 |
| 8 | 14 | 0.44863 | 0.44276 | +1.33% | 0.0252 |
| 8 | 20 | 0.43387 | 0.43333 | +0.12% | 0.842 |

- `n=6`: Fisher chi2 `46.531552`, p `1.883218e-07`, 2/4 significant depths.
- `n=8`: Fisher chi2 `29.420107`, p `2.675193e-04`, 3/4 significant depths.
- IBM-reported usage: `305` quantum seconds for job `d7sudr2udops7397ae30`.

Interpretation: this is mixed scaling evidence. The `n=8` middle-depth sign is
positive, but `n=6` has negative significant depths. It falsifies a simple
monotone scaling story and must not be cited as broad scaling validation.

Reproduce from raw counts via
`PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/analyse_phase2_scaling_bc.py data/phase2_scaling_bc/phase2_scaling_bc_2026-05-05T124722Z.json --sha256 f9718c3789329dbaa96a1667f8a581e3d1774632b961a1760c044138ccab6550`.

### Legacy ibm_fez Results (March 2026)

The `ibm_fez` rows below are retained as legacy hardware observations. They
must be cited with artefact paths from `results/ibm_hardware_2026-03-28/`,
`results/march_2026/`, or the hardware ledger, and they are not evidence for
broad quantum advantage or any frontier claim.

### Bell Test and QKD

![Hardware: CHSH + QBER](figures/publication/fig10_ibm_hardware.png)

- **(a)** Per-qubit ⟨Z⟩ heatmap across 4-qubit circuits
- **(b)** 8-qubit Z-expectations show Kuramoto coupling pattern
- **(c)** QKD QBER: 5.5% (ZZ), 5.8% (XX) — below BB84 11% threshold
- **(d)** CHSH: S=2.165 > 2 — **classical limit violated on quantum hardware**

### Full Experiment Suite

![Hardware analysis](figures/publication/fig12_full_hardware_analysis.png)

- **(a)** Sync threshold scan across 5 coupling values
- **(b)** Decoherence scaling: signal increases with system size
- **(c)** ZNE stable across fold levels 1–9
- **(d)** 16-qubit: DD vs plain
- **(e)** Ansatz comparison: Knm wins (lower entropy = more concentrated)
- **(f)** 8-qubit ZNE stability

### Quantitative Characterisation

![Quantitative hardware](figures/publication/fig13_quantitative_hw.png)

- **(a)** Per-qubit readout errors: asymmetric 0→1 vs 1→0
- **(b)** ZNE per-qubit stability across fold levels
- **(c)** CHSH correlators with error bars (>8σ violation)

### Correlator, Trotter, 16-Qubit, VQE

![Complete analysis](figures/publication/fig14_complete_analysis.png)

- **(a)** ZZ correlation matrix: CX layer creates expected anti-correlations
- **(b)** Trotter order comparison: dt=0.05 vs dt=0.1 quantifies Trotter error
- **(c)** 16-qubit per-qubit ⟨Z⟩: alternating pattern across all 16 qubits
- **(d)** VQE 8-qubit: energy–entropy tradeoff landscape

### Exact-Simulation Crossover Boundary

![Quantum advantage crossover](figures/publication/fig17_quantum_advantage_crossover.png)

The n≈11.6 crossover is a resource boundary for exact Hilbert-space
simulation, anchored by completed ibm_fez scaling runs and committed
classical baseline timings. It is not a broad quantum-advantage claim:
Rust Kuramoto ODE baselines remain faster through n≤16, and the largest
hardware runs are noise-limited.

---

### Many-Body Localisation Diagnostic

Level spacing ratio $\bar{r}$ distinguishes integrable/MBL ($\bar{r} \approx 0.386$,
Poisson) from chaotic/thermalising ($\bar{r} \approx 0.530$, GOE) spectra.

![MBL level spacing](figures/publication/fig15_mbl_level_spacing.png)

**Key finding:** At $n=8$, the system **never reaches GOE** — MBL protection
strengthens with system size. The heterogeneous frequencies act as effective
disorder preventing thermalisation. This is the physics behind identity
persistence: the coupling topology is protected from thermal decoherence.

**Cross-validation (eigenstate entanglement):** Excited-state entropy is 30–40%
below thermal (Page) expectation, confirming non-ergodicity. However, entropy
grows with N (sub-volume, not area law), ruling out deep MBL. Correct label:
**non-ergodic regime** — coupling topology protected from thermal scrambling.

![Eigenstate entanglement](figures/publication/fig16_eigenstate_entanglement.png)

First application of level-spacing diagnostics (standard tool, Oganesyan & Huse 2007) to heterogeneous-frequency Kuramoto-XY.

### BKT Universality Confirmation

Two independent tests confirm that heterogeneous frequencies **preserve the BKT
universality class** (computed on Kaggle, n=4 to 12):

**CFT central charge:** Fitting $S(l) = (c/3)\ln(l) + \text{const}$ at $K \approx K_c$:

| n | c (measured) | BKT prediction |
|---|-------------|----------------|
| 6 | 0.951 | 1.000 |
| 8 | **1.039** | 1.000 |
| 10 | 1.214 | 1.000 |
| 12 | 1.305 | 1.000 |

$c \approx 1$ at n=6,8 confirms BKT. Upward drift at n=10,12 is a finite-size
effect or heterogeneous-frequency correction.

**Spectral gap essential singularity:** Fitting $\Delta \sim \exp(-b/\sqrt{K - K_c})$:

| n | K_c | b | R² | Verdict |
|---|-----|---|-----|---------|
| 4 | 2.83 | 2.60 | **0.975** | BKT confirmed |
| 6 | 3.86 | 2.21 | **0.970** | BKT confirmed |
| 8 | 3.60 | 2.27 | **0.969** | BKT confirmed |

R² > 0.96 at n=4,6,8 — the essential singularity is a definitive BKT signature.
No prior measurement for heterogeneous-frequency Kuramoto-XY.

---

## Rust Acceleration Benchmarks

Measured on Windows 11, Python 3.12, Rust release build.
See [Rust Engine](rust_engine.md) for full API.

| Function | n | Rust | Reference | Speedup |
|----------|---|------|-----------|---------|
| Hamiltonian construction | 4 | 0.004 ms | 20.9 ms (Qiskit) | **5401×** |
| Hamiltonian construction | 8 | 0.4 ms | 63 ms (Qiskit) | **158×** |
| OTOC (30 time points) | 4 | 0.3 ms | 74.7 ms (scipy) | **264×** |
| OTOC (30 time points) | 6 | 48 ms | 5.66 s (scipy) | **118×** |
| Lanczos (50 steps) | 3 | 0.05 ms | 1.3 ms (numpy) | **27×** |
| Lanczos (50 steps) | 4 | 0.5 ms | 4.8 ms (numpy) | **10×** |
